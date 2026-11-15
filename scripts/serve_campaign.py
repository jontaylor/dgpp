#!/usr/bin/env python3
"""Every tracked benchmark for one deployment template, in one command.

The rows docs/benchmarks.md keeps for a deployment all come from one boot of
the template as shipped (plus one boot without its drafter, and one per
extra decode mode asked for). This runner boots it on the bench port, runs
each measurement with the maintained clients, checks the run's validity,
tears the world down, and writes a record directory the `summarize` and
`rows` commands read back. Records stay out of Git (benchmarks/results is
ignored); the doc carries the cells.

  serve_campaign.py run --config deploy/cluster_<model>_<quant>_w<n>.example.json --out benchmarks/results/<label>
                        [--bin build-release/dgpp-serve] [--port 18081]
                        [--phases serving,prefill,context,quality,api,transcripts,plain]
                        [--mode NAME="<knobs>" ...] [--phase-timeout PHASE=SECONDS ...]
                        [--no-swap-scope] [--dry-run]
  serve_campaign.py run --config LOCAL_PAGING.json --out benchmarks/results/paging --compare-experts
                        [--comparison-repeats 2]  # A/B/B/A, only policy changes
  serve_campaign.py summarize DIR                     every cell, labelled
  serve_campaign.py rows DIR --label "Qwen3.8-27B FP8" --nodes 1 --options "DFlash2 drafter, ..."
                                                      the doc's table rows for this deployment

What one run measures (docs/benchmarks.md "Workload and timing" is the
definition of record):
  serving      timed_load at C1/C2/C4/C8 over the five prompt classes, also C16
               on a 16-slot deployment. Three repetitions, 256 tokens: the engine's decode-only rate per class
               and concurrency (the headline), the wall rate (prefill and
               queueing included) beside it.
  prefill      serve_prefill_probe at 512 / 2K / 8K / 32K, three distinct cold
               prefixes each, thinking off where supported (otherwise the
               recorded template default): cold prefill seconds.
  context      serve_context_matrix at 0 / 32K / 64K / 128K / 256K / 512K: cold
               prefill per bucket, decode ms a pass and tok/s from three requests.
  quality      serve_eval on the 300-item HumanEval / GSM8K / extraction set at
               the deployment's slot count; responses cut at the standard cap
               are asked once more at the 8,192 retry cap (recorded, not scored).
  api          serve_api_check: stop, n, logit_bias, usage.
  transcripts  serve_load C1, five classes x 3 greedy transcripts in the shipped
               mode, and the isolation check (prompt 0 alone, then beside
               slots-1 others).
  plain        a second boot of the template without its drafting keys (mtp
               false): the same transcripts, for the plain/default match.
  --mode       a boot per extra decode mode (knobs appended to every rank): the
               C1 serving sweep, for the decode-mode table.
Validity, recorded with the cells: the source revision and the binary's hash,
the resolved deployment, every rank's VmSwap (must stay 0) and binary hash
before and after, zero cached tokens in every cold probe, the ranks' operation
streams identical at teardown. Ranks run in the launcher's swap-limited scopes
(DGPP_NO_SWAP=1) unless --no-swap-scope.

"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import pathlib
import shlex
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLASSES = ("prose", "code", "json", "math", "chat")
BUCKETS = (0, 32768, 65536, 131072, 262144, 524288)
PREFILL_TARGETS = (512, 2048, 8192, 32768)
PHASES = ("serving", "prefill", "context", "quality", "api", "transcripts", "plain")


def workload(a):
    """Record focused diagnostics separately from the standard benchmark rows."""
    focused = getattr(a, "streaming_focused", False)
    if focused and set(a.phases.split(",")) != {"serving", "prefill"}:
        raise ValueError("--streaming-focused requires --phases serving,prefill")
    return {"profile": "streaming-focused" if focused else "standard",
            "warm_windows": 3 if focused else 1, "warm_window_max_tokens": 32,
            "serving_max_tokens": 64 if focused else 256,
            "serving_repeat": 2 if focused else 3,
            "serving_concurrency": "1,2,4" if focused else None,
            "prefill_targets": [512, 2048] if focused else list(PREFILL_TARGETS),
            "prefill_repeat": 2 if focused else 3}

PHASE_TIMEOUTS = {"serving": 3600, "prefill": 3600, "context": 21600,
                  "quality": 14400, "api": 3600, "transcripts": 3600,
                  "plain": 10800, "mode": 3600}


def phase_timeouts(overrides):
    """Resolve explicit client deadlines without changing the benchmark workload."""
    result = dict(PHASE_TIMEOUTS)
    seen = set()
    for spec in overrides or []:
        name, sep, seconds = spec.partition("=")
        if not sep or name not in result or not seconds.isascii() or not seconds.isdecimal():
            raise ValueError("phase timeout must be PHASE=SECONDS with a known phase and positive integer seconds")
        value = int(seconds)
        if value <= 0 or name in seen:
            raise ValueError("phase timeouts must be positive and specified once per phase")
        result[name] = value
        seen.add(name)
    return result


def quality_request_timeout(args):
    value = float(getattr(args, "quality_request_timeout", 900))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("quality request timeout must be finite and positive")
    return value


# ---- running ---------------------------------------------------------------------------------------------

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args, path, env, timeout=3600, check=True):
    """Run one command with its stdout+stderr in `path` and a receipt beside it."""
    receipt = {"command": [str(a) for a in args], "started": utcnow(), "timeout_seconds": timeout}
    print(receipt["started"][11:19], path.name, flush=True)
    with path.open("w") as f:
        try:
            r = subprocess.run(receipt["command"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            receipt["returncode"] = r.returncode
        except subprocess.TimeoutExpired:
            receipt["returncode"] = 124
    receipt["finished"] = utcnow()
    path.with_suffix(path.suffix + ".command.json").write_text(json.dumps(receipt, indent=2))
    if check and receipt["returncode"]:
        raise RuntimeError(f"{path.name} failed: {receipt['returncode']}")
    return receipt["returncode"]


def scoped_env(no_swap_scope):
    """The launcher's environment: the swap-limited scopes (DGPP_NO_SWAP=1 plus the user bus a non-login shell lacks)."""
    env = dict(os.environ)
    if no_swap_scope:
        env.pop("DGPP_NO_SWAP", None)
    else:
        env["DGPP_NO_SWAP"] = "1"
        uid = os.getuid()
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
    return env


def deployment_from(template, port):
    cfg = json.loads(pathlib.Path(template).read_text())
    cfg["http"] = {"bind_host": "127.0.0.1", "port": int(port)}
    return cfg


DRAFT_KEYS = ("dflash_model", "draft_head", "dflash_weights", "mtp_schedule", "mtp_schedule_row_ms", "mtp_schedule_base_ms",
              "mtp_schedule_lambda", "mtp_schedule_min_depth", "mtp_schedule_options", "mtp_schedule_adapt", "mtp_schedule_sampled_scale", "mtp_draft",
              "mtp_draft_temperature", "mtp_verify", "mtp_depth", "lookup_draft", "lookup_tail", "dflash_batch_rows", "dflash_depth")


def drafts(engine):
    """Whether this template drafts at all (a block drafter or the native MTP layer)."""
    return bool(engine.get("dflash_model") or engine.get("mtp"))


def plain_deployment(cfg):
    """The same deployment with its drafting switched off: the draft keys removed and mtp false, so the plain-decode
    boot is the template's own engine settings otherwise (a knob such as --no-dflash would leave the schedule keys
    behind, which the server refuses without a drafter)."""
    plain = json.loads(json.dumps(cfg))
    for k in DRAFT_KEYS:
        plain["engine"].pop(k, None)
    plain["engine"]["mtp"] = False
    return plain


PROBE = ("import hashlib,json,pathlib,sys;st=json.loads(pathlib.Path(sys.argv[1]).read_text());p=pathlib.Path('/proc')/str(st['pid']);"
         "s=dict(l.split(':',1) for l in (p/'status').read_text().splitlines() if ':' in l);"
         "print(json.dumps(dict(pid=st['pid'],sha=hashlib.sha256((p/'exe').read_bytes()).hexdigest(),"
         "swap_kib=int(s['VmSwap'].split()[0]),rss_kib=int(s['VmRSS'].split()[0]),hwm_kib=int(s['VmHWM'].split()[0]))))")


EXPERT_PROBE = r"""
import json, os, pathlib, sys, time, urllib.request
image_directory = os.path.expanduser(sys.argv[1])
result = {"sampled_unix": time.time(), "image_directory": image_directory,
          "io_attribution": "Whole block-device/partition counters include competing I/O; not expert-only attribution."}
try:
    dev = os.stat(image_directory).st_dev
    block = pathlib.Path('/sys/dev/block') / f'{os.major(dev)}:{os.minor(dev)}'
    values = [int(x) for x in (block / 'stat').read_text().split()]
    result['block'] = {"name": block.resolve().name, "read_operations": values[0],
                       "read_bytes": values[2] * 512, "read_milliseconds": values[3]}
except (OSError, ValueError, IndexError) as e:
    result['block'] = {"unknown": str(e)}
if sys.argv[2]:
    try:
        result['prometheus'] = urllib.request.urlopen(sys.argv[2], timeout=5).read().decode()
    except (OSError, ValueError) as e:
        result['prometheus_error'] = str(e)
else:
    result['prometheus_error'] = 'Peer metrics disabled; configure ports.metrics for all-rank counters.'
print(json.dumps(result))
"""


def source_fingerprint():
    """Include uncommitted source files in the experiment's source identity."""
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    names = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, check=True, capture_output=True).stdout.split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(set(names) - {b""}):
        path = ROOT / os.fsdecode(name)
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
        elif path.is_file():
            content = path.read_bytes()
        elif not path.exists():
            content = b"DELETED"
        else:
            raise ValueError(f"unsupported source entry: {path}")
        digest.update(name + b"\0" + hashlib.sha256(content).digest())
    return {"revision": revision, "tree_sha256": digest.hexdigest()}


class Boot:
    """One world: up on the bench port, the checks, down with the op-stream match."""

    def __init__(self, d, cfg, env, binary, knobs=""):
        self.d, self.cfg, self.env, self.binary, self.knobs = d, cfg, env, binary, knobs
        self.world = d / "world"
        self.config = d / "deployment.json"
        self.port = str(cfg["http"]["port"])
        self.host = "127.0.0.1"
        self.world_size = int(cfg.get("world_size") or 1)
        self.status = {}
        self.sha = hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()

    def launcher(self, verb, extra):
        return ["python3", "scripts/dgpp-cluster", verb, "--config", str(self.config), "--bin", self.binary, "--log-dir", str(self.world), *extra]

    def up(self):
        self.d.mkdir(parents=True, exist_ok=False)
        self.config.write_text(json.dumps(self.cfg, indent=2))
        (self.d / "binary.sha256").write_text(self.sha + "\n")
        (self.d / "source.txt").write_text(
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout)
        (self.d / "source-identity.json").write_text(json.dumps(source_fingerprint(), indent=2))
        extra = [f"--knobs={self.knobs}"] if self.knobs else []  # one token: a knob that starts with -- is otherwise read as the next option
        run(self.launcher("up", extra), self.d / "up.log", self.env, timeout=1800)
        resolved = json.loads((self.world / "cluster.resolved.json").read_text())
        self.world_size = len(resolved["nodes"])
        self.snapshot("before")
        run(["python3", "scripts/serve_bench.py", self.host, self.port, "32", "warm"], self.d / "warm.log", self.env)

    def snapshot(self, label):
        """Every rank's dgpp-serve: VmSwap (must be 0) and the binary's hash (must be the one under test)."""
        cfg = json.loads((self.world / "cluster.resolved.json").read_text())
        out = []
        for rank, host in enumerate(cfg["nodes"]):
            state = str(self.world / "rank0.process.json") if rank == 0 else cfg["paths"]["stage_dir"] + f"/rank{rank}.process.json"
            cmd = ["python3", "-c", PROBE, state] if rank == 0 else \
                ["ssh", "-o", "BatchMode=yes", cfg["ssh_user"] + "@" + host, "python3 -c " + shlex.quote(PROBE) + " " + state]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            if p.returncode:
                raise RuntimeError(f"rank {rank} snapshot failed: {p.stderr[-400:]}")
            out.append(dict(rank=rank, host=host, **json.loads(p.stdout)))
        paging = self.cfg.get("engine", {}).get("expert_streaming")
        if paging:
            samples = []
            for rank, host in enumerate(cfg["nodes"]):
                if rank == 0:
                    url = f"http://127.0.0.1:{self.port}/metrics/prometheus"
                else:
                    metrics_port = cfg.get("ports", {}).get("metrics", 0)
                    url = f"http://{host}:{metrics_port}/metrics/prometheus" if metrics_port else ""
                command = ["python3", "-c", EXPERT_PROBE, paging["image_directory"], url]
                if rank:
                    command = ["ssh", "-o", "BatchMode=yes", cfg["ssh_user"] + "@" + host, shlex.join(command)]
                sample = subprocess.run(command, text=True, capture_output=True, timeout=60)
                if sample.returncode:
                    raise RuntimeError(f"rank {rank} expert instrumentation failed: {sample.stderr[-400:]}")
                samples.append(dict(rank=rank, host=host, **json.loads(sample.stdout)))
            (self.d / f"expert-{label}.json").write_text(json.dumps(samples, indent=2))
        (self.d / f"nodes-{label}.json").write_text(json.dumps(out, indent=2))
        bad = [x for x in out if x["sha"] != self.sha or x["swap_kib"] != 0]
        if bad:
            raise RuntimeError(f"swap / binary check failed: {bad}")
        return out

    def down(self):
        if not (self.d / "up.log").exists():
            return
        if self.status.get("ok"):
            try:
                self.snapshot("after")
            except Exception as e:  # noqa: BLE001 — the teardown must still happen
                self.status["ok"] = False
                self.status["error"] = str(e)[-800:]
        run(self.launcher("down", []), self.d / "down.log", self.env, check=False)
        ops = sorted(self.world.glob("serve_rank*.ops"))
        hashes = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in ops}
        match = len(hashes) == self.world_size and len(set(hashes.values())) == 1
        (self.d / "opstreams.json").write_text(json.dumps({"hashes": hashes, "expected_ranks": self.world_size, "match": match}, indent=2))
        self.status["opstreams_match"] = match


EXPERT_POLICIES = ("whole_expert_cache", "adaptive_fractional_cache")


def compare_experts(a):
    selected_workload = workload(a)
    """Alternate paired boots from one frozen template, varying only policy.

    This schedules ordinary serving campaigns. Synthetic routing locality and
    physical block-device attribution require their own evidence and are not
    inferred from these rows.
    """
    timeouts = phase_timeouts(getattr(a, "phase_timeout", []))
    quality_timeout = quality_request_timeout(a)
    cfg = deployment_from(a.config, a.port)
    engine = cfg.get("engine", {})
    paging = engine.get("expert_streaming")
    if not isinstance(paging, dict) or paging.get("policy") not in EXPERT_POLICIES:
        raise ValueError("expert comparison requires engine.expert_streaming in the template")
    if "initial_fraction" in paging:
        raise ValueError("omit initial_fraction for comparison: only policy may differ between legs")
    if engine.get("decode_graph", False) or engine.get("prefill_fold_overlap", True) or drafts(engine):
        raise ValueError("expert comparison requires eager execution, no fold overlap and no drafts")
    if a.mode:
        raise ValueError("expert comparison does not combine policy and decode-mode changes")
    if not 1 <= a.comparison_repeats <= 20:
        raise ValueError("comparison repeats must be in [1, 20]")
    shared = copy.deepcopy(cfg)
    del shared["engine"]["expert_streaming"]["policy"]
    frozen = json.dumps(shared, sort_keys=True, separators=(",", ":"))
    order = [policy for repeat in range(a.comparison_repeats)
             for policy in (EXPERT_POLICIES if repeat % 2 == 0 else EXPERT_POLICIES[::-1])]
    plan = {"shared_config_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
            "template": str(pathlib.Path(a.config).resolve()), "started": utcnow(),
            "order": order, "phases": a.phases, "workload": selected_workload,
            "phase_timeouts_seconds": timeouts, "quality_request_timeout_seconds": quality_timeout, "legs": [],
            "limitations": ["Natural serving routes; no controlled locality replay.",
                            "Block counters include competing I/O; expert-only physical attribution requires an idle-device control.",
                            "Each leg performs the campaign warmup; not a cold-cache mechanism benchmark."]}
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    plan["source"] = source_fingerprint()
    plan["binary_sha256"] = hashlib.sha256(pathlib.Path(a.bin).read_bytes()).hexdigest()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / "comparison.json"
    receipt.write_text(json.dumps(plan, indent=2))
    try:
        for i, policy in enumerate(order):
            if source_fingerprint() != plan["source"] or \
                    hashlib.sha256(pathlib.Path(a.bin).read_bytes()).hexdigest() != plan["binary_sha256"]:
                raise RuntimeError("source or binary changed during the policy comparison")
            leg_cfg = copy.deepcopy(cfg)
            leg_cfg["engine"]["expert_streaming"]["policy"] = policy
            config = out / f"input-{i:02d}-{policy}.json"
            config.write_text(json.dumps(leg_cfg, indent=2))
            leg = copy.copy(a)
            leg.compare_experts = False
            leg.config = str(config)
            leg.out = str(out / f"leg-{i:02d}-{policy}")
            leg.label = a.label or pathlib.Path(a.config).stem
            result = run_campaign(leg)
            if source_fingerprint() != plan["source"] or \
                    hashlib.sha256(pathlib.Path(a.bin).read_bytes()).hexdigest() != plan["binary_sha256"]:
                raise RuntimeError("source or binary changed during a policy leg")
            status = json.loads((pathlib.Path(leg.out) / "status.json").read_text())
            valid = result == 0 and status.get("ok") and status.get("opstreams_match")
            plan["legs"].append({"policy": policy, "directory": leg.out,
                                  "returncode": result, "valid": bool(valid)})
            receipt.write_text(json.dumps(plan, indent=2))
            if not valid:
                raise RuntimeError(f"expert comparison leg {i} failed its campaign or operation-stream gate")
        plan["ok"] = True
    except Exception as e:
        plan.update(ok=False, error=str(e))
    plan["finished"] = utcnow()
    receipt.write_text(json.dumps(plan, indent=2))
    return 0 if plan["ok"] else 1


def run_campaign(a):
    if getattr(a, "compare_experts", False):
        return compare_experts(a)
    selected_workload = workload(a)
    timeouts = phase_timeouts(getattr(a, "phase_timeout", []))
    quality_timeout = quality_request_timeout(a)
    env = scoped_env(a.no_swap_scope)
    out = pathlib.Path(a.out)
    cfg = deployment_from(a.config, a.port)
    engine = cfg["engine"]
    if getattr(a, "streaming_focused", False) and not engine.get("expert_streaming"):
        raise ValueError("--streaming-focused requires expert streaming")
    phases = set(a.phases.split(","))
    unknown = phases - set(PHASES)
    if unknown:
        raise SystemExit(f"unknown phases {sorted(unknown)}; known: {', '.join(PHASES)}")
    slots = int(engine.get("max_concurrency", 8))
    label = a.label or pathlib.Path(a.config).name.replace(".example.json", "").replace(".json", "")
    status = {"config": a.config, "label": label, "started": utcnow(), "phases": sorted(phases),
              "workload": selected_workload,
              "phase_timeouts_seconds": timeouts, "quality_request_timeout_seconds": quality_timeout}
    port = str(a.port)
    kw = dict(binary=a.bin)
    server_timings = ["--server-timings"] if engine.get("expert_streaming") else []

    boot = Boot(out, cfg, env, **kw)
    host = boot.host
    if a.dry_run:
        print("phase timeouts (seconds):", json.dumps(timeouts, sort_keys=True))
        print("quality HTTP read timeout (seconds):", quality_timeout)
        print("workload:", json.dumps(selected_workload, sort_keys=True))
        print("launcher:", " ".join(shlex.quote(str(c)) for c in boot.launcher("up", [])))
        print(f"clients against {host}:{port}; phases {sorted(phases)}; plain boot {'yes' if drafts(engine) else 'not needed'}; modes {a.mode or []}")
        return 0
    try:
        boot.up()
        if selected_workload["warm_windows"] > 1:
            # Fixed, identical exposure under both policies. Preserve each
            # window's counters; a fixed warmup is not proof of convergence.
            boot.snapshot("warm-0")
            for window in range(1, selected_workload["warm_windows"]):
                run(["python3", "scripts/serve_bench.py", host, port,
                     str(selected_workload["warm_window_max_tokens"]), f"warm-{window}"],
                    out / f"warm-{window}.log", env)
                boot.snapshot(f"warm-{window}")
        if "serving" in phases:
            concurrencies = selected_workload["serving_concurrency"] or ("1,2,4,8,16" if slots >= 16 else "1,2,4,8")
            run(["python3", "scripts/timed_load.py", host, port, "--concurrency", concurrencies, "--classes", "all",
                 "--max-tokens", str(selected_workload["serving_max_tokens"]), "--repeat", str(selected_workload["serving_repeat"]),
                 *server_timings, "--json-out", out / "decode.json"], out / "decode.log", env, timeout=timeouts["serving"])
        if "prefill" in phases:
            run(["python3", "scripts/serve_prefill_probe.py", host, port, *map(str, selected_workload["prefill_targets"]),
                 "--repeat", str(selected_workload["prefill_repeat"]),
                 "--no-think", "--tag", f"campaign-{label}-{status['started'][:19]}", "--json-out", out / "prefill.json"],
                out / "prefill.log", env, timeout=timeouts["prefill"])
        if "context" in phases:
            py = ROOT / ".venv/bin/python"
            run([str(py) if py.exists() else "python3", "scripts/serve_context_matrix.py", host, port, "--model", cfg["model"],
                 "--tag", f"ctx-{label}-{status['started'][:19]}", "--json-out", out / "context.json"], out / "context.log", env,
                timeout=timeouts["context"])
        if "quality" in phases:
            run(["python3", "scripts/serve_eval.py", host, port, "--out", out / "eval", "--tasks", "humaneval,gsm8k,extract",
                 "--limit", "300", "--concurrency", str(slots), "--allow-code-execution",
                 "--request-timeout", str(quality_timeout), "--data", "data"], out / "eval.log", env,
                timeout=timeouts["quality"])
        if "api" in phases:
            from cluster_doctor import cache_root, cached_snapshot
            from site_env import cache_environment
            tokenizer = cached_snapshot(cfg["model"], cache_root(cache_environment())) / "tokenizer.json"
            run(["python3", "scripts/serve_api_check.py", host, port, "--tokenizer", tokenizer],
                out / "api.log", env, timeout=timeouts["api"], check=False)
        if "transcripts" in phases:
            run(["python3", "scripts/serve_load.py", host, port, "--concurrency", "1", "--classes", "all", "--repeat", "3",
                 "--max-tokens", "256", "--json-out", out / "transcripts_default.json"], out / "transcripts_default.log", env, timeout=timeouts["transcripts"])
            # The isolation check exits nonzero when the texts differ: a result, not a failure.
            run(["python3", "scripts/serve_load.py", host, port, "--concurrency", "1", "--repeat", "1", "--max-tokens", "256",
                 "--isolation", str(slots), "--json-out", out / "isolation.json"], out / "isolation.log", env, timeout=timeouts["transcripts"], check=False)
        boot.status["ok"] = True
    except Exception as e:  # noqa: BLE001 — record, tear down, continue to the other boots
        boot.status["ok"] = False
        boot.status["error"] = str(e)[-800:]
        print("BOOT FAILED", label, boot.status["error"], flush=True)
    finally:
        boot.down()
    status.update(boot.status)

    if "plain" in phases:
        if drafts(engine):
            plain = Boot(out / "plain", plain_deployment(cfg), env, **kw)
            try:
                plain.up()
                run(["python3", "scripts/serve_load.py", host, port, "--concurrency", "1", "--classes", "all", "--repeat", "3",
                     "--max-tokens", "256", "--json-out", out / "plain" / "transcripts_plain.json"], out / "plain" / "transcripts_plain.log",
                    env, timeout=timeouts["plain"])
                plain.status["ok"] = True
            except Exception as e:  # noqa: BLE001
                plain.status["ok"] = False
                plain.status["error"] = str(e)[-800:]
                print("PLAIN BOOT FAILED", label, plain.status["error"], flush=True)
            finally:
                plain.down()
            status["plain"] = {"deployment": "the template without its drafting keys, mtp false", **plain.status}
        else:
            status["plain"] = {"skipped": "the template drafts nothing: its shipped mode is plain decode"}

    for spec in a.mode or []:
        name, _, knobs = spec.partition("=")
        mode = Boot(out / "modes" / name, cfg, env, knobs=knobs, **kw)
        try:
            mode.up()
            run(["python3", "scripts/timed_load.py", host, port, "--concurrency", "1", "--classes", "all", "--max-tokens", "256",
                 "--repeat", "3", "--json-out", out / "modes" / name / "decode.json"], out / "modes" / name / "decode.log", env, timeout=timeouts["mode"])
            mode.status["ok"] = True
        except Exception as e:  # noqa: BLE001
            mode.status["ok"] = False
            mode.status["error"] = str(e)[-800:]
            print("MODE BOOT FAILED", name, mode.status["error"], flush=True)
        finally:
            mode.down()
        status.setdefault("modes", {})[name] = {"knobs": knobs, **mode.status}

    status["finished"] = utcnow()
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print("CAMPAIGN DONE", json.dumps({k: v for k, v in status.items() if k in ("label", "ok", "opstreams_match", "plain", "modes")}))
    return 0 if status.get("ok") else 1


# ---- reading the records ---------------------------------------------------------------------------------

def span(d):
    return f"{min(d.values()):.1f}–{max(d.values()):.1f}"


def serving_cells(d):
    """Per class and concurrency: the engine's decode-only rate and the wall rate (medians of the repetitions)."""
    phases = json.loads((d / "decode.json").read_text())["phases"]
    cs = sorted({p["concurrency"] for p in phases})
    decode = {n: {c: statistics.median(p["engine"]["tokens_per_s"] for p in phases if p["class"] == c and p["concurrency"] == n)
                  for c in CLASSES} for n in cs}
    wall = {n: {c: statistics.median(p["metrics"]["wall_tokens_per_s"] for p in phases if p["class"] == c and p["concurrency"] == n)
                for c in CLASSES} for n in cs}
    return dict(concurrencies=cs, decode=decode, wall=wall)


def prefill_cells(d):
    cold = json.loads((d / "prefill.json").read_text())["samples"]
    targets = sorted({s["requested_tokens"] for s in cold})
    return dict(prefill={n: statistics.median(s["prefill_ms"] for s in cold if s["requested_tokens"] == n) / 1000 for n in targets},
                actual={n: sorted(s["usage"]["prompt_tokens"] for s in cold if s["requested_tokens"] == n) for n in targets},
                cached=sum(s["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for s in cold))


def context_cells(d):
    rep = json.loads((d / "context.json").read_text())
    out = {}
    for n in BUCKETS:
        s = [x for x in rep["samples"] if x["requested_tokens"] == n]
        if not s:
            out[n] = None
            continue
        cold = [x for x in s if x["repeat"] == 0][0]
        out[n] = dict(prefill=cold["engine"]["prefill_ms"] / 1000,
                      ms_per_pass=statistics.median(x["ms_per_pass"] for x in s),
                      tok_s=statistics.median(x["engine_tokens_per_s"] for x in s),
                      cached_cold=cold["answer"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))
    return dict(buckets=out, limit=next((k["limit"] for k in rep.get("skipped", [])), None))


def eval_cells(d):
    lines = (d / "eval.log").read_text().strip().splitlines()
    return json.loads([l for l in lines if l.startswith("{")][-1])


def api_cells(d):
    lines = (d / "api.log").read_text().splitlines()
    ok = sum(1 for l in lines if l.startswith("ok  "))
    return ok, ok + sum(1 for l in lines if l.startswith("FAIL"))


def class_texts(path):
    rep = json.loads(path.read_text())
    out = {}
    for ph in rep["phases"]:
        if ph["concurrency"] == 1:
            out.setdefault(ph["class"], []).extend((r["text"], r["tokens"]) for r in ph["requests"])
    return out


def transcript_cells(d):
    """Plain/default greedy match per class (every repetition in both modes identical) and the isolation verdict."""
    default = class_texts(d / "transcripts_default.json")
    plain_path = d / "plain" / "transcripts_plain.json"
    plain = class_texts(plain_path) if plain_path.exists() else None
    matched = 0
    detail = {}
    for cls in CLASSES:
        xs = default.get(cls, []) + (plain.get(cls, []) if plain else [])
        same = plain is not None and len(xs) == 6 and len(set(xs)) == 1
        matched += same
        detail[cls] = dict(default_distinct=len(set(default.get(cls, []))),
                           plain_distinct=len(set(plain.get(cls, []))) if plain else None, n=len(xs))
    iso_log = (d / "isolation.log").read_text() if (d / "isolation.log").exists() else ""
    iso_line = next((l for l in iso_log.splitlines() if l.startswith("== isolation")), None)
    return dict(matched=matched if plain is not None else None, detail=detail, plain_measured=plain is not None,
                isolation=None if iso_line is None else ("different" if "transcript isolation failed" in iso_log else "identical"),
                isolation_line=iso_line)


def mode_cells(d):
    out = {}
    for md in sorted((d / "modes").glob("*/decode.json")) if (d / "modes").exists() else []:
        cells = serving_cells(md.parent)
        out[md.parent.name] = cells["decode"][1]
    return out


def validity(d):
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    swaps = {}
    for f in ("nodes-before.json", "nodes-after.json", "plain/nodes-after.json"):
        if (d / f).exists():
            swaps[f] = [x["swap_kib"] for x in json.loads((d / f).read_text())]
    return dict(status=st, swaps=swaps,
                source=(d / "source.txt").read_text().strip()[:12] if (d / "source.txt").exists() else None,
                binary=(d / "binary.sha256").read_text().strip()[:16] if (d / "binary.sha256").exists() else None)


def summarize(d):
    d = pathlib.Path(d)
    v = validity(d)
    st = v["status"]
    print(f"== {d.name}: source {v['source']} binary {v['binary']} ok={st.get('ok')} opstreams={st.get('opstreams_match')} swap={v['swaps']}")
    if (d / "decode.json").exists():
        s = serving_cells(d)
        print("  decode tok/s (engine, decode steps only), range across classes: " + "  ".join(f"C{n} {span(s['decode'][n])}" for n in s["concurrencies"]))
        print("  wall tok/s (prefill and queueing included):                       " + "  ".join(f"C{n} {span(s['wall'][n])}" for n in s["concurrencies"]))
        print("  C1 decode by class: " + ", ".join(f"{c} {s['decode'][1][c]:.1f}" for c in CLASSES))
    if (d / "prefill.json").exists():
        p = prefill_cells(d)
        print("  cold prefill s: " + " / ".join(f"{p['prefill'][n]:.3f}" for n in sorted(p["prefill"])) + f"  (targets {sorted(p['prefill'])}, cached tokens {p['cached']}, actual {p['actual']})")
    if (d / "context.json").exists():
        c = context_cells(d)
        print("  context: " + " | ".join(f"{n}: pf {b['prefill']:.1f} s, {b['ms_per_pass']:.2f} ms/pass, {b['tok_s']:.1f} tok/s, cold cached {b['cached_cold']}" if b else f"{n}: Limit"
                                          for n, b in c["buckets"].items()))
    if (d / "eval.log").exists():
        e = eval_cells(d)
        print("  quality: " + ", ".join(
            f"{k} {x['passed']}/{x['n']} (cap {x['truncated']}" +
            (f", retried {x['retry_completed']} completed / {x['retry_passed']} pass" if x.get('retried') else "") + ")"
            for k, x in e.items()))
    if (d / "api.log").exists():
        print("  api checks: %d/%d" % api_cells(d))
    if (d / "transcripts_default.json").exists():
        t = transcript_cells(d)
        match = f"{t['matched']}/5 classes" if t["plain_measured"] else "not measured (no plain boot)"
        print(f"  plain/default greedy match: {match}; isolation: {t['isolation']}")
    for name, cells in mode_cells(d).items():
        print(f"  mode {name}: C1 decode {span(cells)} ({', '.join(f'{c} {cells[c]:.1f}' for c in CLASSES)})")


def rows(d, label, nodes, options):
    """The doc's rows for this deployment, in table order; the author fills the model label and options once."""
    d = pathlib.Path(d)
    status_path = d / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("workload", {}).get("profile", "standard") != "standard":
        raise ValueError("focused streaming diagnostics cannot produce standard benchmark rows")
    head = f"| {label} | {nodes} | {options} |"
    cfg = json.loads((d / "deployment.json").read_text()) if (d / "deployment.json").exists() else {}
    e = cfg.get("engine", {})
    out = []
    if (d / "decode.json").exists():
        s = serving_cells(d)
        pf = prefill_cells(d)["prefill"] if (d / "prefill.json").exists() else {}
        out.append(("serving", f"{head} " + " | ".join(span(s["decode"][n]) for n in (1, 2, 4, 8))
                    + " | " + (" / ".join(f"{pf[n]:.3f}" for n in (2048, 8192, 32768)) if pf else "—") + " |"))
        out.append(("wall", f"{head} " + " | ".join(span(s["wall"][n]) for n in (1, 2, 4, 8)) + " |"))
        out.append(("classes", f"{head} " + " | ".join(f"{s['decode'][1][c]:.1f}" for c in CLASSES) + " |"))
        for n in s["concurrencies"]:
            if n > 8:
                out.append(("serving-extra", f"{head} {n} | {span(s['decode'][n])} |"))
                out.append(("wall-extra", f"{head} {n} | {span(s['wall'][n])} |"))
    if e:
        if e.get("dflash_model"):
            mode = "DFlash2 adaptive ≤7" if e.get("mtp_schedule") else "DFlash2 fixed 7"
        elif e.get("mtp"):
            mode = f"MTP{e.get('mtp_depth', 1)}" + (" scheduled" if e.get("mtp_schedule") else "")
        else:
            mode = "plain"
        if e.get("prefill_budget_tokens", -1) < 0:
            idle = e.get("prefill_idle_budget_tokens", 0)
            budget = f"adaptive / idle cap {idle}" if idle else "adaptive (default)"
        else:
            budget = f"{e['prefill_budget_tokens']} / {e.get('prefill_idle_budget_tokens', e['prefill_budget_tokens'])}"
        out.append(("config", f"{head} {e.get('max_concurrency')} | {int(e.get('kv_capacity', 0)):,} | {e.get('kv_dtype', 'bf16')} | "
                    f"{float(e.get('prefix_cache_gib', 1.5)):g} | {budget} | {mode} | [JSON](<template path>) |"))
    if (d / "context.json").exists():
        c = context_cells(d)
        b = c["buckets"]
        cells = []
        for n in BUCKETS:
            cells += [f"{b[n]['ms_per_pass']:.2f}", f"{b[n]['tok_s']:.1f}"] if b[n] else ["Limit", "Limit"]
        limit = f"{c['limit']:,}" if c["limit"] else "—"
        out.append(("context", f"{head} {limit} | " + (f"{b[32768]['prefill']:.3f}" if b.get(32768) else "—") + " | " + " | ".join(cells) + " |"))
        out.append(("context-prefill", f"{head} " + " | ".join(f"{b[n]['prefill']:.3f}" if b[n] else "Limit" for n in BUCKETS) + " |"))
    if (d / "eval.log").exists():
        ev = eval_cells(d)
        # The last column: of the responses at the standard cap, how many
        # completed when asked once more at the larger cap, and how many of
        # those passed (serve_eval.py --retry-max-tokens; "—" before 2026-10-22).
        retried = sum(x.get("retried", 0) for x in ev.values())
        beyond = (f"{sum(x['retry_completed'] for x in ev.values())} ({sum(x['retry_passed'] for x in ev.values())} pass)"
                  if retried else ("0" if sum(x['truncated'] for x in ev.values()) == 0 else "—"))
        out.append(("quality", f"{head} {ev['humaneval']['passed']}/{ev['humaneval']['n']} | {ev['gsm8k']['passed']}/{ev['gsm8k']['n']} | "
                    f"{ev['extract']['passed']}/{ev['extract']['n']} | {sum(x['truncated'] for x in ev.values())} | {beyond} |"))
    if (d / "transcripts_default.json").exists() and (d / "api.log").exists():
        t = transcript_cells(d)
        ok, total = api_cells(d)
        match = f"{t['matched']}/5 classes" if t["plain_measured"] else "—"
        out.append(("transcripts", f"{head} {ok}/{total} | {match} | {t['isolation'] or '—'} |"))
    m = mode_cells(d)
    if m:
        out.append(("modes", f"{head} " + " | ".join(f"{name}: {span(cells)}" for name, cells in m.items()) + " |"))
    for table, row in out:
        print(f"-- {table}\n{row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run", help="one template's whole row set")
    r.add_argument("--config", required=True, help="the deployment template (deploy/*.json)")
    r.add_argument("--out", required=True, help="the record directory (new; benchmarks/results/<label> is ignored by Git)")
    r.add_argument("--bin", default=str(ROOT / "build-release/dgpp-serve"), help="the binary under test")
    r.add_argument("--port", type=int, default=18081)
    r.add_argument("--label", default=None)
    r.add_argument("--phases", default=",".join(PHASES))
    r.add_argument("--streaming-focused", action="store_true",
                   help="explicit diagnostic profile: serving C1/C2/C4, five classes, 2 repeats, 64-token cap; cold prefill 512/2K twice; requires --phases serving,prefill; not standard benchmark rows")
    r.add_argument("--quality-request-timeout", type=float, default=900,
                   help="HTTP read timeout per quality reply, including retries; independent of phase deadline (default 900 seconds)")
    r.add_argument("--phase-timeout", action="append", metavar="PHASE=SECONDS",
                   help="override a phase client timeout (repeatable); same deadlines for both policies; workload unchanged")
    r.add_argument("--mode", action="append", metavar='NAME="<knobs>"', help="an extra decode-mode boot: the C1 sweep under these knobs")
    r.add_argument("--no-swap-scope", action="store_true", help="do not run the ranks in the launcher's swap-limited scopes")
    r.add_argument("--compare-experts", action="store_true",
                   help="paired policy campaigns from one expert-streaming template; alternating A/B then B/A")
    r.add_argument("--comparison-repeats", type=int, default=2,
                   help="policy pairs with alternating order (default 2 yields A/B/B/A)")
    r.add_argument("--dry-run", action="store_true", help="print the launcher command for this leg and exit")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    w = sub.add_parser("rows")
    w.add_argument("dir")
    w.add_argument("--label", required=True)
    w.add_argument("--nodes", required=True)
    w.add_argument("--options", required=True)
    a = ap.parse_args()
    if a.verb == "run":
        sys.exit(run_campaign(a))
    elif a.verb == "summarize":
        summarize(a.dir)
    else:
        rows(a.dir, a.label, a.nodes, a.options)


if __name__ == "__main__":
    main()
