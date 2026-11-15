#!/usr/bin/env python3
"""serve_campaign.py's readers on a synthetic record directory: the decode-only
and wall rates per class and concurrency, the cold-prefill medians, the
context cells, the quality summary, the API tally, the plain/default match
and the isolation verdict, and the plain-knobs derivation."""
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import serve_campaign as sc  # noqa: E402


def phase(cls, c, engine_rate, wall_rate, text="x"):
    return {"class": cls, "concurrency": c, "engine": {"tokens_per_s": engine_rate},
            "metrics": {"wall_tokens_per_s": wall_rate}, "requests": [{"text": text, "tokens": 256}] * c}


def write_records(d):
    phases = []
    for cls_i, cls in enumerate(sc.CLASSES):
        for c in (1, 2, 4, 8):
            # three repetitions whose median is 10 * c + cls_i (engine) and 9 * c + cls_i (wall)
            for rep_rate in (-1, 0, 1):
                phases.append(phase(cls, c, 10 * c + cls_i + rep_rate, 9 * c + cls_i + rep_rate))
    (d / "decode.json").write_text(json.dumps({"phases": phases}))
    samples = []
    for n in sc.PREFILL_TARGETS:
        for ms in (n - 1, n, n + 1):
            samples.append({"requested_tokens": n, "prefill_ms": ms, "usage": {"prompt_tokens": n + 5, "prompt_tokens_details": {"cached_tokens": 0}}})
    (d / "prefill.json").write_text(json.dumps({"samples": samples}))
    ctx = []
    for n in (0, 32768):
        for r in range(3):
            ctx.append({"requested_tokens": n, "repeat": r, "engine": {"prefill_ms": 1000 * (n // 32768 + 1)},
                        "ms_per_pass": 100 + r, "engine_tokens_per_s": 20 - r,
                        "answer": {"usage": {"prompt_tokens": n or 100, "prompt_tokens_details": {"cached_tokens": 0 if r == 0 else n}}}})
    (d / "context.json").write_text(json.dumps({"samples": ctx, "skipped": [{"target": 65536, "reason": "exceeds request limit", "limit": 40000}]}))
    (d / "eval.log").write_text("== humaneval: 160/164\n" + json.dumps({"humaneval": {"n": 164, "passed": 160, "truncated": 2},
                                                                         "gsm8k": {"n": 300, "passed": 294, "truncated": 1},
                                                                         "extract": {"n": 100, "passed": 100, "truncated": 0}}) + "\n")
    (d / "api.log").write_text("ok   stop: fine\nok   n: fine\nFAIL usage: missing\n")
    default = {"phases": [phase(cls, 1, 1, 1, text=f"{cls}-answer") for cls in sc.CLASSES for _ in range(3)]}
    (d / "transcripts_default.json").write_text(json.dumps(default))
    (d / "plain").mkdir()
    plain_phases = [phase(cls, 1, 1, 1, text=f"{cls}-answer" if cls != "chat" else "chat-other") for cls in sc.CLASSES for _ in range(3)]
    (d / "plain" / "transcripts_plain.json").write_text(json.dumps({"phases": plain_phases}))
    (d / "isolation.log").write_text("== isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT\ntranscript isolation failed\n")
    (d / "status.json").write_text(json.dumps({"ok": True, "opstreams_match": True}))
    (d / "nodes-before.json").write_text(json.dumps([{"rank": 0, "swap_kib": 0}]))
    (d / "source.txt").write_text("0123456789abcdef\n")
    (d / "binary.sha256").write_text("f" * 64 + "\n")
    (d / "deployment.json").write_text(json.dumps({"engine": {"max_concurrency": 8, "kv_capacity": 786432, "kv_dtype": "fp8",
                                                              "prefix_cache_gib": 20, "dflash_model": "x", "mtp_schedule": True}}))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        write_records(d)
        s = sc.serving_cells(d)
        assert s["concurrencies"] == [1, 2, 4, 8]
        assert s["decode"][1]["prose"] == 10 and s["decode"][8]["chat"] == 84, s["decode"]
        assert s["wall"][4]["json"] == 38, s["wall"]
        assert sc.span(s["decode"][2]) == "20.0–24.0"
        p = sc.prefill_cells(d)
        assert p["prefill"][2048] == 2.048 and p["cached"] == 0, p
        c = sc.context_cells(d)
        assert c["buckets"][0]["ms_per_pass"] == 101 and c["buckets"][0]["tok_s"] == 19, c["buckets"][0]
        assert c["buckets"][32768]["prefill"] == 2.0 and c["buckets"][65536] is None and c["limit"] == 40000
        e = sc.eval_cells(d)
        assert e["gsm8k"]["passed"] == 294
        assert sc.api_cells(d) == (2, 3)
        t = sc.transcript_cells(d)
        assert t["matched"] == 4 and t["detail"]["chat"]["plain_distinct"] == 1 and t["isolation"] == "different", t
        assert sc.drafts({"dflash_model": "z"}) and sc.drafts({"mtp": True}) and not sc.drafts({"mtp": False})
        plain = sc.plain_deployment({"engine": {"mtp": False, "dflash_model": "z", "mtp_schedule": True,
                                               "mtp_schedule_options": 3, "kv_capacity": 4}})["engine"]
        assert plain == {"mtp": False, "kv_capacity": 4}, plain
        # without a plain boot the match is not a result
        import shutil
        shutil.move(str(d / "plain"), str(d / "plain.off"))
        t2 = sc.transcript_cells(d)
        assert t2["matched"] is None and t2["plain_measured"] is False
        shutil.move(str(d / "plain.off"), str(d / "plain"))
        v = sc.validity(d)
        assert v["source"] == "0123456789ab" and v["swaps"]["nodes-before.json"] == [0]
        # The row writer runs end to end and names every table it has records for.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.rows(d, "Model X", 1, "opts")
        out = buf.getvalue()
        for table in ("serving", "wall", "classes", "config", "context", "context-prefill", "quality", "transcripts"):
            assert f"-- {table}\n| Model X | 1 | opts |" in out, table
        assert "| 786,432 | fp8 | 20 | adaptive (default) | DFlash2 adaptive ≤7 |" in out, out
        assert "-- serving-extra" not in out and "-- wall-extra" not in out
        # Wider measurements retain the common C1-C8 rows and report their
        # decode and service-level rates separately, with the configured cap.
        data = json.loads((d / "decode.json").read_text())
        for cls_i, cls in enumerate(sc.CLASSES):
            for offset in (-1, 0, 1):
                data["phases"].append(phase(cls, 16, 160 + cls_i + offset, 144 + cls_i + offset))
        (d / "decode.json").write_text(json.dumps(data))
        cfg = json.loads((d / "deployment.json").read_text())
        cfg["engine"].update(max_concurrency=16, prefill_idle_budget_tokens=2048)
        (d / "deployment.json").write_text(json.dumps(cfg))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.rows(d, "Model X", 1, "opts")
        out = buf.getvalue()
        assert "-- serving-extra\n| Model X | 1 | opts | 16 | 160.0–164.0 |" in out, out
        assert "-- wall-extra\n| Model X | 1 | opts | 16 | 144.0–148.0 |" in out, out
        assert "| adaptive / idle cap 2048 |" in out, out
        # Exercise the runner through command construction: the one-command
        # campaign must actually produce the additional cells it can publish.
        class FakeBoot:
            host = "127.0.0.1"
            def __init__(self, directory, *args, **kwargs):
                self.directory, self.status = directory, {}
            def up(self):
                self.directory.mkdir()
            def snapshot(self, label):
                pass
            def down(self):
                self.status["opstreams_match"] = True
        for slots, wanted in ((8, "1,2,4,8"), (16, "1,2,4,8,16")):
            cfg["engine"]["max_concurrency"] = slots
            (d / "deployment.json").write_text(json.dumps(cfg))
            args = SimpleNamespace(config=str(d / "deployment.json"), out=str(d / f"run-{slots}"),
                                   port=18081, phases="serving", no_swap_scope=True, label="fixture",
                                   bin="fixture-binary", dry_run=False, mode=None)
            with patch.object(sc, "Boot", FakeBoot), patch.object(sc, "run") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                assert sc.run_campaign(args) == 0
            command = run.call_args.args[0]
            assert command[command.index("--concurrency") + 1] == wanted, command
            assert command[command.index("--repeat") + 1] == "3", command
        # Reject bad deadlines before creating a boot or touching hardware.
        for override in ("serving=0", "serving=-1", "serving=1.5", "other=10",
                         "serving=inf", "serving", "serving=１２"):
            args.phase_timeout = [override]
            with patch.object(sc, "Boot") as boot:
                try:
                    sc.run_campaign(args)
                    raise AssertionError(f"accepted invalid timeout {override}")
                except ValueError:
                    pass
                boot.assert_not_called()
        args.phase_timeout = ["serving=10", "serving=20"]
        try:
            sc.run_campaign(args)
            raise AssertionError("accepted duplicate timeout")
        except ValueError:
            pass
        # Extended deadlines reach the clients; benchmark requests remain standard.
        cfg["model"] = "fixture"
        cfg["engine"]["expert_streaming"] = {"policy": "whole_expert_cache"}
        (d / "deployment.json").write_text(json.dumps(cfg))
        args.out = str(d / "extended-run")
        args.phases = "serving,prefill,context,quality,transcripts"
        args.phase_timeout = ["serving=28800", "transcripts=7200"]
        args.quality_request_timeout = 7200
        with patch.object(sc, "Boot", FakeBoot), patch.object(sc, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            assert sc.run_campaign(args) == 0
        calls = {call.args[1].name: call for call in run.call_args_list}
        for name, deadline in (("decode.log", 28800), ("prefill.log", 3600),
                               ("context.log", 21600), ("eval.log", 14400),
                               ("transcripts_default.log", 7200), ("isolation.log", 7200)):
            assert calls[name].kwargs["timeout"] == deadline, name
        quality_command = calls["eval.log"].args[0]
        assert quality_command[quality_command.index("--request-timeout") + 1] == "7200.0"
        command = calls["decode.log"].args[0]
        assert "--server-timings" in command
        assert command[command.index("--max-tokens") + 1] == "256"
        assert command[command.index("--repeat") + 1] == "3"
        status = json.loads((pathlib.Path(args.out) / "status.json").read_text())
        assert status["phase_timeouts_seconds"]["serving"] == 28800
        assert status["quality_request_timeout_seconds"] == 7200
        # The bounded streaming profile preserves the default suite and cannot
        # accidentally publish shorter runs as standard benchmark rows.
        args.streaming_focused = True
        args.phases = "serving,prefill"
        args.out = str(d / "focused-run")
        with patch.object(sc, "Boot", FakeBoot), patch.object(sc, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            assert sc.run_campaign(args) == 0
        calls = {call.args[1].name: call for call in run.call_args_list}
        assert "warm-1.log" in calls and "warm-2.log" in calls
        decode_command = calls["decode.log"].args[0]
        assert decode_command[decode_command.index("--max-tokens") + 1] == "64"
        assert decode_command[decode_command.index("--repeat") + 1] == "2"
        assert decode_command[decode_command.index("--concurrency") + 1] == "1,2,4"
        prefill_command = calls["prefill.log"].args[0]
        assert prefill_command[4:6] == ["512", "2048"]
        try:
            sc.rows(args.out, "fixture", 2, "focused")
            raise AssertionError("focused diagnostics published as standard rows")
        except ValueError:
            pass
        args.phases = "serving,quality"
        with patch.object(sc, "Boot") as boot:
            try:
                sc.run_campaign(args)
                raise AssertionError("focused profile accepted quality phase")
            except ValueError:
                pass
            boot.assert_not_called()
        comparison_template = d / "comparison-template.json"
        comparison_cfg = {"model": "fixture", "world_size": 2, "engine": {
            "decode_graph": False, "prefill_fold_overlap": False, "mtp": False,
            "expert_streaming": {"policy": "whole_expert_cache", "budget_bytes": 123456,
                                  "image_directory": "/ssd/fixture"}}}
        comparison_template.write_text(json.dumps(comparison_cfg))
        binary = d / "fixture-binary"
        binary.write_bytes(b"immutable executable")
        args = SimpleNamespace(config=str(comparison_template), out=str(d / "comparison"),
                               port=18081, phases="serving,prefill", no_swap_scope=False, label="fixture",
                               bin=str(binary), dry_run=False, mode=None, compare_experts=True,
                               comparison_repeats=2, phase_timeout=["serving=28800"], quality_request_timeout=7200)
        observed = []
        def leg_run(leg):
            assert sc.phase_timeouts(leg.phase_timeout)["serving"] == 28800
            assert sc.quality_request_timeout(leg) == 7200
            observed.append(json.loads(pathlib.Path(leg.config).read_text()))
            path = pathlib.Path(leg.out)
            path.mkdir()
            (path / "status.json").write_text(json.dumps({"ok": True, "opstreams_match": True}))
            return 0
        with patch.object(sc, "run_campaign", side_effect=leg_run):
            assert sc.compare_experts(args) == 0
        policies = [x["engine"]["expert_streaming"].pop("policy") for x in observed]
        assert policies == [sc.EXPERT_POLICIES[0], sc.EXPERT_POLICIES[1],
                            sc.EXPERT_POLICIES[1], sc.EXPERT_POLICIES[0]], policies
        assert all(x == observed[0] for x in observed), "only policy may vary"
        receipt = json.loads((d / "comparison/comparison.json").read_text())
        assert receipt["phase_timeouts_seconds"]["serving"] == 28800
        assert receipt["quality_request_timeout_seconds"] == 7200
        assert receipt["ok"] and len(receipt["legs"]) == 4 and receipt["source"]["tree_sha256"]
        # A failed operation-stream gate stops comparison before another boot.
        args.out = str(d / "invalid-comparison")
        def bad_leg(leg):
            result = leg_run(leg)
            (pathlib.Path(leg.out) / "status.json").write_text(json.dumps({"ok": True, "opstreams_match": False}))
            return result
        with patch.object(sc, "run_campaign", side_effect=bad_leg) as mocked:
            assert sc.compare_experts(args) == 1 and mocked.call_count == 1
        # A changing binary invalidates the leg rather than silently comparing builds.
        args.out = str(d / "mutating-comparison")
        def mutating_leg(leg):
            result = leg_run(leg)
            binary.write_bytes(b"replacement executable")
            return result
        with patch.object(sc, "run_campaign", side_effect=mutating_leg) as mocked:
            assert sc.compare_experts(args) == 1 and mocked.call_count == 1
        args.dry_run = True
        args.out = str(d / "dry-comparison")
        with contextlib.redirect_stdout(io.StringIO()):
            assert sc.compare_experts(args) == 0
        assert not pathlib.Path(args.out).exists(), "dry-run must not launch or write a campaign"
        comparison_cfg["engine"]["expert_streaming"]["initial_fraction"] = 0.5
        comparison_template.write_text(json.dumps(comparison_cfg))
        try:
            sc.compare_experts(args)
            raise AssertionError("a B-only setting would violate the single-variable comparison")
        except ValueError:
            pass
    # The shipped image path uses ~, interpreted on each serving host. The
    # device probe must resolve that same path before choosing block counters.
    buf = io.StringIO()
    with patch.dict(os.environ, {"HOME": "/home/probe-fixture"}), \
            patch.object(sys, "argv", ["probe", "~/.cache/dgpp/experts", ""]), \
            patch.object(os, "stat", return_value=SimpleNamespace(st_dev=os.makedev(259, 2))) as stat, \
            patch.object(pathlib.Path, "resolve", return_value=pathlib.Path("/sys/devices/fixture/nvme0n1p2")), \
            patch.object(pathlib.Path, "read_text", return_value="7 0 11 13"), \
            contextlib.redirect_stdout(buf):
        exec(sc.EXPERT_PROBE, {})
    stat.assert_called_once_with("/home/probe-fixture/.cache/dgpp/experts")
    probe = json.loads(buf.getvalue())
    assert probe["image_directory"] == "/home/probe-fixture/.cache/dgpp/experts"
    assert probe["block"] == {"name": "nvme0n1p2", "read_operations": 7,
                              "read_bytes": 5632, "read_milliseconds": 13}, probe
    print("serve_campaign_test: ok")


if __name__ == "__main__":
    main()
