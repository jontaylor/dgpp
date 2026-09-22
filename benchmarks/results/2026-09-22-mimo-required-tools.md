# MiMo required tool arguments, 2026-09-22

The supplied Astroid request reproduces the user's failure exactly on the
compact-format binary: reasoning says "Let me explore the workspace to
understand what repo we're in", followed by `list_directory({})` and `glob({})`.
The manually reconstructed request's messages/tools match the recorded request
byte-for-byte after JSON decoding. The original harness also observed this in
non-streaming mode, excluding SSE assembly as the cause.

Original evidence supplied by the user:
`/mnt/benchmarks/swebench-suite/preparations/20260922T163128Z-direct-mimo-astroid-retry/`.
A portable copy of the model/messages/tools request is retained at
`tests/data/mimo/tool_call_request.json`.

## Correction

The grammar recorded `parameters.required` but only blocked premature closure
when a tool also declared `strict: true`. The harness's valid tool schemas do
not set strict. MiMo now requires every declared required field before closing
a call, even for non-strict tools. This does not promote unsupported schema
keywords to strict validation or change other families' existing non-strict
behavior. Tools with no required fields (including run_tests) can still produce
an empty argument object. No defaults are invented and no completed response is
rewritten to hide missing arguments.

32 tool parser/grammar tests pass. The new regression checks empty and partial
calls with multiple required fields for open/closed property sets, plus valid
zero-required-parameter calls. Existing Qwen/GLM/DSML grammar tests still pass.
The preceding implementation's 40 host serving tests and tokenizer checks are
recorded in the earlier tool-format report; they are not represented as a new
full host run for this follow-up.

## Multi-turn boundary failure and correction

The required-fields-only candidate repaired the exact first turn, but a six-turn
harness continuation exposed corrupted file_path values at turn three. It
produced eight calls with required keys, of which only four executed successfully;
the other four included XML markup in file_path. The initial diagnostic printed
PASS based on too-weak counters; its raw log is retained as `harness.log`, but
this run is explicitly a failure, not acceptance evidence.

The previous compact-format patch rejected tokens spanning a parameter close
and the next tag. That forced alternative tokens and could make closing markup
part of a raw string argument. The corrected compact XML path validates each
token byte across as many grammar states as it spans. Free strings, names, keys,
JSON values and enum values can cross delimiters; every subsequent field still
obeys its schema and required-key ledger. The mask considers native cross-state
tokens instead of forcing artificial token boundaries. Other tool formats retain
their existing path.

Regression coverage includes tokens spanning entire multi-field calls,
JSON-to-enum boundaries, rejection of unknown keys and missing required fields
inside a single token, and MiMo's actual tokenization of the failing read_file
shape. All 32 tool tests and both 94-case tokenizer suites pass. The final
harness gate checks every call's required fields, rejects leaked markup in
path/pattern arguments and requires every executed read-only tool to succeed.

## Deployment and evidence

Binary SHA256:
`3a4e09c0b40310d6e4d0a9a48fdbc615ca3873ec11b93a8d747da5bca5fdb661`.
Config and binary on Spark-1 live under
`/home/jon/dgpp/experimental/mimo-required-boundary-20260922/`.
The configuration retains port 30001, 262,144 tokens, two concurrent requests,
four snapshots, graphs enabled, MTP off and 256/2048 prefill budgets.
Previous deployments are preserved. All working changes remain uncommitted.

Current diagnostic scripts, build/test logs and replay outputs are under
`/tmp/dgpp-mimo-required-20260922/`.
The actual harness integration uses ChatCompletionsClient -> RecordingTransport
-> UrllibModelTransport, with temperature 0, max_tokens 8192 and streaming,
without tool-choice, parallel-call or template overrides. Its first request is
asserted equal to the supplied model-input-001.json. Later turns are confined
to read-only inspection using the harness's RepositoryTools against a separate
Astroid v2.9.1 checkout at e36b4388aebb892c10ed779bafc9c1a07a325b8d.


## Final live acceptance

The exact recorded first request passed through the actual Python harness
client/recording/streaming transport. Six turns produced **12 tool calls**, all
with their required fields and all successfully executed by RepositoryTools
against Astroid: directory listing, glob, searches and file reads, including
separate numeric offset/limit arguments. No path/pattern contained leaked XML.
Raw final evidence: `harness-boundary.log` and `harness-boundary-after/` under
the diagnostic root. This passes the stronger gate that the intermediate
required-only candidate failed.

Three additional streamed requests through the same client correctly populated
run_shell_command (`command=pwd`), read_file (`file_path=astroid/modutils.py`,
`offset=0`, `limit=20`) and write_file (`file_path=notes.txt`, `content=hello`).
These three validate arguments only; generated shell commands and writes were
not executed. Raw: `arguments.log` and `arguments-after/`.

Both ranks have matching final binary hashes and config digest
`dfd06bf77fbc5019`. Namespace `738d67e7124214a7`, PIDs 3546466 / 2315448,
READY 16:53:18 UTC. Health is OK; final metrics show no active/queued requests,
zero request failures, no engine failure and four snapshot slots. Both logs
have no ERROR/FATAL entries. The service remains running on port 30001.

This validates the reported empty-argument regression and a bounded real
multi-turn tool journey, not a completed SWE-bench repair or general model
quality/full-model numerical equivalence. The Astroid investigation was limited
to reading; no source fixes were attempted in that checkout.

Restart on Spark-1 under the maintenance lock:

```bash
DGPP_ENV_FILE=/home/jon/dgpp/releases/dgpp-0.1.0+gf1b05f544293.dirty/.env \
python3 /home/jon/dgpp/releases/dgpp-0.1.0+g116d3c375d87/scripts/dgpp-cluster up \
  --config /home/jon/dgpp/experimental/mimo-required-boundary-20260922/deployment.json \
  --bin /home/jon/dgpp/experimental/mimo-required-boundary-20260922/dgpp-serve
```
