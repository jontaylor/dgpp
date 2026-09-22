# MiMo tool-format repair, 2026-09-22

Superseded by the [required-argument repair](2026-09-22-mimo-required-tools.md),
which also covers the actual Astroid harness request missed by these probes.

The directory-listing request reproduced malformed tool XML with thinking both
on and off. Both automatic-tool requests hit the 256-output-token limit without
returning a structured tool call. The previous forced-weather-call acceptance
check was too narrow to establish usable automatic tool calling.

## Cause and change

MiMo and Qwen share outer tool markers and function/parameter tag names, but
MiMo's published template writes compact tags. DGPP selected the Qwen grammar
from those shared markers and forced Qwen's structural newlines. The grammar
then failed to recognize a parameter closer without its prescribed newline
prefix (including an empty value immediately after the header), keeping the
free-value state open and suppressing the outer tool-call closer. This explains
the repeated closing tags observed in the reproduction.

MiMo's tokenizer-specific audio sentinel now selects compact XML delimiters in
the grammar and parser. Qwen retains its existing newline delimiters. A token
that would close a free argument and swallow bytes from the next tag is masked,
so every following byte is governed by the next grammar state. Compact string
arguments preserve leading/trailing newlines; those are value data, unlike
Qwen's structural separators. No inference kernel or weight change is involved.

## Validation

31 tool parser/grammar tests pass, including compact empty strings, typed and
enum arguments, delimiter-crossing tokens, and exact multiline string parsing.
40 host serving tests pass. MiMo and Qwen tokenizer differential checks each
pass 94 cases, including selection of compact formatting only for MiMo.

Evidence, reproduction requests and acceptance scripts:
`/tmp/dgpp-mimo-tools-20260922/`. The fixed binary SHA256 is
`2f6b388dc68d2443f757728dcc1318e5a3461908da31c838d5c28aac669196ba`.

Live automatic `list_directory` now returns `{"path":"."}` with
`finish_reason=tool_calls` in all four thinking on/off and streaming/non-streaming
combinations. All four follow-up turns receive a real read-only repository
listing and correctly confirm that CMakeLists.txt is present. No XML leaks into
content. First-call times were 2.431 seconds with thinking and 0.840 without;
streamed repeat times were 1.269/0.840 seconds (these are not matched cold timing
comparisons). Concurrent `src` and empty-path requests both produce structurally
valid calls, with correct `src` arguments.

The non-strict empty-string semantic test fails: even explicit zero-character
wording produces a string containing two quote characters. We preserve this
failed result in `accept.log`/`supplement.log`; no parser quote-stripping workaround
is used, because that would alter legitimate string values. Schema enforcement
with `enum:[""]` produces the exact empty string. Integer/boolean arguments and
an enum string containing leading/trailing newlines also pass live checks.
Thus the malformed-format regression is fixed, but this is not a claim that all
tool argument choices or broader model quality are correct. Full-model HF
numerical equivalence remains unverified.

Final health is OK and metrics are idle with zero failed requests and no engine
failure. Both rank hashes and effective settings match. Deployment namespace
is `3b107de62acc3684`, configuration digest `dfd06bf77fbc5019`, PIDs
3542726 / 2309795. READY was 16:27:45 UTC. The service remains running.

## Deployment

Both Sparks retain 262,144 tokens per request, two concurrent requests, exactly
four snapshots (11.297607421875 GiB budget), CUDA graphs, MTP off and prefill
budgets 256/2048. API port remains 30001.

Config and binary on Spark-1 are
`/home/jon/dgpp/experimental/mimo-tool-format-20260922/deployment.json` and
`/home/jon/dgpp/experimental/mimo-tool-format-20260922/dgpp-serve`.
Use these with the cluster launcher documented in the 256K deployment record.
Prior binaries/configurations are preserved for rollback. Changes remain
uncommitted.
