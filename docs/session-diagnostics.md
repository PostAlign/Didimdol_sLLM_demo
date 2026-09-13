# Full-session loading diagnostics

The September 12 phone exports from release `02719f4a4081` (application commit
`722cd56`) passed synthetic residency, OPFS residency, tokenizer plus OPFS
residency, and the small runtime with 120 seconds of observation. Session-only
loading interrupted after 209/251 initializers; evaluation-page loading stopped
after 247/251. Both used 8 MiB staging, nine OPFS cache hits and Asyncify, with no
persisted GPU fault. These observations do not establish an OS-memory kill.

## September 13 phone exports (release `d5e1043f51ac`, commit `673290b`)

Same iPhone (iOS 26.6.2, Chrome 153), 8 MiB staging, Asyncify, resident
execution, nine OPFS cache hits, no persisted GPU fault in any run.

| UTC | Screen / experiment | Outcome |
|---|---|---|
| 05:00:56 | Evaluation page session create | 251/251, 1,023 MiB, `ready` |
| 05:01:09 | Evaluation page warmup inference | interrupted at 2.0 s, 1,059 MiB GPU requested, page back after 0.5 s |
| 05:01:34-05:07:22 | 1b, 1c (two tokenizer paths), 1d, small runtime 120 s, tokenizer | all complete |
| 05:07:48 | 2c session-only | interrupted at 207/251 (`embed_tokens.chunk11`), 813.9 MiB completed / 853.9 MiB allocated, page back after 0.8 s |
| 05:09:28 | 3 full load | 251/251, 1,023 MiB, `ready` |

2c, 3 and the evaluation page share one load path and produced an identical ORT
initializer order (2c's 208 entries are a prefix of the other two), the same
upload rate (about 420 MiB/s), the same 40 MiB allocate-to-complete backlog and
the same pending-check peak. The only code difference is that 2c skips the
tokenizer, so 2c used less memory than the two runs that survived. Neither
interruption left a `pagehide`/`visibilitychange` event, and both recovering
pages appeared under one second after the last checkpoint, which matches a
process termination rather than a navigation. Across the September 12 and 13
exports the interrupted run changes each time (2c at 209, evaluation at 247,
full load at 188, 2c at 207, warmup at 1,059 MiB) while every stop lies between
740 MiB and 1,060 MiB of GPU-resident weights. Resident FP32 therefore runs at
the device's limit with no headroom for warmup; which run crosses it depends on
system state outside the page. Device Jetsam/WebContent logs from the same
minutes are still required to confirm the kill. Recovery now records
`unloadEvidence`, `reentryGapMs` and `navigationType` so this reasoning no
longer has to be reconstructed from exports by hand. The next comparisons are
three 2c repeats at 8 MiB, then 2c and full load with `modelExecution=streamed`
(about 423 MiB resident), with exports and device logs kept from the same runs.

## September 13 afternoon exports (release `7f6d4c99f9ac`, commit `3ecd6ce`)

Same iPhone, same settings, nine OPFS cache hits, no persisted GPU fault, no
device loss. Recovery on every interrupted run recorded `no-unload-event`, a
re-entry gap under 1.2 s and `navigationType: back_forward`.

| UTC | Screen / experiment | Outcome |
|---|---|---|
| 05:30:17 | Evaluation page session create | 251/251, 1,023 MiB, `ready` in 12.1 s |
| 05:30:30 | Evaluation page warmup (greedy, 4 tokens) | complete in 2.0 s |
| 05:30:32 | Evaluation page row 1 (sampled) | **failed in code** at 0.6 s before the first token, see below |
| 05:33:14-05:36:24 | 1, 1b, 1c, 1d, small runtime 120 s, 2b | all complete |
| 05:36:43 | 2c session-only | interrupted at 214/251 (`embed_tokens.chunk9`), 901 MiB allocated, page back after 1.1 s |
| 05:37:26 | 3 full load | 251/251, `ready` in 8.3 s |
| 05:38:07 | 4 cached load 1/5 | 251/251, `ready` in 9.0 s |
| 05:38:16 | 4 cached load 2/5 | interrupted at 188/251 (`embed_tokens.chunk0`), 782 MiB, page back after 0.7 s |
| 05:38:47 | 5 short inference | load 251/251 `ready` in 8.7 s, then interrupted 47 ms after `probe-inference-start`, page back after 0.4 s |

The three experiment interruptions repeat the morning pattern: identical
initializer order (each interrupted order is a prefix of the successful loads),
the same upload rate, and stops between 782 MiB and 1,023 MiB of resident FP32
weights. 2c reached the same initializer about 1.1 s later than the successful
full load despite skipping the tokenizer, which is compatible with system memory
pressure but does not prove it. Nothing in these runs points at the page's code.

The evaluation-page failure was different: the export held only a stack,
`phase@range-loader.js:77:37` called from ORT's `createSession`, with no
message, because JavaScriptCore stacks omit the message and the worker stored
`error.stack`. The message was `Session phase after loader close`. With
`do_sample: true` and `top_k: 64` from `generation_config.json`, transformers.js
creates a tiny ONNX `TopK` session on the first sampled token
(`TensorOpRegistry.top_k`, 73 bytes, no initializers). That creation passes
through the same `__ortExternalTensorLoader.phase('ort-session-start')` hook as
the model session, and the sealed range loader refused it. Warmup and the short
probes use greedy decoding, so they never reached this path; the browser smoke
test only ran greedy probes; and the phone had never survived warmup before, so
the check added in `884ad11` stayed latent until this run.

Allowing the phase exposed a second, older gap on desktop (SwiftShader): the
mobile runtime is a reduced-operator build generated from the deployment graph
(`model/required-operators.config`), and that graph contains no `TopK`, so the
auxiliary session then failed with `Could not find an implementation for
TopK(11)`. Sampled evaluation rows therefore could not have run on any device
with the mobile build; the smoke test's greedy probes never noticed.

Changes:

- **Auxiliary sessions after seal.** A sealed `SessionRangeLoader` now records
  ORT lifecycle phases from later sessions as `auxiliary-session-start`,
  `auxiliary-session-phase` (`ortPhase` carries the native phase) and
  `auxiliary-session-complete`, with an `auxiliarySession` ordinal and the
  sealed model metrics. The model session's `ort-*` milestones are never
  overwritten. Weight reads and initializer allocations after seal still throw.
  Abort, device loss and GPU errors are still checked first.
- **Warmup samples.** The evaluation warmup now uses the same generation settings
  as the rows (sampling, 4 tokens), so the top_k session is created inside the
  discarded warmup instead of inside row 1's TTFT. The seed is restored after
  warmup as before.
- **Error text.** Failure summaries and `fatal` messages store the message
  followed by the stack (`errorText`), so Safari exports name the cause.
- **TopK kernel.** `tools/create_ort_config.py` now appends `ai.onnx;21;TopK`
  (the CPU kernel registered for opsets 11-23) so the reduced mobile build can
  create the transformers.js top_k session. The native runtime has to be rebuilt
  (`tools/build-ort.sh`); CI's runtime cache key includes the config file.
- **Sampled probe.** The `probe` message accepts `sampled: true`, which adds one
  sampled token on the shortest prompt after the two greedy outputs
  (`sampledOutput`). The app-path smoke test uses it and asserts exactly one
  auxiliary session with no change to the GPU ledger.

This follow-up adds a comparison and lowers diagnostic persistence overhead.
FP32 model weights, tokenizer, generation settings, evaluation inputs and graph
optimization settings are unchanged. Native changes add lifecycle checkpoints;
they do not change buffer allocation, packing or execution policy.

## September 13 evening exports (release `963a5d7bbf22`, commit `925de82`)

Same iPhone, same settings, nine OPFS cache hits in every run, no persisted GPU
fault, no device loss. Six runs were interrupted; every one recorded
`no-unload-event`, `navigationType: back_forward` and a re-entry gap between
0.4 s and 4.2 s. The experiment page was exported after each run
(`iphone-fp32-experiments (35)`–`(45)`), the evaluation page once
(`didimdol-diagnostics (9)`).

| UTC | Screen / experiment | Outcome |
|---|---|---|
| 07:20:30 | Evaluation page session create | 251/251, `ready` in 15.3 s |
| 07:20:30 | Evaluation page warmup (sampled, 4 tokens) | complete in 1.5 s, one auxiliary `top_k` session, peak 1,069 MiB requested |
| 07:20:32 | Evaluation page row 1 (sampled) | interrupted 0.6 s after the first token, 1,063 MiB requested, page back after 1.3 s |
| 07:21:08-07:24:34 | 1, 1b, 1c, 1d, small runtime 120 s, 2b | all complete |
| 07:31:04 | 2c session-only, resident | interrupted at 225/251 (`embed_tokens.chunk3`, 4 of 5 staging pieces), 988 MiB allocated, page back after 1.6 s |
| 07:31:39 | 2c session-only, streamed | complete, 423 MiB, 6.7 s |
| 07:32:24 | 3 full load, resident | 251/251, `ready` in 9.3 s |
| 07:32:55 | 3 full load, streamed | 235/235, `ready` in 6.8 s |
| 07:33:37 | 4 cached load 1/5, resident | 251/251, `ready` in 9.5 s |
| 07:33:49 | 4 cached load 2/5, resident | interrupted at 209/251 (`onnx::MatMul_7184`, `range-read`), 917 MiB, page back after 0.8 s |
| 07:34:18, 07:34:26 | 4 cached load 1/5, 2/5, streamed | both `ready` (6.5 s, 7.7 s) |
| 07:34:30 | 4 cached load 3/5, streamed | **interrupted at `ort-plan-start` with no weights loaded**, 0 MiB GPU, page back after 4.2 s |
| 07:35:14 | 5 short inference, resident 8 MiB | load 251/251 `ready` in 9.1 s, interrupted within 0.4 s of `probe-inference-start` |
| 07:35:50 | 5 short inference, resident 2 MiB | load 251/251 `ready` in 9.3 s, interrupted within 0.8 s of `probe-inference-start` |
| 07:36:57 | 5 short inference, streamed | complete in 41.4 s, peak 498 MiB requested |

2c and the successful 3 load were compared directly: 2c's 226-entry initializer
order is a prefix of the 251-entry order (so is the interrupted cached load's
210), the first 225 initializers (948 MiB) uploaded in 2.28 s against 2.24 s and
2.14 s for the successful loads, and 2c's ORT phases (session start to plan,
plan to initializers) were no slower. 2c reached the same initializer 1.4 s
later only because its pre-ORT preparation was slower (evaluation input 808 ms
against 7 ms, graph 2.4 s). The cleanup of every preceding run had succeeded
with zero live buffers. Nothing distinguishes 2c's code path from the load that
survived; this is the third session in a row (207, 214, 225 of 251) in which 2c
was interrupted and 3 completed.

Resident inference did not survive in this session: the evaluation warmup
needed 47 MiB above the 1,023 MiB of weights and finished, row 1 then ended
after its first token, and both short-inference probes ended before their first
500 ms sample. The streamed path finished both probes (32 tokens each, 18.1 s
and 16.7 s, about 1.8 tokens/s, about 670 MB of weights uploaded per token) with
a 498 MiB peak.

The streamed cached load 3/5 does not fit the resident-limit explanation. It
ended between `ort-plan-start` and the first initializer, with no GPU weights,
as the third run in one page with 0.3-0.5 s between runs (the resident cached
load ended on the second such run). Whether something accumulates inside the
renderer process across runs cannot be read from exports: `jsMemory` is null
on iOS, and until this commit the streamed body loader reported a 0-byte WASM
heap before its first weight read because the ORT bridge had handed the heap
accessor to the head-session placeholder.

Changes in this commit, all instrumentation and documentation:

- **Streamed WASM heap.** `SessionRangeLoader` accepts `getHeap` at
  construction; the streamed body loader inherits it from the head placeholder,
  so `ort-session-start` and `ort-plan-start` records carry the heap size.
- **Run context.** Each experiment run's `environment.runContext` (also on the
  result row) records how many results were appended since the last interrupted
  row, the gap since the previous run ended, and the previous run's kind,
  execution mode, outcome and GPU request. Rows now store `endedAt` and
  `lastRecordAt`.
- **Repeat delay.** The experiment page can wait 0, 10 or 30 s between repeats
  (`repeatDelaySeconds` in the environment). The wait is shown and can be
  cancelled with the stop button; nothing is scheduled until it elapses.
- **Device log correlation.** Both exports carry `deviceClock` (time zone,
  offset, local export time); experiment results carry `startedAtLocal` and
  `lastRecordAtLocal`; `diagnosticSummary` exposes `lastRecordAt`. Each result
  row has a device-log note (file name, `reason`, footprint from MiB/GB or
  `rpages` × 16 KiB) that is stored with the row and exported as `deviceLog`.

### Collecting the device log

On the phone: Settings → Privacy & Security → Analytics & Improvements →
Analytics Data lists `JetsamEvent-YYYY-MM-DD-HHMMSS.ips` files (device local
time). If the list holds no JetsamEvent files, turn on Share iPhone Analytics
before reproducing. In the file, find `com.apple.WebKit.WebContent` (Chrome on
iOS runs pages in WKWebView), read its `reason` (`per-process-limit` versus
`vm-pageshortage`) and `rpages × pageSize`. If there is no JetsamEvent at the
minute of an interruption but a `com.apple.WebKit.WebContent` crash file exists,
keep that instead: WebKit's own memory-limit termination is reported there.
A Mac shows the same files in Xcode's Devices window or Console's Crash
Reports; a sysdiagnose (both volume buttons and the side button for 1.5 s) adds
system memory state. The evening interruptions, on a KST phone, ended at
16:20:32, 16:31:04, 16:33:49, 16:34:30, 16:35:14 and 16:35:50.

Open decisions that wait for that log: whether the evaluation page's iOS
default should become `modelExecution=streamed` (resident inference completed
0 of 3 attempts here, streamed 1 of 1), and whether the streamed cached load
3/5 recurs with a 30 s repeat delay. The acceptance run for the streamed path
is two 100-row evaluations; at about 17 s per row each pass takes close to
30 minutes on this device.

## Experiment 1d: small ORT session plus stored-weight residency

`runtime-resident` runs through the application worker, under the same origin
preparation lock as full loading. It creates and executes the existing small
FP32 fixture twice to instantiate the actual selected ORT WASM runtime, then
keeps that session alive while verifying every production weight using the
existing integer checksum probe. All buffers use the actual ORT GPUDevice.
The production weights must already be verified in OPFS; this comparison never
downloads or substitutes them. The small diagnostic fixture has its own store.

The retained small model contributes 10,496,000 bytes of external weights plus
its internal buffers. `runtimeMetrics`, `runtimeStorage` and
`runtimeGpuBeforeResidency` describe that component. Top-level weight counters
and `storage` describe the production residency probe. `gpuLedger` includes
both components; `runtimeDeviceId`, `residentDeviceId` and `sameDevice` verify
device sharing. `smallSessionRetained` and `ortWasmInstantiated` are required
for the distinct `runtime-and-gpu-residency` completion scope. This experiment
does not create a Gemma session, run Gemma inference, or measure WASM-only cost.

The resident probe destroys only its own buffers when borrowing the device.
The runtime owner releases the small session, destroys its device and restores
tracking. Its reusable loading scratch is released before residency starts.

## Session phases and GPU counters

The pinned patch records WASM initialization, ORT environment initialization,
entry/return of native session creation, allocation planning, initializer
loading, kernel creation and prepacking (when enabled). These markers use the
existing Asyncify/JSPI suspension mechanism at lifecycle boundaries. GPU API
wrappers remain synchronous where the API is synchronous; asynchronous pipeline
creation returns its original promise. The additive native diagnostic version
is 1; the external-range ABI remains 2. Building from an older factory fails
explicitly and requires rebuilding the native runtime.

Phases raised after the model session is sealed (transformers.js's lazily created
`top_k` session, for example) are recorded as `auxiliary-session-*` records and do
not touch the model session's milestones.

`gpuLedger.categories` separates production weights, runtime fixture weights,
verification buffers and other requests. It reports requested bytes, creation
counts and explicit release counts. The last 16 allocation descriptors retain
size, usage, stage and initializer context. Shader and pipeline call counts have
their own coverage status; these counts do not measure compilation memory.
Late-observed GPU buffers still produce partial allocation tracking.

These are logical requests and explicit destruction, not physical residency or
RSS. Garbage collection and native reference release without an explicit
`GPUBuffer.destroy()` are not measured. Do not infer a leak from a live count
alone. Original faults and cleanup results are recorded separately.

## Incremental diagnostic storage

Schema 4 defaults to `compact`. The existing IndexedDB store holds an immutable
run header and journal entries. Each checkpoint atomically commits its head,
ring slot and changed metadata in one transaction. The caller still waits for
transaction completion before the next guarded allocation/read/write. Failed
transactions retain metadata deltas for retry. Synchronous `put()` errors abort
the entire transaction. Persistence failures remain visible and cannot promise
durable recovery until a later write succeeds.

Recent history has 64 slots, each capped at 16,384 JSON characters. Oversized
history entries carry `historyTruncated`; the full latest event and original
fault remain separately available. Initializer order is capped at 2,048 entries.
Terminal summaries are stored once rather than repeated in recent events.
`readRun()` reconstructs the usual export shape and also reads schema 2/3 files
and snapshot records. Cleanup metadata uses a separate key so it does not replace
the interrupted position or original fault.

The experiment screen's diagnostic-storage selector supports `compact` and
`snapshot` within the same release. The evaluation page accepts
`?diagnosticsMode=snapshot`; the default is `compact`. Snapshot mode retains the
current instrumentation and record bounds while rewriting the entire run state,
allowing comparison of storage strategy without changing the runtime build.
It is not an exact recreation of the previous release.

## Inference sampling and durable experiment results

The September 13 phone exports (release `16e0a0baa98a`, commit `08bf38f`)
created the full session with 251/251 weights and then reopened the page about
three seconds after `warmup-start`. Nothing was recorded between the start of the
first `generate()` call and the termination, and the evaluation page described
the event only as an interrupted evaluation. Two changes address this:

- **`inference-sample` records.** While `generate()` runs (warmup, evaluation
  rows, short probes), the worker samples the GPU ledger, program counts and the
  WASM heap every 500 ms between event-loop turns. A record is written when a
  growth signal changes (observed GPU peak, program counts, WASM heap size, a
  device loss or error), at least every five seconds as a heartbeat, and once at
  the first generated token. Current bytes and buffer counts are recorded but do
  not trigger writes, because they move on every decode step. The sampler never
  waits on the GPU queue and does not touch ORT state. The `warmup-complete`, `row-complete`,
  `probe-inference-complete` and failure summaries carry an `inference` summary
  (sample counts, duration and the last digest). `recoveryEvidence` classifies a
  run whose last record is `warmup-start`, `row-start`, `probe-inference-start`
  or `inference-sample` as `interruptedPhase: 'inference'` with the phase and row;
  `diagnosticSummary` exposes `interruptedPhase`, `interruptedInference`,
  `inference` (the last sample) and `lastInference`. Both pages show the phase
  and, when a sample exists, the GPU request, pipeline count and WASM heap at
  that moment. These remain observations of process-internal counters, not a
  measurement of the OS memory limit.
- **Durable experiment results.** The diagnostic page keeps its results table and
  device settings in `localStorage`
  (`didimdol.device-experiments.results.v3`), so closing the browser after an
  interruption no longer discards the rows before export. `active` and
  `continue` stay in the tab's `sessionStorage`, so a second tab cannot recover
  another tab's running experiment. A legacy single-tab state migrates once;
  when durable storage refuses writes, the full state falls back to the tab.
  The newest result renders first and interrupted rows are highlighted. An empty
  device field is prefilled with the OS and browser versions parsed from the
  user agent; the device model still has to be typed.

Local verification (SwiftShader, `npm run test:streamed`, 2-token probes on the
35- and 266-token prompts): the resident probe produced 26 samples in 31.9 s with
a peak GPU request of 1,160,940,768 B against 1,072,392,704 B of weights and 65
shader modules; the streamed probe peaked at 516,625,696 B. Desktop SwiftShader
numbers do not stand in for iPhone memory behaviour; they show what the records
contain when the phone is interrupted during warmup.

## Cleanup and acceptance

Session release, OPFS closure and queue completion share a four-second cleanup
deadline. Device destruction and tracking restoration are attempted even if
release fails or stalls. Application-worker cancellation allows cooperative
cleanup before a five-second main-page termination fallback. An OS process kill
cannot execute these cleanup paths. Simple probe workers may still require
immediate termination when stopped by their runner.

Use one immutable release and record device, browser and inspector status.
Compare 1b, 1d and 2c at 8 MiB first, then full loading. Compare storage modes
with other settings fixed; use 4/2 MiB staging only as separate comparisons.
Small-model overhead and allocation order differ from full Gemma initialization,
so a successful combined probe narrows investigation without proving its cause.

Phone acceptance requires three complete session-only loads, three complete
application loads, five cached loads, short/long input inference, then two full
100-row evaluations in one session. Correlate interruptions with same-time
device/Jetsam logs. Desktop SwiftShader checks verify functionality and output
compatibility, not iPhone memory acceptance or performance.

Local verification and the exact tested release are recorded in
[session-diagnostics-validation.json](session-diagnostics-validation.json).
The suite passes 50 JavaScript tests, two Python tests and 22 browser cases,
including full-size 1b/1c/1d/2c comparisons, output compatibility, real IndexedDB
transaction abort/recovery, and cooperative application-worker cancellation.
