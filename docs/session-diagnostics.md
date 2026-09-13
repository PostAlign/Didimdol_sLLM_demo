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

This follow-up adds a comparison and lowers diagnostic persistence overhead.
FP32 model weights, tokenizer, generation settings, evaluation inputs and graph
optimization settings are unchanged. Native changes add lifecycle checkpoints;
they do not change buffer allocation, packing or execution policy.

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
