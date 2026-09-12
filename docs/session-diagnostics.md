# Full-session loading diagnostics

The September 12 phone exports from release `02719f4a4081` (application commit
`722cd56`) passed synthetic residency, OPFS residency, tokenizer plus OPFS
residency, and the small runtime with 120 seconds of observation. Session-only
loading interrupted after 209/251 initializers; evaluation-page loading stopped
after 247/251. Both used 8 MiB staging, nine OPFS cache hits and Asyncify, with no
persisted GPU fault. These observations do not establish an OS-memory kill.

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
