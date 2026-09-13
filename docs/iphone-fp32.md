# iPhone 14 Pro Max FP32 loading

Target: iPhone 14 Pro Max, iOS 26, Chrome for iOS. FP32 weights, model revision,
100 evaluation rows, prompt template, sampling settings and the 512-token limit remain the same.
This change has local browser tests; iPhone success must be established on the actual device.

The next follow-up adds small-runtime plus full-weight residency, native session
phase markers, incremental diagnostic storage and bounded cleanup. See
[session-diagnostics.md](session-diagnostics.md) for its comparison protocol and
the interpretation of the September 12 `02719f4a4081` phone logs.

## Compact BPE and full-session comparisons

The September 12 phone exports from release `c418434115fb` (commit `53303eb`)
completed tokenizer preparation but interrupted during full-model uploads:
180/251 initializers with 8 MiB staging on the evaluation page and 241/251 with
2 MiB on the diagnostic page. Both had nine OPFS cache hits, a 23.125 MiB WASM
heap and no recorded fault/device loss. These are interruption observations,
not proof of jetsam or a controlled staging-size comparison.

The new `compact-bpe-v2` bundle stores numeric token-ID pair keys and merge ranks
in a fixed open-addressed table (12 MiB for this tokenizer). It preserves the last
rank for duplicate rules, exact integer keys and a fallback for rules whose pieces
are absent from the base vocabulary. The source is pinned by version and SHA-256;
the transformers source retaining tokenizer JSON is also checked before building.
Added tokens that change a base vocabulary ID are rejected during preparation.

After construction, `releaseBpeSource` replaces the three private source-JSON
references with lightweight configuration and drops the source merge array.
The caller's JSON is not mutated. Runtime vocabulary, normalizer, pre/post
processors, decoder, added tokens and all generation/evaluation settings remain
available. This application bundle intentionally does not retain a serializable
original model under private `_tokenizerJSON`, `tokenizer.model` or `model.config`.
Use the immutable tokenizer assets to reconstruct another tokenizer. Releasing
references makes temporary objects collectible; it does not force browser GC.

`tokenizer.memoryLayout` records source release, merge/rank counts and table bytes.
The Node memory comparison includes upstream, previous `incremental-bpe-v1`, and
current implementations in fresh processes. Compare **heapUsed + arrayBuffers**
as well as maximum RSS; typed-array storage must not disappear from the reported
memory budget. These are Node measurements, not iPhone process memory.
In the local three-run comparison, median retained heap plus array buffers fell
from 142.49 MiB (v1) to 48.00 MiB (v2), including the 12 MiB rank table. Median
maximum RSS fell from 331.35 to 298.50 MiB. Exact release identity, measurements
and browser results are in [compact-tokenizer-validation.json](compact-tokenizer-validation.json).

| Comparison | UI experiment | Tokenizer retained | Full ORT session | Upload order |
|---|---|---|---|---|
| A | 1b: stored-weight residency | no | no | manifest |
| B | 1c: tokenizer + stored-weight residency | yes | no | manifest |
| C | 2c: session only | no | yes | ORT |
| D | 3: full model load | yes | yes | ORT |

A/B now use the same application worker and import the same ORT/transformers
JavaScript. Neither instantiates ORT WASM. Their runtime mode is null, with the
JavaScript mode and imports recorded separately. C/D call the same
`loadModelSession` function, including graph verification, OPFS preparation,
session options and range loader. Both prepare the template/evaluation data;
only D prepares the tokenizer and ROUGE closure. C completes with the distinct
`model-session` scope and never emits application `ready` or accepts inference.
B keeps its tokenizer reachable until every weight has been verified.
Experiment controls stay disabled until release/bootstrap initialization finishes,
so a slow startup cannot overwrite the user's selection with the default case.

Compare A/B and C/D at 2 MiB first, using the same release, device, inspector
setting and verified OPFS cache. Each can request three runs. Every run creates
a new page/worker; an interruption stops automatic continuation. A new page
does not guarantee a new OS process. Initializer order, cache use, component
completion and recovery evidence are retained in exports. The existing synthetic
residency probe and the legacy direct probe worker remain available separately.

Local checks: `npm test`, `npm run build`, `npm run verify:release`,
`node tools/measure-tokenizer.mjs`, and `TEST_APP_LOAD=1 npm run test:browser`.
Browser checks cover full-model load/short-long prompt decoding, A/B/C through
the UI, missing-cache rejection, completion recovery and cleanup. Phone
acceptance remains three full loads, five cached loads, short/long inference,
then two 100-row evaluations. Compare any new interruption with the same-time
device termination report before assigning an OS/GPU cause. Session-only failure
on the phone is the trigger for further native ORT/driver-memory investigation.

## Previous tokenizer preparation change (incremental-bpe-v1)

The two iPhone full-load records reached `session-create-complete`, with all 251
external initializers uploaded and validated, then stopped at `tokenizer-load`.
This locates the interruption after session creation; it does not establish an
OS-memory termination. The current path prepares the tokenizer before model GPU
allocation so tokenizer construction does not overlap with the full weight set.
Its retained vocabulary still consumes memory alongside the finished model.
Dropping temporary references does not force browser garbage collection.

`web/sllm/tokenizer-loader.js` reads the two JSON files sequentially from immutable
release URLs. It uses normal HTTP caching, without cloning responses or adding a
second Cache Storage copy. Read, UTF-8 decode, JSON parse and constructor start/end
markers are persisted before the next operation. Configuration and tokenizer file
records remain under `preparation` after the 64-entry recent-event history rolls
over. Template and evaluation-data preparation have separate stages. The expected
asset SHA-256 is release metadata, not a claimed runtime rehash; loaded byte sizes
are checked. `jsMemory` is `null` where `performance.memory` is unavailable, and
even supported JS-heap samples are not total browser/process RSS.

The previous patch checked the exact tokenizers 0.1.3 and transformers 4.2.0
source hashes and patched the browser bundle in memory. BPE/vocabulary Maps were
filled incrementally without creating complete arrays of key/value pairs. Merge
keys, insertion order, tokenizer class selection and retained source configuration
were preserved. The patch also exposes upstream class selection as `from_json` so
the application can instrument file parsing separately. The patch identity is
included in release hashes and diagnostics. `node_modules` is not modified.

Experiment **2b. 토크나이저 준비만 확인** calls the same loader through the application
worker and reports only tokenizer preparation success. It imports the application
and ORT JavaScript but never creates a model session or instantiates ORT WASM.
The effective vocabulary has 262,145 slots: 262,144 base entries plus the added
`<image_soft_token>` at ID 262144. There are 514,906 BPE merge rules.

Validation commands:

```bash
npm test
npm run build
npm run verify:release
node tools/measure-tokenizer.mjs > .work/tokenizer-memory.json
TEST_APP_LOAD=1 npm run test:browser
```

The Node memory comparison runs baseline and patched BPE implementations three
times each in fresh processes, using the same new preparation loader. It isolates
the BPE change, not the effect of reordering GPU allocation. Browser tests compare
the tokenizer diagnostic and full app load/probe, including recorded FP32 token
outputs. `docs/tokenizer-validation.json` records local results. On the phone,
run tokenizer preparation, full load three times, five cached loads, short/long
inference, then two 100-row evaluations in that order. A new interruption should
be compared with same-time WebContent/GPU or Jetsam device records; a successful
desktop check is not iPhone memory acceptance.

## Loading path

1. Prepare the tokenizer, chat template and evaluation data before creating any model GPU buffers.
   Then fetch the small ONNX graph and match its SHA-256 to `model/initializers.json`.
2. Download each external file sequentially into OPFS. A legacy Cache Storage response is
   streamed into OPFS when available; it is removed only after successful verification.
3. Validate exact file size and SHA-256 for each 8 MiB block. Flush the data, close its
   handle, and write a completion marker containing the graph/file verification identity.
   A missing, mismatched or incomplete marker never counts as a cache hit. Interrupted
   downloads restart that file; this is not HTTP byte-range download resumption.
4. Give ORT ABI 2 descriptors with `size` and `readRangeInto`, not whole-file Blobs.
5. Read OPFS directly into a reusable 8 MiB JS scratch, upload to ORT's GPU buffer,
   and await the GPU queue before reusing the scratch. Only one OPFS read handle is open.
6. Seal the source after session creation. Subsequent inference must not read external weights.
7. Report `ready` only after both tokenizer preparation and session creation have completed.

The OPFS verification marker trusts an already verified, unchanged local file with matching
size and metadata. It is not a fresh rehash on every visit. The original model bytes are unchanged.
OPFS/storage failures are reported; there is no full-model ArrayBuffer or CPU fallback.
`navigator.locks` prevents concurrent production loaders in the same origin where supported.
OPFS itself also enforces exclusive access handles.

The expected external GPU residency remains 1,072,392,704 bytes (1,022.7 MiB).
An 8 MiB scratch is not the whole browser memory budget. GPU driver allocations, file caching,
WebKit/WASM compilation and the tokenizer are not included in this counter.

## Diagnostics and stop behavior

### 2026-09-12: release identity and interruption evidence

The landing pages now bootstrap `web/vendor/releases/<releaseId>/`. That immutable
snapshot contains application modules, workers, local model/tokenizer metadata and
ORT JS/factory/WASM/transformers. Relative worker imports and local data reads stay
inside the snapshot. `web/vendor/build.json` is published last and points at the
verified snapshot; the nested manifest must describe the same release. Both the
browser and the verification tool check the manifest identity. Individual asset
bytes are verified by the Node tool, not by downloading an extra WASM copy on the phone.

`npm run release` packages current application files with the available native
binaries, including after a CI cache hit. Run it after application changes before
browser testing or serving the site. `npm run build` also packages a release after
bundling ORT. New native builds record the ORT commit, actual Emscripten compiler
version and build command; old cached binaries without this record have unknown
native provenance (`nativeBuild: null` when rebundled), not an inferred build history.
The application commit, dirty-worktree flag, asset hashes and release ID identify
local builds as well as committed deployments.

```bash
npm run release
npm run verify:release
# Verify an assembled directory or the actual served bytes (supports Pages subpaths):
npm run verify:release -- _site
npm run verify:release -- https://HOST/REPOSITORY/
```

CI verifies the local snapshot, assembled site and deployed URLs. Old open pages
continue to reference their original release; retaining that release on the host
is necessary for later lazy imports/reads to succeed. A missing old release fails
instead of loading files from a new one. The snapshot duplicates about 40 MiB of
local application/tokenizer data on disk; it does not preload those files into RAM.

New runs use diagnostic schema 3; schema 2 records remain readable. In addition
to the latest 64 events, runs retain preparation/session milestones, the final
preparation state of up to 64 files, and up to 2,048 allocation-order entries
(with a truncation counter). Original `firstFault`, the priority `fault` (device
loss takes precedence), error type/operation and last stage survive cleanup.
Recovery is stored under `recovery:<UUID>` and merged on export. It records
`interrupted`/`unknown` without rewriting the worker's original status/last/fault.

Every range checkpoint includes the GPU allocation ledger. `gpuWeightAllocated`
counts destination buffer sizes; `gpuWeightUploaded` counts bytes whose queue wait
completed without observed device loss. The latter does not measure physical
residency or prove error scopes were clean; `initializer-complete` follows their
validation. `gpuWeightBufferCount` counts weight buffers, while the GPU ledger
also counts other live buffers and their requested bytes. `gpuWriteMs` measures
the synchronous write call and `gpuWaitMs` measures only the queue wait. IndexedDB
time is excluded and reported separately under `persistence` (completed writes
before the current snapshot). These counters still exclude process RSS, compiler
and driver memory.

The experiment controls offer 2/4/8 MiB staging, inspector attachment status, and
one or three full loads. Five cached loads remain a separate experiment. Each
repeat preserves its device/settings/release ID and stops if the release changes.
Unexpected restarts stop repeats. Smaller staging changes transfer/scratch size,
not the approximately 1,023 MiB weight residency. The small-runtime probe records
requested and elapsed idle time; `idleAcceptanceCompleted` is true only after the
full 120-second observation. A short smoke run is not idle acceptance.

```bash
npm test
npm run test:browser
ORT_MODES=asyncify TEST_IDLE_SECONDS=120 npm run test:browser
ORT_MODES=asyncify TEST_APP_LOAD=1 TEST_FULL_MODEL=1 npm run test:browser
```

Use the same release on the phone for resident-only, small-runtime and full-model
comparisons. Record exact OS/browser versions and whether an inspector is attached.
Compare `persistence` times and session durations across repeated runs to quantify
diagnostic overhead; timing counters alone cannot establish a memory cause.
Only handset testing and system crash/Jetsam evidence can establish whether the
reported restart is fixed. Weight swapping and compiler/runtime changes remain
conditional follow-ups after the experiments distinguish their causes.

Reviewed local results and exact release IDs are in [diagnostics-validation.json](diagnostics-validation.json).
The synthetic 756-checkpoint IndexedDB comparison measured 625–663 ms before and
840–880 ms after this change (two samples each); final snapshot sizes were 27,702
and 59,534 bytes. This additional diagnostic cost must be checked on the phone.

Each load and evaluation has its own UUID. The active marker/history live in sessionStorage
so another tab cannot supply the previous attempt. IndexedDB `didimdol-runtime-diagnostics/runs`
stores `run:<UUID>` records with environment/build identity, a bounded 64-event history,
the last stage, a separately preserved fault, and an evaluation summary without prompt text.
The last 8 main-page runs can be exported using **진단 기록 저장**.

Durable stages include allocation, range read, GPU write, GPU wait and initializer completion.
The current initializer's buffer is counted before its upload checkpoint. `device-lost` is
persisted before notifying the window; cleanup cannot erase that fault.
An unfinished marker reports interruption, not a proven OOM. Page visibility/navigation
events are hints, not proof of an OS kill. iOS minor/Chrome versions should be entered manually
on the experiment screen because the user agent may not expose their full versions.

Generation inputs/outputs are disposed by their owner. Transformers.js owns its normal KV
cache cleanup. A generation exception ends the worker instead of reusing an uncertain native
allocator. A normal stop interrupts generation; if it has not acknowledged within 3 seconds,
the window terminates the worker, records a user cancellation and requests a fresh page.
All-row averages are shown only after a complete 100-row run.

## Build and asset matching

```bash
source .venv/bin/activate
npm ci --ignore-scripts
bash tools/build-ort.sh asyncify
bash tools/build-ort.sh jspi
```

Defaults: `ORT_PROFILE=mobile`, `ORT_THREADS=0`. The mobile configuration uses
`model/required-operators.config` for this graph and keeps FP32/int/bool types.
The generated file also lists `TopK` (opset 21) for the small session that
transformers.js creates on the first sampled token; without it sampled evaluation
rows fail with a missing `TopK(11)` kernel.
This is a compile-time single-thread build, not just `numThreads=1` at runtime.
For controlled comparisons, `ORT_PROFILE=baseline ORT_THREADS=1` retains the upstream broad
operator configuration. Each combination has its own build directory. Building a comparison
replaces the local served variant; rebuild the mobile variant before deployment.

`web/vendor/build.json` records the profile, thread option, WASM size/hash, operator configuration
hash and patch hash. The app reads fresh build metadata and adds the binary hash to runtime
asset URLs to avoid pairing an old factory with a new WASM file. ABI 2 is checked before
mounting range descriptors. JSPI is explicit and feature-detected; Asyncify remains the default.
No `wasm-opt --asyncify` second transformation or speculative compiler workaround is applied.

When exporting a new model, regenerate both its metadata/hashes and operator configuration:

```bash
python tools/inspect_initializers.py model/web/model.onnx --revision HF_COMMIT --output model/initializers.json
python tools/create_ort_config.py model/web/model.onnx --output model/required-operators.config
```

The small device probe embeds only an ONNX graph and metadata in `fixture.js`. Its 10 MiB
weights are generated locally, verified and stored in OPFS; no diagnostic model upload is needed.
Regenerate it after changing `tools/make_test_model.py`:

```bash
python tools/make_test_model.py
python tools/create_probe_fixture.py .work/test-model/model.onnx --output web/sllm/experiments/fixture.js
```

## Device experiments

Open `web/sllm/experiments/` on the actual phone, enter the complete OS/browser version and run:

| Experiment | Purpose |
| --- | --- |
| GPU residency | No ORT import. Fill buffers matching all external weights and checksum every word, keeping every buffer alive until completion. |
| Stored-weight GPU residency | Application worker/JS imports, no ORT WASM or session. Read verified OPFS files and retain all weight buffers. Requires files already prepared by a full-load attempt. |
| Tokenizer + stored-weight residency | Same path as stored-weight residency, retaining the application's prepared tokenizer throughout verification. |
| Small runtime | Execute the 10 MiB FP32 model twice, then wait 120 seconds to expose delayed compilation/resource growth. |
| Tokenizer preparation | Same application worker and tokenizer loader, without a model session or weight requests. Imports application/ORT JavaScript; does not instantiate the ORT WASM runtime. |
| Full session only | Same session-creation path as full load, with tokenizer preparation omitted. Reports session completion separately from application readiness. |
| Full load | Actual app/from_pretrained path with verified OPFS weights. |
| Five cached loads | Five fresh pages/workers reusing completed files. Run after the first full load. |
| Short inference | Actual shortest (35 tokens) and longest (266 tokens) evaluation inputs, up to 32 greedy tokens each. The worker also accepts `sampled: true` (used by the smoke test) for one extra sampled token that creates the top_k auxiliary session. |
| Two evaluations | Two full 100-row evaluations in one session with per-row GPU allocation snapshots. |

Successful repeated-load experiments deliberately navigate to the next page. Unexpected
restarts stop the sequence and show the interrupted run. Every run ID is included in JSON export.
New pages/workers do not guarantee a new OS process, nor do logical GPU counters measure RSS.
Run with and without an attached inspector. A successful small probe cannot rule out
compilation pressure unique to the full graph.

Acceptance on the phone: complete 251-initializer loading, a first download and five cached
loads, two 100-row evaluations without restart/device loss, stable resource usage across
repeats, and FP32 output/ROUGE/latency comparison with the reference. Record system jetsam
logs where available to distinguish WebContent/GPU/compilation memory from other failures:
the page's file input reads `JetsamEvent-*.ips` and WebContent crash files and attaches the
kill reason, footprint and free memory to the interrupted rows before export. Files that
match no row are still kept in the export (`deviceReports`) with whether they fall inside
the span of the runs.

On iOS the acceptance path is the streamed one. Resident inference is diagnostic only: in
three September 13 sessions every resident inference ended within a second of its first
token at about 1,037 MiB requested, and the one with a same-minute Jetsam report was a
`highwater` kill at 2,284 MiB of WebContent footprint. The evaluation page streams by
default on iOS (`modelExecution=streamed` unless the URL says otherwise) and the experiment
page defaults to streamed execution with a 30 s repeat delay. Streamed loads and probes have
completed in every session with a 498 MiB peak request; the streamed cached-load repeat
still needs the 30 s-delay comparison because two sessions ended a third rapid repeat at
`ort-plan-start`.

Streamed throughput on the September 13 phone was about 1.5 tokens/s at 2 MiB staging
(about 78 s per evaluation row, more than two hours per 100-row pass). The evaluation page
accepts `rowLimit=N` to time a prefix of the rows; such a run is labelled a partial
evaluation and is not acceptance. See docs/session-diagnostics.md.

If the residency-only test fails, source/scratch improvements cannot guarantee success.
Splitting into simultaneously resident sessions does not reduce the 1,023 MiB weight total.
Weight swapping, lower precision and server inference are not implemented in this change.

## Local verification

```bash
npm test
python -m unittest discover -s tests -p 'test_*.py'
python tools/make_test_model.py
python tools/prepare_experiments.py .work/test-model .work/test-experiments
npm run test:browser
# Requires the unchanged full graph/weights in .work/full-model and prepared .work/full-experiments:
TEST_APP_LOAD=1 TEST_FULL_MODEL=1 npm run test:browser
```

Browser tests cover real OPFS/ABI 2 loading, exact fixture output, retained-buffer compute,
cache reuse, per-tab recovery and the full model's recorded two-token FP32 baseline when enabled.
The full app probe exercises both real prompt lengths. Tests on SwiftShader are functionality
checks, not iPhone performance or OS-memory acceptance. Results are written to
`test-results/browser-smoke.json`; reviewed results belong in `docs/iphone-validation.json`.

Relevant upstream investigations: [WebKit WASM compilation memory](https://bugs.webkit.org/show_bug.cgi?id=304810),
[WebGPU submission pressure](https://bugs.webkit.org/show_bug.cgi?id=311598),
[WebKit OPFS](https://webkit.org/blog/12257/the-file-system-access-api-with-origin-private-file-system/).
They motivate the probes and do not prove the cause of the reported phone restart.

### GPU tracking and interruption follow-up

The worker now observes `GPUAdapter.requestDevice`, `GPUDevice.createBuffer` and
explicit buffer/device destruction through prototypes, with instance fallbacks.
The range bridge also registers the device/buffer actually supplied by native ORT.
This preserves ORT's native device-creation path and the synchronous `createBuffer`
API. Hooks remain installed during inference and are restored when the worker
finishes or fails. Multiple devices have separate identities in the ledger.

`gpuLedger.tracking.status` is `complete`, `partial`, or `unbound`. Late bridge
registration counts the buffers actually observed, but leaves historical
`requestedPeak` and `mappedUploadRequested` unknown (`null`). `observedPeak` is
only the largest observed request total, not a reconstruction of earlier memory
use. Explicit `destroy()` updates logical live counts; physical reclamation is
not measured. Old schema-3 exports with nonzero weights and a zero ledger are
shown as partial, and absent schema-2 counters remain unknown.

Allocation error scopes are pushed before `createBuffer` and popped immediately
after its synchronous call. Their asynchronous results preserve allocation-time
initializer context; pending fault writes are drained before uploading weights.
These extra scope operations have a cost that must be compared on the phone.
Upload scopes still validate each initializer independently.

Range metrics distinguish `gpuWriteReturnedBytes`, `gpuQueueCompletedBytes` and
`gpuValidatedInitializerCount`. `gpuWeightUploaded` remains a compatible alias for
queue-completed bytes. A `gpu-wait` checkpoint has `phase: "before-call"`: it
follows a returned write but precedes the queue wait. An interrupted final
checkpoint cannot prove the exact native crash instruction or an OOM.

Recovery retains raw lifecycle hints in `lifecycleHistory`; `lifecycle` only
contains events inside the run interval with a matching UUID when available.
It also records `unloadEvidence` (`unload-observed` when a `pagehide` or hidden
`visibilitychange` event fell inside the run, `no-unload-event` otherwise,
`recorded-fault` when the worker persisted a fault), `reentryGapMs` between the
last checkpoint and the recovering page, and the recovering page's
`navigationType`. A silent re-entry (no unload event, a gap under a few seconds)
matches a WebContent/GPU process termination and rules out a normal navigation,
but `cause` stays `unknown`: only device logs establish a memory kill.
The original worker status and fault are preserved. The UI shows interrupted
runs and their tracking coverage, and experiment rows/export include release,
reported device, runtime/staging, OPFS cache/migration/download counts, completed
initializers, weight requests, returned/completed transfers and the last position.

Phone validation order for this follow-up:

1. Use one immutable release and record the device/browser and inspector setting.
2. Run resident-only, then the small runtime with the full 120-second idle period.
3. Run the full model from verified OPFS cache at 8, 4 and 2 MiB staging, keeping
   other settings constant. Compare cumulative allocated bytes as well as the
   last initializer: native initializer order can differ between runs.
4. Repeat a successful setting with five cached loads and short/long input probes.
5. Correlate any interruption with device crash/Jetsam logs. Smaller staging does
   not reduce the approximately 1,023 MiB FP32 weight residency. Default staging
   or model residency changes require those device results.

### Experiment scope and OPFS comparison

The experiment export keeps UI preferences under `screenSettings`. Each result
has `execution` evidence and belongs to a `series` with requested, started and
successful run counts. An evaluation's load/probe/evaluation UUIDs are not counted
as repeated loading attempts. Older records remain readable; missing repeat
counts or measured idle durations stay unknown. Resident probes have
`runtimeMode: null` and zero requested idle time. The runtime and repeat selectors
are disabled when they do not apply to the selected experiment. Runtime results
report measured idle milliseconds; only a completed observation of at least
120 seconds grants the full idle acceptance label.

`resident-opfs` requires already verified production OPFS files. It never downloads,
migrates, or substitutes model weights. A failed full load may have finished file
preparation and therefore can still supply this comparison. Missing files produce
an explicit error. This path uses the production store and the same origin lock,
keeps one read handle open, and closes it on completion or failure. Fixture data
creation is available only via the existing test worker's explicit `fixture` flag.

All resident modes use `verification: u32-fnv1a-64-lanes-v1`: CPU and GPU
compute matching integer checksums over every 32-bit word of the FP32 data. The
GPU returns 256 bytes per initializer. This is a checksum comparison, not a
cryptographic proof or a full-byte readback. All modes retain all weight buffers,
use the same scratch size, shader and manifest allocation order, and create no
ORT session. The application-worker OPFS pair imports ORT JavaScript without
instantiating WASM. CPU checksum work is additional work relative
to production loading. The old resident probe used floating-point sums of ones,
so its durations should not be compared directly with the new verification method.
The manifest order also differs from ORT's observed allocation order. Neither a
resident success nor logical buffer sizes prove physical residency, full-model
inference success, or prolonged stability.

Production range checkpoints now sample the current WASM heap capacity and the
OPFS store's completed range-read count, bytes, total/peak read duration and active
file/handle state. Read time includes acquiring a new file handle when needed,
and excludes diagnostic persistence and file preparation/hash verification. A
completed session reports closed handles; an interrupted upload retains its most
recent storage snapshot. Full-load results also record graph preparation, weight
preparation, the `from_pretrained` call, tokenizer and evaluation-input
preparation and total load durations. These phase durations include awaited
checkpoints; GPU/read operation timers and `persistence` remain separate. Complete
GPU tracking permits reporting total requested GPU bytes alongside weight bytes;
partial tracking does not fabricate a total. Process RSS and driver memory are
still unmeasured.

On the same phone/browser/release, compare synthetic residency, small-runtime
inference with the full 120-second idle period, full loading at 8 MiB, and stored
OPFS residency. If full loading still interrupts, compare 4/2 MiB using the same
cache and inspector conditions. Complete five warm loads and two evaluations
before claiming handset acceptance. Correlate unexplained restarts with device
logs; this change does not establish an iPhone crash fix by itself.

Local results and the exact tested release identities are recorded in
[experiment-path-validation.json](experiment-path-validation.json). The full-size
app and OPFS comparison, real-prompt FP32 outputs, and the 120-second runtime
observation passed locally. The final export refinement was then checked with
all runtime modes, including a corrupted-upload rejection and a missing-cache
failure without a model network request. Phone validation remains outstanding.
