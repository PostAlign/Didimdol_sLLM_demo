# iPhone 14 Pro Max FP32 loading

Target: iPhone 14 Pro Max, iOS 26, Chrome for iOS. FP32 weights, model revision,
100 evaluation rows, prompt template, sampling settings and the 512-token limit remain the same.
This change has local browser tests; iPhone success must be established on the actual device.

## Loading path

1. Fetch the small ONNX graph and match its SHA-256 to `model/initializers.json`.
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
7. Load the tokenizer after session creation and source cleanup, reducing overlap with loading.

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
| GPU residency | No ORT import. Fill and read every float of buffers matching all external weights, keeping every buffer alive until completion. |
| Small runtime | Execute the 10 MiB FP32 model twice, then wait 120 seconds to expose delayed compilation/resource growth. |
| Full load | Actual app/from_pretrained path with verified OPFS weights. |
| Five cached loads | Five fresh pages/workers reusing completed files. Run after the first full load. |
| Short inference | Actual shortest (35 tokens) and longest (266 tokens) evaluation inputs, up to 32 greedy tokens each. |
| Two evaluations | Two full 100-row evaluations in one session with per-row GPU allocation snapshots. |

Successful repeated-load experiments deliberately navigate to the next page. Unexpected
restarts stop the sequence and show the interrupted run. Every run ID is included in JSON export.
New pages/workers do not guarantee a new OS process, nor do logical GPU counters measure RSS.
Run with and without an attached inspector. A successful small probe cannot rule out
compilation pressure unique to the full graph.

Acceptance on the phone: complete 251-initializer loading, a first download and five cached
loads, two 100-row evaluations without restart/device loss, stable resource usage across
repeats, and FP32 output/ROUGE/latency comparison with the reference. Record system jetsam
logs where available to distinguish WebContent/GPU/compilation memory from other failures.

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
