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
