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

One JetsamEvent file can list several kills (every process with a `reason`),
including kills that happened before the file's own time stamp, so a
termination may appear in the report written at the next one. Drop the files
into the experiment page's device-log input rather than reading them by hand;
it matches WebContent kills to interrupted rows in time order.

Open decisions that wait for that log: whether the evaluation page's iOS
default should become `modelExecution=streamed` (resident inference completed
0 of 3 attempts here, streamed 1 of 1), and whether the streamed cached load
3/5 recurs with a 30 s repeat delay. The acceptance run for the streamed path
is two 100-row evaluations; at about 17 s per row each pass takes close to
30 minutes on this device.

## September 13 late-afternoon exports with device logs (release `18cab5534d84`, commit `af91292`)

Same iPhone (iPhone15,3, 5.5 GB, 16 KiB pages), Chrome 153, Asyncify, nine OPFS
cache hits in every run, no persisted GPU fault, no device loss. The experiment
page used `snapshot` diagnostics and a 0 s repeat delay; the evaluation page ran
`compact`. Exports `iphone-fp32-experiments (46)`–`(54)` are each a prefix of the
next, and all 19 rows survived two process terminations, so the durable results
store did its job. For the first time the phone's `JetsamEvent` files from the
same minutes were collected (`JetsamEvent-2026-09-13-173918.ips`,
`JetsamEvent-2026-09-13-174006.ips`), together with a Networking-process disk-write
resource report (`…diskwrites_resource-2026-09-13-172712.ips`).

| KST | Screen / experiment | Outcome |
|---|---|---|
| 17:06:39 | Evaluation page session create, resident | 251/251, `ready` in 14.1 s |
| 17:06:53 | Evaluation page warmup (sampled) | interrupted at `warmup-start` before any sample, page back after 1.4 s |
| 17:23:38–17:29:33 | 1, 1b, 1c, 1d, small runtime 120 s | all complete |
| 17:32:41 | tokenizer | complete |
| 17:33:07 | 2c session-only, streamed | complete, 11.1 s |
| 17:33:38 | 2c session-only, resident | **complete**, 251/251, 12.8 s (first success in four sessions) |
| 17:34:28 / 17:35:36 | 3 full load, streamed / resident | both `ready` |
| 17:36:41–17:37:16 | 4 cached load 1–3/5, streamed | all `ready`, 0.15–0.86 s apart |
| 17:37:16 | 4 cached load 4/5, streamed | **failed in code** at `streamed-head-create` / `ort-wasm-start`, 0 bytes of weights, see below |
| 17:38:43 / 17:38:56 | 4 cached load 1–2/5, resident | both `ready` |
| 17:39:09–17:39:23 | 4 cached load 3/5, resident | interrupted at 224/251 (`onnx::MatMul_7520`), 940 MiB allocated, page back after 1.0 s |
| 17:39:52–17:40:06 | 5 short inference, resident | load `ready` in 11.6 s, interrupted 1.0 s after the first token at 1,037 MiB requested, page back after 1.4 s |
| 17:40:37–17:41:24 | 5 short inference, streamed | complete in 46.9 s, 32 tokens × 2 (17.1 s, 19.7 s), peak 498 MiB |

The three interruptions recorded `no-unload-event` and `navigationType:
back_forward` as before. The cached load 3/5 initializer order (224 entries) is a
prefix of the 251-entry order of the two loads that survived, and its upload of
940 MiB took 5.4 s against 4.6–5.2 s for them.

### What the Jetsam reports say

| Report (device local) | Content |
|---|---|
| 17:39:18.43 | `SharingUIService` (suspended, 22 MiB) killed `highwater`; `AppSSODaemon` `fc-thrashing`. Free 67 MiB, compressor 582 MiB. WebContent pid 2023: 564 MiB resident, lifetime maximum 1,841 MiB, not killed |
| 17:40:06.65 | WebContent pid 2023 killed `highwater` at 1,754 MiB. WebContent pid 2076 killed `highwater` at 2,284 MiB (its lifetime maximum). Free 170 MiB, compressor 1,150 MiB |

- The 17:39:18 snapshot falls 1.1 s after cached load 3/5 began uploading
  initializers. The system was already short of memory (free 67 MiB) and the
  previous run's cleanup had returned WebContent from 1,841 MiB to 564 MiB, so
  nothing accumulated across runs inside the renderer.
- The pid 2023 kill lies between the row's last record (17:39:22.75) and its
  re-entry (17:39:23.76); the OS footprint was 1,754 MiB against the page's
  940 MiB GPU request.
- pid 2076 is the replacement process created at that re-entry (its age field is
  negative in the report, 38.6 s of CPU time in 43 s). Its kill at 17:40:06.65
  matches the resident probe's end (last sample 17:40:05.53, re-entry
  17:40:06.95): 2,284 MiB against 1,037 MiB requested.
- The WebKit GPU process held 36 MiB in both snapshots, so the weight memory is
  charged to WebContent, at 1.87× and 2.20× the page's GPU request. Whether that is a
  shared-memory copy or Metal buffer attribution cannot be read from these files.
- `highwater` is the kill of a process above its own soft limit when the jetsam
  thread runs under system pressure. The same process had survived at 1,841 MiB
  earlier, so there is no fixed threshold: resident FP32 keeps WebContent above
  the limit, and which run dies depends on when pressure arrives. About 830 MiB
  of suspended third-party apps (Slack, KakaoTalk, Runner, Gmail, Podcasts,
  Preferences) were resident at the time.
- No report exists in the collection for the 17:06:53 evaluation-page
  interruption; that one is still unconfirmed.

The streamed cached load 4/5 was not a kill. The worker failed with

```
no available backend found. ERR: [webgpu] RuntimeError: Aborted(NetworkError:  A network error occurred.)
@…/releases/18cab5534d84…/web/vendor/ort.asyncify.mjs:1:1822
```

1.2 s after `ort-wasm-start`, before any weight read, on the fourth back-to-back
load. The Networking process kept its pid (1134) across both Jetsam snapshots, so
it was not restarted. The previous session's streamed 3/5 stopped at
`ort-plan-start` with no weights on the third rapid repeat; both remain
unexplained and the 30 s repeat-delay comparison is still owed.

The Networking resource report covers 15:57–17:27 and records 4.29 GB of
file-backed memory dirtied through WebCore's SQLite storage with no action taken.
`snapshot` diagnostics rewrite the whole run state on each of about 2,300
checkpoints per run (0.47 MB each, about 1 GB per run); the results table in
`localStorage` is 0.32 MB and is not a contributor.

Changes:

- **Evaluation page iOS default.** Without a `modelExecution` URL parameter, iOS
  now streams; other platforms keep `resident`. `environment.modelExecutionSource`
  (`url`, `default-ios`, `default`) records where the choice came from, and the
  ready line names the mode.
- **Device report matching.** The experiment page accepts `.ips` files. JetsamEvent
  bodies are parsed for WebContent kills (pid, reason, `rpages`, lifetime maximum),
  free memory and compressor size; kills are matched in time order to interrupted
  rows, several per report when the report holds several. WebContent crash files
  count as kills without a footprint; resource reports are named and skipped. The
  row shows the footprint, the lifetime maximum, the ratio to the page's GPU
  request and free memory, and `deviceLog.matchedBy: 'report'` marks file-derived
  notes. Hand-written notes are never replaced.
- **WASM failure evidence.** When a session fails between `ort-wasm-start` and
  `ort-wasm-complete`, the worker records an `ort-wasm-error` fault with the error
  type, the WASM asset URL, `navigator.onLine`, the HTTP-cache status of that
  asset and a fresh `HEAD` status. `executionEvidence.ortWasmInstantiated` is
  `false` for such runs instead of `null`, and `wasmFailure` carries the record.
- **Snapshot mode label.** The selector states the disk-write cost.

Still owed from the phone: two streamed 100-row evaluations from the evaluation
page opened without parameters, streamed cached load ×5 with a 30 s delay, and
the `.ips` files of any interruption dropped into the experiment page before
export.

## September 13 night exports (release `217b6bd51b75`, commit `82adfc7`)

Same iPhone (iPhone15,3, 5.5 GB), Chrome 153, Asyncify, nine OPFS cache hits in
every run, no persisted GPU fault, no device loss, `compact` diagnostics on both
pages, 0 s repeat delay. The evaluation page was opened without parameters and
therefore streamed (`modelExecutionSource: default-ios`); it was exported once
(`didimdol-diagnostics (11)`), the experiment page ten times
(`iphone-fp32-experiments (55)`–`(64)`, each a prefix of the next; `(58)` is a
re-export of `(57)` after the phone had slept). Three device files were
collected: `JetsamEvent-2026-09-13-225106.ips`, a Networking disk-write report
(`…diskwrites_resource-2026-09-13-231303.ips`) and nothing from the minutes of
the three interruptions.

| KST | Screen / experiment | Outcome |
|---|---|---|
| 22:54:44 | Evaluation page session create, streamed 2 MiB | 235/235, 383 MiB weights, 423 MiB requested, `ready` in 10.8 s |
| 22:54:55–23:01:28 | Evaluation page 100-row evaluation, streamed | **rows 1–5 complete in 389 s**, one auxiliary `top_k` session, peak 496 MiB; stopped by the tester in row 6 (`AbortError` at output-upload chunk 14, status `cancelled`) |
| 23:02:20–23:06:43 | 1, 1b, 1c, 1d, small runtime 120 s, tokenizer, 2c resident, 2c streamed | all complete; 2c resident 251/251 in 10.4 s (second consecutive session) |
| 23:11:41 / 23:12:13 | 3 full load, streamed / resident | both `ready` (6.7 s / 9.6 s) |
| 23:12:55 | 4 cached load 1/5, resident | `ready` in 8.9 s |
| 23:13:04 | 4 cached load 2/5, resident | interrupted at `gpu-wait`, 123/251 (`embed_tokens.chunk12`), **607 MiB** allocated, page back after 0.7 s |
| 23:13:38 / 23:13:45 | 4 cached load 1–2/5, streamed | both `ready` (7.2 s, 10.8 s; `modelLoadCall` 6.5 s against 10.0 s) |
| 23:13:56 | 4 cached load 3/5, streamed | **interrupted at `ort-plan-start` with no weights**, 4.4 s in, 0.1–0.3 s after the previous runs, page back after 1.6 s |
| 23:14:30 | 5 short inference, streamed | complete in 46.7 s, 32 tokens × 2 (20.5 s, 18.0 s), peak 498 MiB |
| 23:15:38 | 5 short inference, resident | load `ready` in 9.9 s, interrupted 1.0 s into the first prompt, 0.5 s after the first token, 1,037 MiB requested, page back after 1.2 s |

All three interruptions recorded `no-unload-event` and `navigationType:
back_forward`. Row 5 of the evaluation (50-token prompt, 137 generated tokens)
took 91.6 s: about 665 ms per token, of which the 671 MB output-projection
upload took about 230 ms and its 16 head projections about 230 ms. The five rows
read 420 GB from OPFS in 201,759 range reads of 2 MiB. At that rate one 100-row
pass takes more than two hours.

What repeated from earlier sessions:

- Resident short inference ended within a second of its first token at 1,037
  MiB requested for the third session in a row (07:35, 17:40, 23:15); the 17:40
  instance is the one Jetsam confirmed as a `highwater` kill at 2,284 MiB.
- The third rapid streamed cached load ended at `ort-plan-start` with no GPU
  weights, as at 16:34 (evening session, 3/5). Both ran with a 0 s repeat delay;
  the 30 s comparison was still not run.

What changed: the resident cached load stopped earlier than any previous
interruption (607 MiB against 740–1,060 MiB before), 2c resident completed a
second time, and the evaluation page ran five streamed rows without a restart,
the first inference session on this phone to pass its first token.

What the device files say:

- The Jetsam report is stamped 22:51:06, three and a half minutes before the
  browser's Networking process started (22:54:06 by the disk-write report) and
  before any run. It lists no browser coalition at all; Toss was in front with a
  587 MiB WebContent process, and the only kill was `CloudTelemetryService`
  (`per-process-limit`, 6.8 MiB). It documents the system state before the test
  (free 156 MiB, compressor 1,049 MiB, about 1.1 GB of suspended third-party
  apps), not any of the three interruptions, and the experiment page's matcher
  correctly attached it to no row. No JetsamEvent or WebContent crash file for
  23:13:15, 23:14:03 or 23:15:52 has been collected yet.
- The Networking process wrote 1,073.75 MB of file-backed memory through
  WebCore's SQLite storage between 22:54:06 and 23:13:01 (19 minutes, no action
  taken). In that window the pages committed about 6,400 compact journal entries
  on the evaluation page and about 40,000 in experiment runs, four durable
  transactions per staging piece at roughly 5.8 KB each. Compact mode had cut
  the earlier 4.29 GB per 90 minutes but not far enough.

Changes in this commit:

- **Streamed output upload.** `StreamedWeights` queues every scratch-sized
  write of a chunk and waits once per chunk instead of after every piece; the
  scratch follows the staging size (default 8 MiB, previously fixed at 2 MiB),
  which also applies to the streamed body load. Per generated token this is 80
  reads and 16 queue waits instead of 320 and 320. Metrics split `outputReadMs`
  and `queueWaitMs` out of `uploadMs` and count `readCalls`, `writeCalls` and
  `queueWaits`. Up to one chunk (40 MiB) of `writeBuffer` data may be in flight
  before the wait; the phone measurement decides whether that stays.
- **Durable evaluation rows.** `row-complete` records now carry the row's
  timing and ROUGE, and the journal keeps them under `rows` outside the
  64-event ring, so a cancelled or interrupted evaluation exports every finished
  row. The evaluation page accepts `rowLimit=N` for throughput checks; the
  `evaluation-rows` milestone and the summary record the limit, the result
  says `부분 평가 · 수용 기준 아님`, and a partial run is never acceptance.
- **Journal traffic.** `range-complete` is emitted in memory only (the next
  pre-call record implies it), and the three pre-call range records drop the
  ledger's `recentAllocations` list (`recentAllocationsOmitted` counts it). One
  durable `streamed-step-complete` per eight tokens during evaluations; probes
  keep every step. `persistence.serializedBytes` records the JSON length of
  committed compact entries for comparison with the phone's disk-write report.
- **Device reports.** JetsamEvent parsing adds `suspendedMiB`, `suspendedTop`,
  `frontmost` and `compressorMiB`; `reportCoverage` classifies a report as
  `before-runs`, `within-runs` or `after-runs` against the results on the page;
  every dropped file is kept under `deviceReports` in the export with its
  coverage, kills, matched rows and pressure figures, and the status line says
  when a report holds no browser process.
- **Repeats.** `diagnosticSummary.ortPlan` (time from `ort-session-start` to
  `ort-plan-start`, WASM heap at that point) is a column on the experiment page
  for comparing cached-load repeats. On iOS the page defaults to streamed
  execution and a 30 s repeat delay; stored choices are kept, so a phone that
  already saved 0 s has to select 30 s once.

Still owed from the phone: the `.ips` files for the three interruptions above,
streamed cached load ×5 with the 30 s delay, a streamed short probe and a
`rowLimit=10` evaluation on this commit to measure the upload change, and then
the two 100-row streamed evaluations.

## September 14 morning exports (release `5c7261485e7c`, commit `fae2b3a`)

Same iPhone, Chrome 153, Asyncify, nine OPFS cache hits in every run, no
persisted GPU fault, no device loss, `compact` diagnostics, 8 MiB staging and
streamed scratch, 30 s repeat delay on the experiment page. The evaluation page
was opened without parameters and therefore streamed; it was exported once
(`didimdol-diagnostics (12)`), the experiment page five times
(`iphone-fp32-experiments (65)`–`(69)`, each a prefix of the next). No device
file was collected in this session.

| KST | Screen / experiment | Outcome |
|---|---|---|
| 07:34:26 | Evaluation page session create, streamed 8 MiB | 235/235, 383 MiB weights, `ready` in 11.2 s |
| 07:34:37–07:40:53 | Evaluation page 100-row evaluation, streamed | **rows 1–6 complete in 373 s**, one auxiliary `top_k` session, peak 520 MiB; stopped by the tester in row 7 (`AbortError` at output-upload chunk 13, status `cancelled`) |
| 07:41:26–07:46:41 | 1, 1b, 1c, 1d, small runtime 120 s | all complete, 251/251 |
| 07:57:40–07:59:39 | tokenizer, 2c streamed, 2c resident, 3 streamed, 3 resident | all complete (2c 7.3 s / 9.0 s, 3 `ready` 6.1 s / 9.4 s) |
| 08:00:45–08:03:09 | 4 cached load 1–5/5, streamed, 30 s delay | 5/5 `ready`, 5.5–6.1 s |
| 08:03:46–08:06:20 | 4 cached load 1–5/5, resident, 30 s delay | 5/5 `ready`, 8.0–8.7 s |
| 08:07:09 | 5 short inference, resident | load `ready` in 9.1 s, **interrupted 38 ms after `probe-inference-start`** (row 3, before the first token), 1,072 MiB requested, page back after 0.9 s |
| 08:07:43 | 5 short inference, streamed | complete in 47.9 s, 32 tokens × 2 (20.3 s, 18.6 s), peak 522 MiB |

The interruption recorded `no-unload-event` and `navigationType: back_forward`,
like the three before it. The evaluation stop is not an interruption: the
worker caught the `AbortError` of the stop message and finished the journal as
`cancelled`; a page that went away would have left the run `running`, as the
resident probe did.

Evaluation rows, all with `eos`:

| Row | Prompt | Tokens | ms per token | TTFT | ROUGE-1 F1 |
|---|---|---|---|---|---|
| 1 | 211 | 85 | 598 | 1,113 | 0.335 |
| 2 | 37 | 213 | 465 | 732 | 0.185 |
| 3 | 35 | 110 | 476 | 729 | 0.234 |
| 4 | 217 | 71 | 516 | 1,168 | 0.393 |
| 5 | 50 | 137 | 526 | 828 | 0.320 |
| 6 | 245 | 108 | 516 | 1,305 | 0.289 |

Row 5 (the same 137-token row that took 91.6 s the night before) took 72.1 s,
and rows 1–5 took 317 s against 389 s. In the steady state a token costs about
500 ms: 261 ms uploading the 672 MB output projection (162 ms of OPFS reads in
80 calls, 19 ms in 16 queue waits, the rest `writeBuffer` copies), 140 ms in the
16 head projections and readbacks, and about 100 ms in the body graph. Row 1 is
slower (598 ms) because the output projection is not in the file cache after a
load; the streamed probe shows the same, its first prompt reading pieces in
138–346 ms before settling at 148 ms. At this rate one 100-row pass takes about
100 minutes. GPU requests stayed at 508–520 MB with about 1,030 live buffers
across the six rows; the WASM heap grew once, from 24.2 MB to 29.1 MB, in row 6.

Journal traffic: the evaluation run committed 434 entries (0.8 MB) in 375 s
against about 6,400 entries the night before, but each load still wrote 3,255
entries (6.3 MB) and the experiment session's 24 runs 73,708 entries (153 MB).

What repeated: the resident short inference ended within a second of its first
prompt for the fourth session in a row (07:35, 17:40, 23:15 on September 13,
08:07 on September 14), this time before the first token. What changed: with
the 30 s delay, ten consecutive cached loads completed (the night before, 0 s
delay ended a resident repeat at 607 MiB and a streamed repeat at
`ort-plan-start`), and the streamed evaluation's tokens cost 500 ms instead of
665 ms. The streamed probe did not change (47.9 s against 46.7 s) because its
first prompt pays the cold file cache; row-level numbers are the comparison.

Changes in this commit:

- **Overlapped output projection.** `StreamedWeights` keeps two 40 MiB output
  buffers by default; while the head session projects chunk *i* the next chunk
  is read from OPFS and queued into the other buffer. `outputBuffers=1` (the
  evaluation URL and the experiment page's selector) restores the serial path
  for comparison. The streamed GPU request grows by 40 MiB to about 463 MiB of
  weights and buffers. `streaming` gains `gpuBufferCount` and `projectionMs`,
  the wall time of the output loop; `uploadMs + outputComputeMs − projectionMs`
  is what the overlap saved. `streamed-error` records carry `computing` (the
  chunk in flight) next to `position`. Error scopes now bracket each
  `writeBuffer` synchronously, since an upload can run while the runtime's own
  `run()` is pending on the same device. Not yet measured on the phone.
- **Position records.** The pre-call records (`upload-initializer`,
  `range-read`, `gpu-write`, `gpu-wait`, and the resident probe's read, write
  and wait) carry the position and progress counters only, with a slim ledger
  (`positionRecord: true`), and update the durable head's `last` without a ring
  slot. The ring keeps the allocation and completion of every initializer and
  numbers its slots by `eventCount`; older journals still read. The fixed
  persistence note is attached by `readRun` instead of being written with every
  head. Expected: about 2,100 entries and under 2 MB per full load.
- **Stop source.** A run ended by the page's stop message finishes with
  `stopSource: 'user-stop'` and, for evaluations, `completedRows`; both pages
  show "사용자 정지 · N행 완료" instead of a bare abort message.
- **Resident inference note.** The experiment page shows a warning, not a
  block, when resident execution is selected for a probe or evaluation on iOS.

Still owed from the phone: the `.ips` files for the four resident interruptions
(the latest 08:07:19), a streamed probe and a `rowLimit=10` evaluation with
`outputBuffers=2` against `outputBuffers=1` on this commit, a load export to
confirm the journal size, and then the two 100-row streamed evaluations.

## September 14 evening exports (release `8a6116bb9a49`, commit `225f1bb`)

Same iPhone, Chrome 153, Asyncify, nine OPFS cache hits in every run, no
persisted GPU fault, no device loss, `compact` diagnostics, 8 MiB staging and
streamed scratch, 30 s repeat delay. This build projected the streamed output
with two buffers by default. The evaluation page was opened without parameters
(streamed, two buffers) and exported once (`didimdol-diagnostics (13)`), the
experiment page three times (`iphone-fp32-experiments (70)`–`(72)`, each a
prefix of the next). No device file was collected in this session.

| KST | Screen / experiment | Outcome |
|---|---|---|
| 21:36:47 | Evaluation page session create, streamed 8 MiB, 2 buffers | 235/235, 383 MiB weights, `ready` in 11.1 s; load journal 2,315 entries, 3.95 MB |
| 21:36:59–21:44:03 | Evaluation page 100-row evaluation, streamed, 2 buffers | **rows 1–5 complete in 421 s** (317 s in the morning with one buffer); stopped by the tester in row 6 after its first token, recorded as `stopSource: user-stop`, `completedRows: 5`, status `cancelled` |
| 21:44:52 | 5 short inference, streamed, 2 buffers | load `ready` in 10.3 s; **interrupted 22 tokens into the first prompt** (21:45:25, 728 ms per token), 471 MiB requested; a `pagehide` with `persisted: true` and a hidden `visibilitychange` 7 ms apart, 0.5 s after the last checkpoint; page back 14 s later |
| 21:45:39 | 5 short inference, streamed, 2 buffers | complete in 59.7 s, 32 tokens × 2 (27.9 s, 24.6 s), peak 538 MiB |
| 21:49:13 | 5 short inference, resident | load `ready` in 14.6 s, first token at 435 ms, **interrupted 1.0 s after `probe-inference-start`** (three samples), 1,036 MiB requested, no unload event |
| 21:50:30 | 5 short inference, streamed, `outputBuffers=1` | complete in 48.1 s, 32 tokens × 2 (22.3 s, 17.4 s), peak 522 MiB |

The three probes and the morning probe generated identical token sequences, and
rows 4 and 5 of the evaluation scored the same ROUGE as in the morning (0.393,
0.320), so the second buffer changed nothing but time.

Streaming totals of the 64-projection probes:

| | Morning, 1 buffer | Evening, 2 buffers | Evening, 1 buffer |
|---|---:|---:|---:|
| Probe | 47.9 s | 59.7 s | 48.1 s |
| `projectionMs` (output loop wall) | — | 39.9 s | 30.0 s |
| `uploadMs` | 20.5 s | 38.7 s | 20.0 s |
| `outputReadMs` | 12.1 s | 19.0 s | 11.4 s |
| `queueWaitMs` | 1.5 s | 7.9 s | 1.7 s |
| `outputComputeMs` | 9.6 s | 34.7 s | 10.0 s |

Evaluation rows, both with `eos`:

| Row | Prompt | Tokens | ms per token (morning) | TTFT (morning) | ROUGE-1 F1 |
|---|---|---|---|---|---|
| 4 | 217 | 71 | 703 (516) | 1,160 (1,168) | 0.393 |
| 5 | 50 | 137 | 678 (526) | 878 (828) | 0.320 |

With one buffer `uploadMs + outputComputeMs` equals `projectionMs` exactly
(20.0 + 10.0 = 30.0 s): 19.5 ms of upload and 9.8 ms of projection per 40 MiB
chunk. With two buffers both intervals span nearly the whole chunk period, so
the overlap did happen, but the period grew from 29 ms to 39 ms per chunk. The
OPFS reads themselves took 67% longer for the same bytes, the queue wait 4.5×,
and the compute interval 3.5× (mostly waiting, since the GPU work per chunk is
unchanged). Three mechanisms in that loop fit the numbers: the upload of chunk
*i+1* called `onSubmittedWorkDone`, which now waited for projection *i* as
well; the readback of chunk *i* (`getData`, a copy plus `mapAsync` on the same
in-order queue) was submitted after the 40 MiB of `writeBuffer` calls for chunk
*i+1* and therefore waited behind them, serialising compute(i), blit(i+1),
readback(i), compute(i+1) with no gain; and the synchronous OPFS reads and
`writeBuffer` copies competed with the GPU process's blit and the
bandwidth-bound M=1 Gemm for memory bandwidth. The error scopes are not the
cause: the serial path on the same build matched the morning (48.1 s against
47.9 s). TTFT did not change, so prefill is unaffected.

The interruptions differ. The evaluation stop is a tester stop (`user-stop`,
five rows). The resident probe is the fifth resident interruption in a row
(07:35, 17:40, 23:15 on September 13, 08:07 and 21:49 on September 14), this
time after a first token in 435 ms and about 250 ms per token, with no unload
event, which matches a process termination. The streamed probe at 21:44 is the
first interruption with an unload event: `pagehide` with `persisted: true` and
a hidden `visibilitychange` in the same moment, half a second after a
checkpoint, which matches the page leaving the foreground (an app switch,
lock screen or tab change) and a background discard before it was reopened 14 s
later, not a crash in the foreground. Whether the tester left the app at
21:45:25 is unconfirmed.

Journal traffic: the load journal was 2,315 entries and 3.95 MB (3,255 and
6.3 MB the morning before). Every `allocate-initializer` and
`initializer-complete` ring record carried the ledger's 16 `recentAllocations`
(3.4 KB of its 5.2 KB), 1.6 MB of the total. The evaluation run wrote 415
entries (0.8 MB) in 7 minutes and the eight experiment runs 10,012 entries
(16.9 MB) against 73,708 (153 MB) for the morning's 24.

Changes in this commit:

- **One output buffer by default.** `DEFAULT_OUTPUT_BUFFERS` is 1 in the
  worker, the evaluation URL (`outputBuffers=2` selects the overlap) and the
  experiment page; the release-independent stored choice on the phone still
  applies.
- **Reordered overlap.** With two buffers the loop calls `getData()` as soon as
  `run()` returns, so the readback copy is queued before the next chunk's
  writes (the runtime's downloader encodes and submits synchronously), and
  only then starts the next upload, which skips its queue wait
  (`upload(index, { queueWait: false })`): the following readback bounds the
  backlog and the buffer being rewritten held a projection already read back.
  The serial path drops the duplicate wait after the readback, which the
  in-order queue makes redundant.
- **Chunk timelines.** The first two projections of a session record every
  chunk's `runSubmit`, `runResolved`, `readbackSubmit`, `readbackResolved`,
  `uploadStart`, `uploadEnd`, `readMs` and `waitMs` (ms from the loop start);
  the next `streamed-step-complete` carries them as `chunkTimelines`, about
  1.5 KB each. Evaluations, which checkpoint every eight tokens, get both on
  their first checkpoint.
- **Ring ledger.** `allocate-initializer` and `initializer-complete` records
  drop `recentAllocations` (`recentAllocationsOmitted` counts it) like the
  position records; milestones and the terminal summary keep the list.
  Expected: about 2.3 MB per full load.
- **Unload kind.** Recovery records `unloadKind`: `persisted-hide` when a
  `pagehide` with `persisted: true` fell inside the run, `unload` for other
  navigation or reload events; both pages label it and show the re-entry gap
  and navigation type alongside.
- **Browser test.** `npm run test:streamed` runs the streamed probe with one
  and two buffers and checks identical tokens, the queue-wait counts and the
  timelines.

Still owed from the phone: a streamed probe and a `rowLimit=10` evaluation with
`outputBuffers=2` on this commit against the one-buffer defaults, with their
`chunkTimelines`; the `.ips` files for the 21:49 resident interruption and the
21:45 streamed one; whether the tester left the app at 21:45:25; a load export
to confirm the journal size; and then the two 100-row streamed evaluations.

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
fault remain separately available. Pre-call position records (`upload-initializer`,
`range-read`, `gpu-write`, `gpu-wait`, `resident-read`, `resident-write`,
`resident-wait`) update the head's `last` only and take no ring slot, so the
ring holds the allocation and completion of about 32 initializers rather than
the pieces of the last ten; ring slots are numbered by `eventCount`, and
journals numbered by `recordCount` still read. The allocation and completion
records carry the full metrics and storage snapshot but not the ledger's
`recentAllocations` list (`recentAllocationsOmitted` counts it); milestones
and the terminal summary keep it. Initializer order is capped at 2,048 entries.
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
