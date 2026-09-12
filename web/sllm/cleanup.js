/** Best-effort cleanup with a single deadline; device destruction and hook
 * restoration still run if session release or a queue completion never settles. */
export async function disposeResources({ releaseSession, store, loader, tracker, timeoutMs = 4000 } = {}) {
  const started = performance.now(), errors = [];
  let timedOut = false;
  const attempt = async (name, operation) => {
    let timer;
    try {
      await Promise.race([
        Promise.resolve().then(operation),
        new Promise((_, reject) => {
          timer = setTimeout(() => { timedOut = true; reject(new Error('cleanup deadline exceeded')); },
            Math.max(0, timeoutMs - (performance.now() - started)));
        }),
      ]);
    } catch (error) { errors.push({ operation: name, message: String(error?.message || error) }); }
    finally { clearTimeout(timer); }
  };
  try {
    try { store?.close(); } catch (error) { errors.push({ operation: 'store-close', message: String(error) }); }
    try { if (loader && !loader.closed) loader.close(false); }
    catch (error) { errors.push({ operation: 'loader-close', message: String(error) }); }
    if (releaseSession) await attempt('session-release', releaseSession);
    if (!timedOut) await attempt('queue-completion', () => Promise.all((tracker?.devices || []).map(device => device.queue.onSubmittedWorkDone())));
    if (!timedOut) await attempt('gpu-checks', () => tracker?.flush());
  } finally {
    for (const device of tracker?.devices || []) {
      try { device.destroy(); } catch (error) { errors.push({ operation: 'device-destroy', message: String(error) }); }
    }
    try { tracker?.restore(); } catch (error) { errors.push({ operation: 'tracking-restore', message: String(error) }); }
  }
  return { success: errors.length === 0, timedOut, errors, durationMs: performance.now() - started,
    gpuLedger: tracker?.ledger ?? null };
}
