function pause(ms, signal) {
  return new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timer); signal?.removeEventListener('abort', abort);
      if (error) reject(error); else resolve();
    };
    const abort = () => finish(signal.reason);
    const timer = setTimeout(() => finish(), ms);
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}

/** The caller owns and retains the session until observation and cleanup finish. */
export async function observeSession({ seconds, signal, checkpoint, sample = () => ({}),
  clock = () => performance.now(), wait = pause }) {
  if (!Number.isFinite(seconds) || seconds < 0 || seconds > 120) throw new Error('Session observation must be 0–120 seconds');
  if (!seconds) return { idleSeconds: 0, idleElapsedMs: 0, idleAcceptanceCompleted: false };
  signal?.throwIfAborted();
  const started = clock();
  await checkpoint({ stage: 'session-idle-start', idleSeconds: seconds, idleElapsedMs: 0, ...sample() });
  while (clock() - started < seconds * 1000) {
    await wait(Math.min(5000, seconds * 1000 - (clock() - started)), signal);
    signal?.throwIfAborted();
    await checkpoint({ stage: 'session-idle', idleSeconds: seconds, idleElapsedMs: clock() - started, ...sample() });
  }
  signal?.throwIfAborted();
  const summary = { idleSeconds: seconds, idleElapsedMs: clock() - started,
    idleAcceptanceCompleted: seconds >= 120 && clock() - started >= 120000 };
  await checkpoint({ stage: 'session-idle-complete', ...summary, ...sample() });
  signal?.throwIfAborted();
  return summary;
}
