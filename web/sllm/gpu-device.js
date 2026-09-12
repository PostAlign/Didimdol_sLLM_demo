import { errorDetails } from './diagnostics.js';

/** Observe ORT's native device creation in a dedicated worker. Counters describe
 * requests/explicit destruction, never physical GPU memory or process RSS. */
export async function installGpuTracking(largestTensorBytes, emit = () => {}, options = {}) {
  // Retain the GPU wrapper as well as intercepting prototypes: an instance-only
  // requestAdapter hook is insufficient when the runtime uses another wrapper.
  const gpu = options.gpu ?? globalThis.navigator?.gpu;
  const adapter = await gpu?.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  if (Math.min(adapter.limits.maxBufferSize, adapter.limits.maxStorageBufferBindingSize) < largestTensorBytes) {
    throw new Error('Largest tensor exceeds WebGPU device limits');
  }
  const context = options.context ?? (() => ({}));
  const deviceStates = new Map(), buffers = new WeakMap(), adapters = new WeakSet();
  const patches = [], pending = new Set(), reasons = new Set();
  const counts = { requestedCurrent: 0, observedPeak: 0, mappedUploadRequested: 0, bufferCount: 0,
    liveBufferCount: 0, peakLiveBufferCount: 0, lateBufferCount: 0,
    deviceLost: null, lastError: null, firstError: null };
  let primary, restored = false;
  const watch = promise => {
    const task = Promise.resolve(promise).catch(() => { reasons.add('diagnostic-callback-failed'); });
    pending.add(task); task.then(() => pending.delete(task));
  };
  const publish = record => {
    try { const result = emit(record); if (result?.then) watch(result); }
    catch { reasons.add('diagnostic-callback-failed'); }
  };
  function fault(error, details) {
    const record = { ...details, ...errorDetails(error) };
    counts.firstError ||= record; counts.lastError = record.message;
    publish({ ...record, stage: details.stage || 'gpu-error' });
  }
  function patch(target, key, wrap) {
    if (!target || target === Object.prototype || typeof target[key] !== 'function') return null;
    const descriptor = Object.getOwnPropertyDescriptor(target, key);
    const wrapper = wrap(target[key]);
    try {
      Object.defineProperty(target, key, { configurable: true, writable: true, value: wrapper });
      patches.push({ target, key, descriptor, wrapper });
      return wrapper;
    } catch { return null; }
  }
  function markReleased(buffer) {
    const entry = buffers.get(buffer);
    if (!entry || entry.destroyed || entry.owner.destroyed) return;
    entry.destroyed = true;
    counts.requestedCurrent -= entry.bytes; counts.liveBufferCount--;
    entry.owner.current -= entry.bytes; entry.owner.live--;
  }
  function destroyBuffer(original) {
    return function (...args) { const result = original.apply(this, args); markReleased(this); return result; };
  }
  function destroyDevice(original) {
    return function (...args) {
      const result = original.apply(this, args), state = deviceStates.get(this);
      if (state && !state.destroyed) {
        counts.requestedCurrent -= state.current; counts.liveBufferCount -= state.live;
        state.current = 0; state.live = 0; state.destroyed = true;
      }
      return result;
    };
  }
  function trackBuffer(device, buffer, descriptor, late = false) {
    if (buffers.has(buffer)) return;
    const owner = trackDevice(device, false);
    const bytes = descriptor.size;
    buffers.set(buffer, { owner, bytes, destroyed: false });
    owner.current += bytes; owner.live++;
    counts.requestedCurrent += bytes; counts.bufferCount++; counts.liveBufferCount++;
    counts.observedPeak = Math.max(counts.observedPeak, counts.requestedCurrent);
    counts.peakLiveBufferCount = Math.max(counts.peakLiveBufferCount, counts.liveBufferCount);
    if (late) {
      counts.lateBufferCount++; owner.fullHistory = false;
      reasons.add('buffer-observed-after-allocation');
    } else if (descriptor.mappedAtCreation) counts.mappedUploadRequested += bytes;
    if (!bufferDestroyHook || buffer.destroy !== bufferDestroyHook) {
      if (!patch(buffer, 'destroy', destroyBuffer)) reasons.add('buffer-destroy-hook-unavailable');
    }
  }
  function createBuffer(original) {
    return function (descriptor) {
      const owner = trackDevice(this, false);
      const details = { ...context(), operation: 'createBuffer', deviceId: owner.id,
        bytes: descriptor.size, allocatedBefore: counts.requestedCurrent };
      let scopes = 0;
      try {
        // Pop both scopes synchronously below so they cannot capture a later
        // upload or ORT operation. Only their results are awaited asynchronously.
        this.pushErrorScope('out-of-memory'); scopes++;
        this.pushErrorScope('validation'); scopes++;
      } catch { reasons.add('allocation-error-scopes-unavailable'); }
      publish({ ...details, stage: 'gpu-create-buffer-start' });
      try {
        const buffer = original.call(this, descriptor);
        trackBuffer(this, buffer, descriptor);
        publish({ ...details, stage: 'gpu-create-buffer', totalAllocated: counts.requestedCurrent });
        return buffer; // preserve the synchronous WebGPU API
      } catch (error) { fault(error, details); throw error; }
      finally {
        while (scopes-- > 0) {
          try {
            watch(this.popErrorScope().then(error => { if (error) fault(error, details); },
              error => fault(error, { ...details, operation: 'createBuffer-error-scope' })));
          } catch (error) { fault(error, { ...details, operation: 'createBuffer-error-scope' }); }
        }
      }
    };
  }
  function trackDevice(device, fromCreation) {
    let state = deviceStates.get(device);
    if (state) return state;
    state = { id: deviceStates.size + 1, fullHistory: fromCreation, current: 0, live: 0, destroyed: false };
    deviceStates.set(device, state); primary ||= device;
    if (!fromCreation) reasons.add('device-observed-after-creation');
    if (!deviceCreateHook || device.createBuffer !== deviceCreateHook) {
      if (!patch(device, 'createBuffer', createBuffer)) reasons.add('buffer-create-hook-unavailable');
    }
    if (!deviceDestroyHook || device.destroy !== deviceDestroyHook) {
      if (!patch(device, 'destroy', destroyDevice)) reasons.add('device-destroy-hook-unavailable');
    }
    device.lost.then(info => {
      if ((restored || state.destroyed) && info.reason === 'destroyed') return;
      counts.deviceLost = { deviceId: state.id, reason: info.reason, message: info.message };
      publish({ ...context(), stage: 'device-lost', deviceId: state.id, info: { ...counts.deviceLost } });
    });
    device.addEventListener('uncapturederror', event => {
      fault(event.error, { ...context(), stage: 'gpu-uncaptured-error', deviceId: state.id, operation: 'uncaptured' });
    });
    return state;
  }
  function requestDevice(original) {
    return async function (...args) {
      const device = await original.apply(this, args);
      trackDevice(device, true);
      return device;
    };
  }
  function trackAdapter(found) {
    if (!found || adapters.has(found)) return found;
    adapters.add(found);
    if (!adapterRequestHook || found.requestDevice !== adapterRequestHook) {
      if (!patch(found, 'requestDevice', requestDevice)) reasons.add('device-request-hook-unavailable');
    }
    return found;
  }
  const deviceCreateHook = patch(options.devicePrototype ?? globalThis.GPUDevice?.prototype, 'createBuffer', createBuffer);
  const deviceDestroyHook = patch(options.devicePrototype ?? globalThis.GPUDevice?.prototype, 'destroy', destroyDevice);
  const bufferDestroyHook = patch(options.bufferPrototype ?? globalThis.GPUBuffer?.prototype, 'destroy', destroyBuffer);
  const adapterRequestHook = patch(options.adapterPrototype ?? Object.getPrototypeOf(adapter), 'requestDevice', requestDevice);
  trackAdapter(adapter);
  patch(gpu, 'requestAdapter', original => async function (...args) { return trackAdapter(await original.apply(this, args)); });

  function snapshot() {
    const complete = deviceStates.size > 0 && !reasons.size && [...deviceStates.values()].every(s => s.fullHistory);
    return { ...counts, requestedPeak: complete ? counts.observedPeak : null,
      observedPeakLiveBufferCount: counts.peakLiveBufferCount,
      peakLiveBufferCount: complete ? counts.peakLiveBufferCount : null,
      observedMappedUploadRequested: counts.mappedUploadRequested,
      mappedUploadRequested: complete ? counts.mappedUploadRequested : null,
      tracking: { status: complete ? 'complete' : deviceStates.size ? 'partial' : 'unbound',
        reasons: [...reasons], activeDeviceId: deviceStates.get(primary)?.id ?? null,
        deviceCount: deviceStates.size, pendingChecks: pending.size,
        devices: [...deviceStates.values()].map(s => ({ ...s })) } };
  }
  return {
    get device() { return primary; },
    get devices() { return [...deviceStates.keys()]; },
    get ledger() { return snapshot(); },
    // Bind the actual native device/buffer even if interception was bypassed.
    observeBuffer(device, buffer) {
      trackDevice(device, false); primary = device;
      trackBuffer(device, buffer, { size: buffer.size }, true);
      return snapshot();
    },
    async flush() { while (pending.size) await Promise.all([...pending]); },
    restore() {
      if (restored) return;
      restored = true;
      for (const { target, key, descriptor, wrapper } of patches.reverse()) {
        if (target[key] !== wrapper) continue;
        if (descriptor) Object.defineProperty(target, key, descriptor); else delete target[key];
      }
    },
    adapterInfo: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture },
  };
}
