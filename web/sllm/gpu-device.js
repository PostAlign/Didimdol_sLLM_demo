/** Allocation requests and explicit destroy calls, not physical GPU/process RAM. */
export async function installGpuTracking(largestTensorBytes, emit = () => {}) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  if (Math.min(adapter.limits.maxBufferSize, adapter.limits.maxStorageBufferBindingSize) < largestTensorBytes) {
    throw new Error('Largest tensor exceeds WebGPU device limits');
  }
  let device;
  const ledger = { requestedCurrent: 0, requestedPeak: 0, mappedUploadRequested: 0, bufferCount: 0, deviceLost: null };
  function track(created) {
    device = created;
    const create = device.createBuffer.bind(device);
    device.createBuffer = descriptor => {
      emit({ stage: 'gpu-create-buffer-start', bytes: descriptor.size, totalAllocated: ledger.requestedCurrent });
      const buffer = create(descriptor);
      ledger.requestedCurrent += descriptor.size;
      ledger.requestedPeak = Math.max(ledger.requestedPeak, ledger.requestedCurrent);
      ledger.bufferCount++;
      if (descriptor.mappedAtCreation) ledger.mappedUploadRequested += descriptor.size;
      emit({ stage: 'gpu-create-buffer', bytes: descriptor.size, totalAllocated: ledger.requestedCurrent });
      const destroy = buffer.destroy.bind(buffer);
      let destroyed = false;
      buffer.destroy = () => {
        if (!destroyed) { ledger.requestedCurrent -= descriptor.size; destroyed = true; }
        destroy();
      };
      return buffer;
    };
    device.lost.then(info => {
      ledger.deviceLost = { reason: info.reason, message: info.message };
      emit({ stage: 'device-lost', info: ledger.deviceLost });
    });
    device.addEventListener('uncapturederror', event => emit({ stage: 'gpu-uncaptured-error', message: event.error.message }));
    return device;
  }
  // Observe ORT's own device creation in this dedicated worker. Keep the native
  // default-instance path (which enables TimedWaitAny), rather than importing a
  // custom device through the pinned runtime's wgpuCreateInstance(0) path.
  const originalRequest = navigator.gpu.requestAdapter;
  navigator.gpu.requestAdapter = async function (...args) {
    const found = await originalRequest.apply(this, args);
    if (found) {
      const requestDevice = found.requestDevice.bind(found);
      found.requestDevice = async descriptor => track(await requestDevice(descriptor));
    }
    return found;
  };
  return { get device() { return device; }, ledger,
    restore() { navigator.gpu.requestAdapter = originalRequest; },
    adapterInfo: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture } };
}
