import { readRelease } from '../release-manifest.js';
try {
  const release = await readRelease();
  await import(release.asset('web/sllm/experiments/device-runner.js'));
} catch (error) {
  document.getElementById('status').textContent = `실행 파일을 준비하지 못했습니다: ${error.message}`;
  document.getElementById('start').disabled = true;
}
