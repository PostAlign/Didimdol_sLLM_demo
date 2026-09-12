// Stable entry point. All application imports/workers/data below it use one snapshot.
import { readRelease } from './sllm/release-manifest.js';
try {
  const release = await readRelease();
  await import(release.asset('web/app-main.js'));
} catch (error) {
  console.error(error);
  const phase = document.getElementById('phase');
  if (phase) phase.textContent = `실행 파일을 준비하지 못했습니다. 새로고침해 주세요. (${error.message})`;
}
