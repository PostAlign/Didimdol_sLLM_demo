#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-asyncify}
case "$mode" in asyncify|jspi) ;; *) echo 'Usage: tools/build-ort.sh [asyncify|jspi]' >&2; exit 2;; esac
ort_source=${ORT_SOURCE:-"$task_root/.work/onnxruntime"}
ort_commit=b7804b056c30aa35c1748f8e4e239d0e2ff25d6d
mkdir -p "$task_root/.work"
if [[ ! -d "$ort_source/.git" ]]; then
  git init "$ort_source"
  git -C "$ort_source" remote add origin https://github.com/microsoft/onnxruntime.git
  git -C "$ort_source" fetch --depth 1 origin "$ort_commit"
  git -C "$ort_source" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$ort_source" rev-parse HEAD) == "$ort_commit" ]] || { echo 'ORT source commit mismatch' >&2; exit 1; }
patch="$task_root/patches/ort-session-range-loader.patch"
if git -C "$ort_source" apply --check "$patch" 2>/dev/null; then
  git -C "$ort_source" apply "$patch"
elif ! git -C "$ort_source" apply --reverse --check "$patch"; then
  echo 'ORT patch conflicts with checkout; inspect it before rebuilding.' >&2
  exit 1
fi
extra=()
[[ "$mode" != jspi ]] || extra+=(--enable_wasm_jspi)
python3 "$ort_source/tools/ci_build/build.py" \
  --config Release --update --build --parallel "${ORT_JOBS:-4}" \
  --build_wasm --enable_wasm_simd --enable_wasm_threads --use_webgpu \
  --disable_rtti --skip_tests --target onnxruntime_webassembly \
  --build_dir "$task_root/.work/ort-build-$mode" --cmake_generator Ninja \
  --include_ops_by_config "$ort_source/onnxruntime/wasm/reduced_types.config" \
  --enable_reduced_operator_type_support "${extra[@]}" \
  --cmake_extra_defines CMAKE_CXX_STANDARD=20 \
    "NODE_EXECUTABLE=$(command -v node)" "NPM_CLI=$(command -v npm)"
ORT_SOURCE="$ort_source" ORT_MODE="$mode" node "$task_root/tools/build-runtime.mjs"
