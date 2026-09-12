#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-asyncify}
profile=${ORT_PROFILE:-mobile}
case "$profile" in mobile|baseline) ;; *) echo 'ORT_PROFILE must be mobile or baseline'; exit 2;; esac
threads=${ORT_THREADS:-0}
case "$threads" in 0|1) ;; *) echo 'ORT_THREADS must be 0 or 1'; exit 2;; esac
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
[[ "$threads" != 1 ]] || extra+=(--enable_wasm_threads)
ops="$task_root/model/required-operators.config"
[[ "$profile" != baseline ]] || ops="$ort_source/onnxruntime/wasm/reduced_types.config"
build_dir="$task_root/.work/ort-build-$mode-$profile-t$threads"
build_args=( "$ort_source/tools/ci_build/build.py" \
  --config Release --update --build --parallel "${ORT_JOBS:-4}" \
  --build_wasm --enable_wasm_simd --use_webgpu \
  --disable_rtti --skip_tests --target onnxruntime_webassembly \
  --build_dir "$build_dir" --cmake_generator Ninja \
  --include_ops_by_config "$ops" \
  --enable_reduced_operator_type_support "${extra[@]}" \
  --cmake_extra_defines CMAKE_CXX_STANDARD=20 \
    "NODE_EXECUTABLE=$(command -v node)" "NPM_CLI=$(command -v npm)" )
python3 "${build_args[@]}"
node "$task_root/tools/record-ort-build.mjs" "$build_dir/Release" "$ort_source" python3 "${build_args[@]}"
ORT_SOURCE="$ort_source" ORT_MODE="$mode" ORT_PROFILE="$profile" ORT_THREADS="$threads" \
  ORT_ARTIFACTS="$build_dir/Release" node "$task_root/tools/build-runtime.mjs"
