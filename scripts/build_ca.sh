#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
CANN_ROOT="${LI_CANN_ROOT:-/home/Ascend/cann-9.2.0}"
LIB_DIR="${CANN_ROOT}/aarch64-linux/lib64"
# -L resolves direct -l inputs; -rpath-link resolves their shared dependencies.
# Do not embed runtime RPATH: the existing CA environment selects simulator libraries.
link_dirs=("${LIB_DIR}")
for dir in "${LIB_DIR}/plugin/opskernel" "${LIB_DIR}/plugin/nnengine"; do
  [[ ! -d "$dir" ]] || link_dirs+=("$dir")
done
IFS=: read -r -a extra_dirs <<< "${LI_CANN_LINK_DIRS:-}"
for dir in "${extra_dirs[@]}"; do
  [[ -z "$dir" ]] || link_dirs+=("$dir")
done
link_flags=()
for dir in "${link_dirs[@]}"; do
  link_flags+=("-L${dir}" "-Wl,-rpath-link,${dir}")
done
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra native/li_aclnn_bench.cpp -o build/li_aclnn_bench \
  "-I${CANN_ROOT}/include" "${link_flags[@]}" \
  -lopapi -lnnopbase -lascendcl -lruntime -lpthread
