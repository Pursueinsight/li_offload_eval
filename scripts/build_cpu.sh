#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
arch=(-march=armv9-a+sve2+bf16)
blas=()
defines=()
score_impl=sve2
while (($#)); do
  case "$1" in
    --scalar)
      if [[ "${score_impl}" == "sme" ]]; then
        echo '--scalar cannot be combined with --sme' >&2; exit 2
      fi
      arch=()
      score_impl=scalar
      ;;
    --sme)
      if ((${#arch[@]}==0)); then
        echo '--sme cannot be combined with --scalar' >&2; exit 2
      fi
      arch=(-march=armv9-a+sve2+bf16+sme)
      defines+=(-DLI_SCORE_USE_SME)
      score_impl=sme
      ;;
    --blas)
      defines+=(-DLI_USE_BLAS)
      read -r -a blas <<< "${BLAS_LIBS:--lopenblas}"
      ;;
    *) echo 'Usage: bash scripts/build_cpu.sh [--scalar] [--sme] [--blas]; optional CXX, BLAS_LIBS, BLAS_CFLAGS' >&2;exit 2 ;;
  esac
  shift
done
read -r -a includes <<< "${BLAS_CFLAGS:-}"
"${CXX:-g++}" -O3 -std=c++17 -Wall -Wextra -fopenmp -fPIC -shared "${arch[@]}" "${defines[@]}" "${includes[@]}" native/cpu_li.cpp -o build/libli_cpu.so "${blas[@]}"
echo "Built build/libli_cpu.so (ABI 3, score_impl=${score_impl}); scalar-only build requires --hist scalar in cpu_bench.py"
