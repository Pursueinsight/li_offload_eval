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

# GCC 14+/Clang 16+ are the first releases with usable SME ACLE support.
# GCC 12/13 ship arm_sme.h but silently ignore the ZA/streaming attributes.
sme_capable() {
  printf '%s\n' \
    '#include <arm_sme.h>' \
    '#if defined(__clang__)' \
    '#define LI_LS __arm_locally_streaming' \
    '#define LI_NZ __arm_new("za")' \
    '#else' \
    '#define LI_LS __attribute__((arm_locally_streaming))' \
    '#define LI_NZ __attribute__((arm_new("za")))' \
    '#endif' \
    'LI_LS LI_NZ void fp(void){svzero_za();}' \
  | "$1" -march=armv9-a+sve2+bf16+sme -x c++ -fsyntax-only - >/dev/null 2>&1
}
cc="${CXX:-g++}"
if [[ "${score_impl}" == "sme" ]] && ! sme_capable "${cc}"; then
  if [[ -n "${CXX:-}" ]]; then
    echo "CXX=${CXX} cannot compile SME ACLE (-march=...+sme); use GCC 14+ or Clang 16+" >&2
    exit 3
  fi
  found=''
  for cand in clang++ clang++-18 clang++-17 clang++-16 g++-15 g++-14; do
    if command -v "${cand}" >/dev/null 2>&1 && sme_capable "${cand}"; then
      found="${cand}"; break
    fi
  done
  if [[ -z "${found}" ]]; then
    echo 'No SME-capable compiler found (need GCC 14+ or Clang 16+ with arm_sme.h)' >&2
    exit 3
  fi
  cc="${found}"
fi

"${cc}" -O3 -std=c++17 -Wall -Wextra -fopenmp -fPIC -shared "${arch[@]}" "${defines[@]}" "${includes[@]}" native/cpu_li.cpp -o build/libli_cpu.so "${blas[@]}"
echo "Built build/libli_cpu.so (ABI 3, score_impl=${score_impl}, compiler=${cc}); scalar-only build requires --hist scalar in cpu_bench.py"
