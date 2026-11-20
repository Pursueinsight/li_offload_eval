#!/usr/bin/env bash
# Sweep all presets over core counts (with and without SMT), then aggregate
# results/_chart_summary.csv. scan_cpu.py refuses to overwrite a non-empty --out,
# so finished directories are skipped instead of failing the whole run.
set -euo pipefail
cd "$(dirname "$0")"

declare -A TILE=([decode_16k]=128 [decode_32k]=128 [decode_64k]=128 [decode_128k]=128 \
                 [batch8_64k]=256 [batch32_64k]=512 [prefill128_16k]=128 [regression_a2]=128)
declare -A QT=([prefill128_16k]=8)
LOADS=(decode_16k decode_32k decode_64k decode_128k batch8_64k batch32_64k prefill128_16k regression_a2)

for load in "${LOADS[@]}"; do
  [[ -d cases/$load ]] || python3 generate_case.py --preset "$load" --out "cases/$load"
  for c in 1 16 48 96 192; do
    case $c in 1|16|48) node=0 ;; 96) node=0,1 ;; 192) node=0,1,2,3 ;; esac
    for smt in "" "--smt"; do
      out=results/_chart/${load}_c${c}${smt:+_smt}
      if [[ -d $out ]]; then
        echo "skip existing $out"
        continue
      fi
      # shellcheck disable=SC2086  # $smt must expand to zero or one flag
      python3 scan_cpu.py --case "cases/$load" --out "$out" \
        --node "$node" --cores "$c" $smt \
        --core-policy balanced --memory-policy auto --wait-policy ACTIVE \
        --tile "${TILE[$load]}" --query-tile "${QT[$load]:-1}" --hist histseg \
        --score-backend svebf16 --warmup 5 --iters 60 --verify golden
    done
  done
done

python3 - <<'EOF'
import csv, glob, json, os
rows = []
for d in sorted(glob.glob('results/_chart/*/')):
    load, rest = os.path.basename(d.rstrip('/')).rsplit('_c', 1)
    smt = rest.endswith('_smt')
    scan = json.load(open(d + 'scan.json'))
    for r in scan['rows']:
        result = json.load(open(f"{d}cores_{r['physical_cores']}/result.json"))
        p = result['parameters']
        rows.append(dict(load=load, cores=r['physical_cores'], smt=smt, threads=r['threads'],
                         real_cores=result['actual_threads'], native_us=r['native_us'],
                         host_us=r['host_us'], score_us=r['score_us'], topk_us=r['topk_us'],
                         prepare_us=r['prepare_us'], tile=p['tile'], query_tile=p['query_tile'],
                         backend=p['score_backend'], groups=r['topk_groups_actual'],
                         vl=';'.join(map(str, sorted(set(result['worker_vl_bits']))))))
with open('results/_chart_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} rows -> results/_chart_summary.csv')
EOF
