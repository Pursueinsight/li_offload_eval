"""Join one CA model result with a CPU core scan. Ratios remain provisional if scope/order/accuracy is unresolved."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from caseio import Case,save_json,sha256
from validation import selection_on_scores

def build_report(case_path,cpu_scan,ca_result,out,transport_us=(0,5,10,20),atol=1e-3,rtol=1e-4):
    case=Case(case_path);scan=json.loads(Path(cpu_scan).read_text());ca=json.loads(Path(ca_result).read_text());ca_dir=Path(ca_result).resolve().parent
    if ca['case_id']!=case.id or scan['case_id']!=case.id:raise ValueError('CPU/CA inputs differ: case_id mismatch')
    for name,h in ca.get('output_sha256',{}).items():
        if sha256(ca_dir/name)!=h:raise ValueError('CA output changed after collection')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];sensitivity=[]
    if ca.get('timing_scope_confirmed'):
        reference=ca.get('model_li_latency_us');reference_kind='confirmed_li_critical_path'
    elif 'li_latency_us' in ca:
        reference=ca.get('li_latency_us');reference_kind=ca.get('li_latency_kind','unavailable')
    else:
        # Legacy files did not gate their candidate on single-call completion.
        reference=None;reference_kind='legacy_result_requires_recollection'
    ca_idx=np.fromfile(ca_dir/ca['indices_file'],dtype='<i4').reshape(case.B,case.S,1,case.K) if ca.get('indices_file') else None
    for point in scan['rows']:
        path=Path(point['result']);cpu=json.loads(path.read_text())
        for name,h in cpu.get('output_sha256',{}).items():
            if sha256(path.parent/name)!=h:raise ValueError('CPU output changed after measurement')
        if cpu['case_id']!=case.id:raise ValueError('A CPU point has different input')
        cpu_idx=np.fromfile(path.parent/cpu['indices_file'],dtype='<i4').reshape(case.B,case.S,1,case.K)
        exact=None;diagnostic=None
        if ca_idx is not None:
            exact=all(np.array_equal(np.sort(x[x>=0]),np.sort(y[y>=0])) for x,y in zip(cpu_idx.reshape(-1,case.K),ca_idx.reshape(-1,case.K)))
            if cpu.get('scores_file'):
                scores=np.fromfile(path.parent/cpu['scores_file'],dtype='<f4').reshape(case.B,case.S,case.T)
                diagnostic=selection_on_scores(case,ca_idx,scores,'unordered',atol,rtol)
        reasons=[]
        if reference is None:reasons.append('single-LI latency unavailable; recollect/parse timing')
        if not ca.get('timing_scope_confirmed'):reasons.append('CA tick scope not confirmed')
        if ca.get('process_returncode')!=0:reasons.append('successful CA process exit not observed')
        if not ca.get('validation',{} ) or not ca['validation'].get('pass_'):reasons.append('CA structural check unavailable/failed')
        if cpu['validation'].get('status')!='PASS' or cpu['validation'].get('mode')!='golden':reasons.append('CPU independent golden not passed')
        if not(exact or diagnostic and diagnostic['pass_']):reasons.append('cross-device numerical check unavailable/failed')
        order=ca.get('output_order_contract','unverified')
        if order not in ('sorted','unordered') or cpu['parameters']['order']!=order:reasons.append('output ordering contract unresolved/mismatched')
        elapsed=cpu['native_median_us'];budget=reference-elapsed if reference is not None else None
        row=dict(physical_cores=point['physical_cores'],threads=point['threads'],cpu_native_us=elapsed,cpu_host_us=cpu['host_median_us'],
            ca_reference_us=reference,ca_reference_kind=reference_kind,cpu_over_ca_ratio=elapsed/reference if reference and reference>0 else None,
            communication_budget_us=budget,model_compute_comparison_ready=not reasons,limitations='; '.join(reasons),
            exact_index_set_match=exact,threshold_diagnostic=diagnostic,cpu_result=str(path))
        rows.append(row)
        for extra in transport_us:
            sensitivity.append(dict(physical_cores=point['physical_cores'],assumed_total_transport_sync_us=float(extra),cpu_plus_assumption_us=elapsed+extra,ca_model_reference_us=reference,
                modeled_advantage_us=reference-elapsed-extra if reference is not None else None,confirmed_model_comparison=not reasons))
    result=dict(schema_version=1,case_id=case.id,case=case.shape_info(),ca_timing_scope_confirmed=ca.get('timing_scope_confirmed'),rows=rows,transport_sensitivity=sensitivity,
        logical_indexer_key_bytes=int(case.ak.sum())*case.D*2,physical_indexer_key_bytes=case.k.nbytes,
        interpretation='CPU is measured; A5 is CA-model estimated. Communication budget and sensitivity omit actual transfers, runtime launch and downstream attention. No real A5 or full-model speedup is asserted.')
    save_json(out/'comparison.json',result)
    flat=[{k:v for k,v in row.items() if k!='threshold_diagnostic'} for row in rows]
    if flat:
        with (out/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
    lines=['# LI CPU / A5 CA 对比报告','',f'Case ID: `{case.id}`','',result['interpretation'],'',
           '|物理核|线程|CPU原生 μs|CPU封装 μs|CA参考 μs|CPU/CA|传输同步预算 μs|口径就绪|',
           '|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(v):return '未确认' if v is None else f'{v:.3f}'
    for r in rows:lines.append(f"|{r['physical_cores']}|{r['threads']}|{fmt(r['cpu_native_us'])}|{fmt(r['cpu_host_us'])}|{fmt(r['ca_reference_us'])}|{fmt(r['cpu_over_ca_ratio'])}|{fmt(r['communication_budget_us'])}|{'是（模型比较）' if r['model_compute_comparison_ready'] else '暂定'}|")
    lines+=['','## 未决条件','']
    for r in rows:
        if r['limitations']:lines.append(f"- {r['physical_cores']}核：{r['limitations']}")
    lines+=['','## 解释','',
            f'- CA参考类型：{reference_kind}。single_li_model_lifecycle_proxy 表示单次LI使用模型生命周期总时间作近似，尚未扣除模型内额外开销。',
            '- CPU/CA < 1仅表示CPU耗时小于对应CA模型参考值；口径未确认时只是暂定数值。',
            '- 传输同步预算 = CA参考时间 - CPU原生时间。为正表示在该模型下留有预算，为负表示纯计算已较慢。',
            '- JSON中的transport_sensitivity是明确假设的开销扫描，不是传输实测。',
            '- 缓存初始状态、CA校准误差、排序要求、核资源成本和下游Attention仍需纳入部署决策。']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--cpu-scan',required=True);p.add_argument('--ca-result',required=True);p.add_argument('--out',required=True)
    p.add_argument('--transport-us',default='0,5,10,20');p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-4)
    a=p.parse_args();extras=list(map(float,a.transport_us.split(',')))
    if any(x<0 or not np.isfinite(x) for x in extras):p.error('Transport assumptions must be finite nonnegative values')
    r=build_report(a.case,a.cpu_scan,a.ca_result,a.out,extras,a.atol,a.rtol);print(f"Wrote {a.out}/comparison.md; {len(r['rows'])} CPU points")
if __name__=='__main__':main()
