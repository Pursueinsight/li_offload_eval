"""Run one CA logical LI in an isolated input copy; collect external ticks and output.
Automatically estimate one LI from model-total ticks / configured MHz.
No host wall time is interpreted as A5 latency. Exact scope confirmation remains separate.
"""
import argparse,json,os,re,shutil,signal,subprocess,time
from pathlib import Path
import numpy as np
from caseio import Case,INPUT_NAMES,sha256,save_json
from validation import structural

ROOT=Path(__file__).parent

def tick_records(text):
    records=[]
    for n,line in enumerate(text.splitlines(),1):
        m=re.search(r'\bTotal\s+tick\s*:\s*([0-9][0-9,]*)',line,re.I)
        if m:records.append(dict(line_number=n,ticks=int(m.group(1).replace(',','')),raw=line))
    return records

def auto_li_estimate(text,records,chosen,calls,frequency_mhz,returncode,validation):
    """Lifecycle-total proxy for exactly one successfully completed LI; never divide by calls."""
    reasons=[]
    if len(records)!=1:reasons.append('expected exactly one model Total tick record')
    if chosen is None or chosen<0 or chosen>=len(records) or records[chosen]['ticks']<=0:
        reasons.append('no positive selected tick')
    if calls!=1:reasons.append('expected exactly one completed logical LI call')
    requested=re.findall(r'EXECUTION requested_li_calls=(\d+) warmup=(\d+) iters=(\d+)',text)
    if requested and requested!=[('1','0','1')]:
        reasons.append('requested calls must be warmup=0, iters=1')
    # Imported logs have no observed exit code: require explicit completion markers.
    logged_success=('Model stopped successfully.' in text and
                    'DONE status=PASS_STRUCTURAL_ONLY' in text)
    if returncode!=0 and not(returncode is None and logged_success):
        reasons.append('successful process/log completion not established')
    if not validation or not validation.get('pass_'):reasons.append('output structural check failed/unavailable')
    if reasons:return None,reasons
    return records[chosen]['ticks']/frequency_mhz,[]

def collect(case,run_dir,log,indices,returncode,exe=None,model_config=None,evidence=None,freq_mhz=1650.,tick_index=None):
    root=Path(run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if not 0<freq_mhz<float('inf'):raise ValueError('Frequency must be positive and finite')
    text=Path(log).read_text(errors='replace');records=tick_records(text)
    calls=re.findall(r'EXECUTION completed_li_calls=(\d+)',text);calls=sum(map(int,calls)) if calls else None
    chosen=0 if len(records)==1 and tick_index is None else tick_index
    if chosen is not None and not 0<=chosen<len(records):raise ValueError('Tick index outside record list')
    candidate=records[chosen]['ticks']/freq_mhz if chosen is not None and records[chosen]['ticks']>0 else None
    ev=json.loads(Path(evidence).read_text()) if evidence else None
    timing_confirmed=bool(ev and ev.get('confirmed') is True and ev.get('scope')=='single_complete_li_critical_path' and ev.get('evidence') and ev.get('clock_domain') not in (None,'','TO_BE_VERIFIED') and ev.get('frequency_mhz')==freq_mhz and calls==1 and chosen is not None and candidate is not None)
    evidence_error='Timing evidence inconsistent with log call count, scope, frequency or selected tick' if ev and ev.get('confirmed') and not timing_confirmed else None
    output=Path(indices);validation=None
    if output.exists():
        values=np.fromfile(output,dtype='<i4')
        if values.size==case.B*case.S*case.K and output.stat().st_size==case.B*case.S*case.K*4:
            values.tofile(root/'indices.i32');validation=structural(case,values)
        else:validation=dict(pass_=False,error='Output byte count disagrees with case')
    estimate,estimate_reasons=auto_li_estimate(text,records,chosen,calls,freq_mhz,returncode,validation)
    if returncode not in (0,None) or not validation or not validation.get('pass_'):
        timing_confirmed=False
    if ev:save_json(root/'timing_evidence.json',ev)
    li_latency=candidate if timing_confirmed else estimate
    result=dict(schema_version=2,kind='A5_CA_MODEL',case_id=case.id,config=case.shape_info(),
        process_returncode=returncode,completed_logical_li_calls=calls,frequency_mhz=freq_mhz,
        tick_records=records,selected_tick_index=chosen,candidate_duration_us=candidate,
        model_total_latency_us=candidate if len(records)==1 else None,
        li_latency_us=li_latency,estimated_li_latency_us=estimate,
        li_latency_kind=('confirmed_li_critical_path' if timing_confirmed else
                         'single_li_model_lifecycle_proxy' if estimate is not None else 'unavailable'),
        li_latency_is_approximate=not timing_confirmed if li_latency is not None else None,
        estimate_unavailable_reasons=estimate_reasons,
        tick_conversion='duration_us = Total tick / frequency_mhz',
        frequency_source='configured --freq-mhz; user-provided tick-to-clock mapping, not inferred from periods',
        estimate_note='A single-LI model lifecycle total is used as an LI latency estimate. It can include modeled setup/teardown, scheduling or other activity; overhead is not subtracted. Not real A5 hardware timing.',
        model_li_latency_us=candidate if timing_confirmed else None,timing_scope_confirmed=timing_confirmed,
        timing_scope=('resident_complete_li_model_critical_path' if timing_confirmed else
                      'model_lifecycle_single_li_proxy' if estimate is not None else 'unverified_external_tick_scope'),
        validation=validation,timing_evidence_error=evidence_error,output_sha256={'indices.i32':sha256(root/'indices.i32')} if validation and 'error' not in validation else {},indices_file='indices.i32' if validation and 'error' not in validation else None,
        log_file='ca.log',model_config_sha256=sha256(model_config) if model_config else None,
        runner_sha256=sha256(exe) if exe else None,
        kernel_log_hints=[line for line in text.splitlines() if re.search(r'kernel.?name|block.?dim|tiling.?key|core num|config file',line,re.I)],
        real_hardware_measurement=False,source_revision=ev.get('source_revision') if ev else None,
        output_order_contract=ev.get('output_order','unverified') if ev else 'unverified')
    save_json(root/'result.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--out',required=True)
    p.add_argument('--exe',default=str(ROOT/'build/li_aclnn_bench'));p.add_argument('--env-script',help='Source an ENVIRONMENT-ONLY shell fragment; must not execute the old benchmark')
    p.add_argument('--model-config');p.add_argument('--timing-evidence');p.add_argument('--freq-mhz',type=float,default=1650)
    p.add_argument('--tick-index',type=int);p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--import-log',help='Parse an existing complete log instead of rerunning CA');p.add_argument('--import-indices')
    a=p.parse_args()
    if not 0<=a.timeout<float('inf') or not 0<a.freq_mhz<float('inf'):p.error('Timeout must be nonnegative and finite; frequency must be positive and finite')
    if a.timing_evidence:
        ev=json.loads(Path(a.timing_evidence).read_text())
        if ev.get('confirmed') and (ev.get('scope')!='single_complete_li_critical_path' or ev.get('clock_domain') in (None,'','TO_BE_VERIFIED') or ev.get('frequency_mhz')!=a.freq_mhz or not ev.get('evidence')):
            p.error('Confirmed timing evidence missing valid scope/clock/frequency/description')
    case=Case(a.case);dest=Path(a.out).resolve()
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Use a new CA result directory')
    dest.mkdir(parents=True,exist_ok=True)
    if a.model_config:shutil.copyfile(a.model_config,dest/'model_config.ini')
    if a.import_log:
        if not a.import_indices:p.error('--import-log requires --import-indices')
        shutil.copyfile(a.import_log,dest/'ca.log')
        result=collect(case,dest,dest/'ca.log',a.import_indices,None,model_config=a.model_config,evidence=a.timing_evidence,freq_mhz=a.freq_mhz,tick_index=a.tick_index)
        result['imported']=True;result['process_returncode']=None;result['import_note']='Original process exit status not independently observed; inspect source log.'
        save_json(dest/'result.json',result)
    else:
        exe=Path(a.exe).resolve()
        if not exe.is_file():raise FileNotFoundError(f'Compile CA runner first: {exe}')
        inputs=dest/'inputs';inputs.mkdir()
        for name in [*INPUT_NAMES,'manifest.json']:shutil.copyfile(case.path/name,inputs/name)
        cmd=[str(exe),'--dir',str(inputs),'--iters','1','--warmup','0']
        if a.env_script:
            fragment=Path(a.env_script).resolve();body=fragment.read_text()
            if re.search(r'^\s*(?:(?:exec|nohup)\s+)?(?:\S*/)?li_aclnn_bench\S*(?:\s|$)',body,re.M):
                raise ValueError('Environment fragment appears to launch an old runner; copy only source/export environment setup')
            cmd=['bash','-c','set -euo pipefail; source "$1"; shift; exec "$@"','li-ca',str(fragment),*cmd]
            shutil.copyfile(fragment,dest/'environment.sh')
        save_json(dest/'launch.json',dict(command=cmd,timeout_seconds=a.timeout,case_id=case.id,model_config_path=str(Path(a.model_config).resolve()) if a.model_config else None))
        with (dest/'ca.log').open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:
                code=proc.wait(timeout=None if a.timeout == 0 else a.timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=proc.returncode
        # Waited for complete process, not the first tick/DONE line. Verify inputs weren't changed.
        integrity_error=None
        try:Case(inputs)
        except Exception as e:integrity_error=str(e)
        result=collect(case,dest,dest/'ca.log',inputs/'li_out.i32',code,exe,a.model_config,a.timing_evidence,a.freq_mhz,a.tick_index)
        result['timed_out']=timed_out;result['input_integrity_error']=integrity_error
        if integrity_error or timed_out:
            result['timing_scope_confirmed']=False;result['model_li_latency_us']=None
            result['li_latency_us']=None;result['estimated_li_latency_us']=None
            result['li_latency_kind']='unavailable';result['li_latency_is_approximate']=None
            result['estimate_unavailable_reasons'].append('input integrity failure or simulator timeout')
        save_json(dest/'result.json',result)
    value=result['li_latency_us']
    print('CA_LI_RESULT latency_us='+('unavailable' if value is None else f'{value:.6f}')+
          f" kind={result['li_latency_kind']} frequency_mhz={a.freq_mhz:g}")
    print(json.dumps(result,indent=2))
    if not result['validation'] or not result['validation'].get('pass_') or result.get('timed_out') or result.get('input_integrity_error') or result.get('process_returncode') not in (None,0):raise SystemExit(1)
if __name__=='__main__':main()
