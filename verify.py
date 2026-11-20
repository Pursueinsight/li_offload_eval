"""Offline output/CPU-score verification. Does not collect performance."""
import argparse,json
from pathlib import Path
import numpy as np
from caseio import Case,save_json
from validation import structural,golden_scores,selection_on_scores,validate_cpu

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--indices',required=True)
    p.add_argument('--scores');p.add_argument('--golden',action='store_true');p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-4);p.add_argument('--out',required=True)
    a=p.parse_args();c=Case(a.case);idx=np.fromfile(a.indices,dtype='<i4').reshape(c.B,c.S,1,c.K)
    result=dict(case_id=c.id,structural=structural(c,idx))
    if a.scores:
        scores=np.fromfile(a.scores,dtype='<f4').reshape(c.B,c.S,c.T)
        result['cpu']=validate_cpu(c,idx,scores,'golden' if a.golden else 'selection','unordered',a.atol,a.rtol)
    elif a.golden:result['golden_selection']=selection_on_scores(c,idx,golden_scores(c),'unordered',a.atol,a.rtol)
    result['status']='PASS' if result['structural']['pass_'] and result.get('cpu',{'status':'PASS'})['status']=='PASS' and result.get('golden_selection',{'pass_':True})['pass_'] else 'FAIL'
    result['note']='FP32 oracle/tolerances are diagnostic; exact target repository acceptance may differ.'
    save_json(a.out,result);print(json.dumps(result,indent=2))
    if result['status']=='FAIL':raise SystemExit(1)
if __name__=='__main__':main()
