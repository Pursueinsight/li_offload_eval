"""Independent NumPy FP32 oracle and ordering-independent diagnostics, outside timed regions."""
import numpy as np
from caseio import to_float

def structural(case,indices):
    idx=np.asarray(indices).reshape(case.B,case.S,case.K)
    count_bad=range_bad=duplicates=0;valid=pad=0
    for b in range(case.B):
        for i in range(case.S):
            row=idx[b,i];chosen=row[row!=-1];v=case.visible(b,i)
            valid+=chosen.size;pad+=int(np.sum(row==-1))
            count_bad+=int(chosen.size!=min(case.K,v));range_bad+=int(np.any((chosen<0)|(chosen>=v)))
            duplicates+=int(np.unique(chosen).size!=chosen.size)
    return dict(pass_=not(count_bad or range_bad or duplicates),valid=int(valid),padding=int(pad),bad_count_rows=count_bad,bad_range_rows=range_bad,duplicate_rows=duplicates)

def selection_on_scores(case,indices,scores,order='unordered',atol=0.,rtol=0.):
    idx=np.asarray(indices).reshape(case.B,case.S,case.K);score=np.asarray(scores).reshape(case.B,case.S,case.T)
    exact=violations=order_bad=0
    for b in range(case.B):
        for i in range(case.S):
            v=case.visible(b,i);k=min(case.K,v)
            if not k:continue
            chosen=idx[b,i][idx[b,i]>=0].astype(np.int64)
            if chosen.size!=k or np.any(chosen>=v) or np.unique(chosen).size!=k:
                violations+=1;continue
            s=score[b,i,:v]
            if not np.isfinite(s).all():violations+=1;continue
            expected=np.lexsort((np.arange(v),-s))[:k]
            exact+=int(not np.array_equal(np.sort(chosen),np.sort(expected)))
            threshold=float(s[expected[-1]]);tol=atol+rtol*abs(threshold)
            mask=np.ones(v,bool);mask[chosen]=False
            violations+=int(float(np.min(s[chosen]))<threshold-tol or (k<v and float(np.max(s[mask]))>threshold+tol))
            if order=='sorted':order_bad+=int(not np.array_equal(chosen,expected))
    return dict(exact_set_mismatch_rows=exact,threshold_violation_rows=violations,stable_order_mismatch_rows=order_bad,
                pass_=not(violations or (order=='sorted' and order_bad)),atol=atol,rtol=rtol)

def golden_scores(case):
    q=to_float(case.q);w=to_float(case.w);result=np.full((case.B,case.S,case.T),-np.inf,np.float32)
    for b in range(case.B):
        k=case.logical_key(b)
        for i in range(int(case.aq[b])):
            v=case.visible(b,i)
            if v:
                dots=q[b,i]@k[:v].T
                result[b,i,:v]=np.sum(w[b,i,:,None]*np.maximum(dots,0.),axis=0,dtype=np.float32)
    return result

def validate_cpu(case,indices,scores,mode='golden',order='sorted',atol=1e-3,rtol=1e-4):
    if mode=='none':return dict(status='NOT_RUN')
    structure=structural(case,indices);selection=selection_on_scores(case,indices,scores,order)
    report=dict(status='PASS' if structure['pass_'] and selection['pass_'] else 'FAIL',structural=structure,selection=selection,mode=mode)
    if mode=='golden':
        ref=golden_scores(case);mask=np.isfinite(ref)
        same_mask=np.array_equal(np.isfinite(scores),mask)
        close=same_mask and bool(np.allclose(np.asarray(scores)[mask],ref[mask],atol=atol,rtol=rtol))
        report['golden']=dict(score_allclose=close,max_abs_error=float(np.max(np.abs(np.asarray(scores)[mask]-ref[mask]))) if np.any(mask) else 0.,atol=atol,rtol=rtol)
        if not close:report['status']='FAIL'
    return report
