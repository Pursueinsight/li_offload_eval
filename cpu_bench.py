"""Resident BF16 CPU LI. Main metric = native entry-to-completion incl OpenMP and necessary conversion."""
import argparse,json,os,time,platform
from pathlib import Path
import numpy as np
from caseio import Case,save_json,sha256
from cpu_api import CPUOp
from validation import validate_cpu

ROOT=Path(__file__).parent

def run_benchmark(case_path,out,threads=1,tile=128,query_tile=1,groups=0,hist='histseg',order='unordered',score_backend='simd',warmup=10,iters=100,input_copies=1,allocation='reuse',verify='golden',atol=1e-3,rtol=1e-4,require_pinning=False,save_scores=False,library=None):
    if iters<1 or warmup<0 or input_copies<1:raise ValueError('Invalid sampling parameters')
    if allocation not in ('reuse','per-call') or verify not in ('none','selection','golden'):raise ValueError('Invalid allocation/verification mode')
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all():raise ValueError('Tolerances must be finite and nonnegative')
    case=Case(case_path);dest=Path(out).resolve()
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Choose a new CPU result directory')
    dest.mkdir(parents=True,exist_ok=True)
    arrays=[case.q,case.k,case.w,case.bt,case.aq,case.ak]
    pool=[arrays]+[[a.copy() for a in arrays] for _ in range(input_copies-1)]
    pointers=[tuple(a.ctypes.data for a in chunk) for chunk in pool]
    options=dict(threads=threads,key_tile=tile,query_tile=query_tile,groups=groups,hist=hist,order=order,score_backend=score_backend,library=library)
    reuse=CPUOp(case,**options) if allocation=='reuse' else None

    def call(i,capture_info=False):
        op=reuse or CPUOp(case,**options)
        try:
            idx,score,timing=op.run(pointers[i%input_copies]);info=op.info() if reuse is None and capture_info else None
            return idx,score,timing.copy(),info
        finally:
            if reuse is None:op.close()

    try:
        for i in range(warmup):call(i)
        result=call(0,capture_info=True);info=reuse.info() if reuse else result[3]
        if require_pinning and (info['actual_threads']!=threads or len(set(info['worker_cpus']))!=threads or os.environ.get('OMP_PROC_BIND','').lower() not in ('true','close','spread')):
            raise RuntimeError('Thread pinning check failed; use scan_cpu.py')
        samples=[];stages=[]
        for i in range(iters):
            result=None;t=time.perf_counter_ns();result=call(i);samples.append((time.perf_counter_ns()-t)/1e6);stages.append(result[2].copy())
        idx,score,_,_=result

        # Native samples measure identical work whether caller chooses reuse or per-call allocation.
        stage=np.array(stages);native=stage[:,3];wall=np.array(samples)
        validation=validate_cpu(case,idx,score,verify,order,atol,rtol)
        idx.astype('<i4',copy=False).tofile(dest/'indices.i32')
        if save_scores:score.astype('<f4',copy=False).tofile(dest/'scores.f32')
        lib=Path(library).resolve() if library else ROOT/'build/libli_cpu.so'
        result=dict(schema_version=1,kind='CPU_MEASUREMENT',case_id=case.id,case_path=str(case.path),config=case.shape_info(),
            parameters=dict(threads=threads,tile=tile,query_tile=query_tile,topk_groups_requested=groups,hist=hist,order=order,score_backend=score_backend,allocation=allocation,warmup=warmup,iters=iters,input_copies=input_copies),
            timing_scope='resident_algorithm_native_entry_to_completion_including_conversion_and_thread_team',
            native_median_us=float(np.median(native)*1000),native_p90_us=float(np.percentile(native,90)*1000),
            host_median_us=float(np.median(wall)*1000),host_p90_us=float(np.percentile(wall,90)*1000),
            stage_median_us=dict(zip(('prepare','score','topk','total'),(np.median(stage,axis=0)*1000).tolist())),
            samples_native_us=(native*1000).tolist(),samples_host_us=(wall*1000).tolist(),
            query_tokens_per_second=float(case.aq.sum())/(float(np.median(native))/1000),validation=validation,
            resident_input_bytes_per_copy=case.resident_bytes,input_pool_bytes=case.resident_bytes*input_copies,
            cache_condition='repeated_single_buffer' if input_copies==1 else 'rotating_buffers_residency_not_guaranteed',
            main_thread_affinity=sorted(os.sched_getaffinity(0)),omp_environment={k:os.environ.get(k) for k in ('OMP_PLACES','OMP_PROC_BIND','OMP_WAIT_POLICY','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')},
            platform=platform.platform(),numpy=np.__version__,native_library_sha256=sha256(lib),
            output_sha256={name:sha256(dest/name) for name in (['indices.i32','scores.f32'] if save_scores else ['indices.i32'])},
            indices_file='indices.i32',indices_shape=list(idx.shape),scores_shape=list(score.shape),scores_file='scores.f32' if save_scores else None,**info)
        save_json(dest/'result.json',result)
        if validation['status']=='FAIL':raise RuntimeError(f'Correctness failed; see {dest}/result.json')
        return result
    finally:
        if reuse:reuse.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--out',required=True)
    for n,v in [('threads',1),('tile',128),('query-tile',1),('groups',0),('warmup',10),('iters',100),('input-copies',1)]:p.add_argument('--'+n,type=int,default=v)
    p.add_argument('--hist',choices=['histseg','scalar'],default='histseg');p.add_argument('--order',choices=['sorted','unordered'],default='unordered')
    p.add_argument('--score-backend',choices=['simd','blas','svebf16','sme'],default='simd');p.add_argument('--allocation',choices=['reuse','per-call'],default='reuse')
    p.add_argument('--verify',choices=['none','selection','golden'],default='golden');p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-4)
    p.add_argument('--require-pinning',action='store_true');p.add_argument('--save-scores',action='store_true');p.add_argument('--library')
    a=p.parse_args();r=run_benchmark(a.case,a.out,threads=a.threads,tile=a.tile,query_tile=a.query_tile,groups=a.groups,hist=a.hist,order=a.order,score_backend=a.score_backend,warmup=a.warmup,iters=a.iters,input_copies=a.input_copies,allocation=a.allocation,verify=a.verify,atol=a.atol,rtol=a.rtol,require_pinning=a.require_pinning,save_scores=a.save_scores,library=a.library)
    print(json.dumps({k:r[k] for k in ('case_id','native_median_us','host_median_us','stage_median_us','actual_threads','worker_cpus','validation')},indent=2))

if __name__=='__main__':
    main()