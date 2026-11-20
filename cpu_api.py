"""NumPy/ctypes interface: buffers and native workspace reused across calls."""
import ctypes as C
import functools,threading
from pathlib import Path
import numpy as np
from caseio import Case
ERRORS={-1:'invalid arguments/lengths/PA blocks',-2:'HISTSEG requires VL=256 on every worker',-4:'HISTSEG unavailable in scalar-only build',-5:'OpenMP team size differs from requested',-6:'nonfinite computed score',-7:'BLAS backend not compiled',-8:'accelerated BF16 backend unavailable (scalar build or H%8/D%2/VL=256 not satisfied)',-9:'SME score backend not compiled; rebuild with scripts/build_cpu.sh --sme'}

def exclusive(method):
    @functools.wraps(method)
    def wrapper(self,*args,**kwargs):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent use/close of one CPUOp workspace is not allowed')
        try:return method(self,*args,**kwargs)
        finally:self._lock.release()
    return wrapper

class CPUOp:
    def __init__(self,case,threads=1,key_tile=128,query_tile=1,hist='histseg',order='sorted',groups=0,score_backend='simd',library=None):
        self.handle=None;self._lock=threading.Lock();self.case=case if isinstance(case,Case) else Case(case)
        c=self.case
        if min(threads,key_tile,query_tile)<1 or not 0<=groups<=threads:raise ValueError('Invalid parallel parameters')
        if hist not in ('histseg','scalar') or order not in ('sorted','unordered') or score_backend not in ('simd','blas','svebf16','sme'):raise ValueError('Unknown backend/order')
        self.lib=C.CDLL(str(Path(library).resolve() if library else Path(__file__).parent/'build/libli_cpu.so'))
        l=self.lib
        if l.li_abi_version()!=3:raise RuntimeError('Wrong native ABI: rebuild this project')
        if score_backend=='sme' and not getattr(l,'li_has_sme',lambda:0)():
            self.lib=None;raise RuntimeError(ERRORS[-9])
        l.li_create.argtypes=[C.c_int]*9;l.li_create.restype=C.c_void_p
        l.li_destroy.argtypes=[C.c_void_p];l.li_destroy.restype=None;l.li_set_score_backend.argtypes=[C.c_void_p,C.c_int]
        l.li_execute.argtypes=[C.c_void_p]*7+[C.c_int]*8+[C.c_void_p]*3;l.li_execute.restype=C.c_int
        l.li_worker_info.argtypes=[C.c_void_p]*3+[C.c_int,C.c_void_p];l.li_worker_info.restype=C.c_int
        if hist=='histseg' and not l.li_has_sve2():raise RuntimeError('Use --hist scalar for scalar-only build')
        self.out=np.empty((c.B,c.S,1,c.K),np.int32);self.score=np.empty((c.B,c.S,c.T),np.float32);self.timing=np.zeros(4,np.float64)
        self.options=(int(c.k.size//c.D),c.k.shape[1],c.bt.shape[1],1,c.meta['sparse_mode'],int(order=='sorted'),int(hist=='histseg'),groups)
        self.threads=threads
        self.handle=l.li_create(c.B,c.S,c.T,c.H,c.D,c.K,threads,key_tile,query_tile)
        if not self.handle:raise MemoryError('Native workspace allocation failed')
        self.blas_thread_setter=False
        if score_backend=='blas':
            for name in ('openblas_set_num_threads','MKL_Set_Num_Threads','mkl_set_num_threads'):
                setter=getattr(l,name,None)
                if setter is not None:
                    setter.argtypes=[C.c_int];setter(1);self.blas_thread_setter=True
        rc=l.li_set_score_backend(self.handle,0 if score_backend=='simd' else 1 if score_backend=='blas' else 2)
        if rc:self.close();raise RuntimeError(ERRORS.get(rc,str(rc)))
        self.input_ptrs=(c.q.ctypes.data,c.k.ctypes.data,c.w.ctypes.data,c.bt.ctypes.data,c.aq.ctypes.data,c.ak.ctypes.data)
    @exclusive
    def run(self,pointers=None):
        if not self.handle:raise RuntimeError('Closed workspace')
        rc=self.lib.li_execute(self.handle,*(pointers or self.input_ptrs),*self.options,self.out.ctypes.data,self.score.ctypes.data,self.timing.ctypes.data)
        if rc:raise RuntimeError(f'Native error {rc}: {ERRORS.get(rc,"unknown")}')
        return self.out,self.score,self.timing
    @exclusive
    def info(self):
        if not self.handle:raise RuntimeError('Closed workspace')
        cpu=np.empty(self.threads,np.int32);vl=np.empty_like(cpu);g=np.zeros(1,np.int32)
        n=self.lib.li_worker_info(self.handle,cpu.ctypes.data,vl.ctypes.data,self.threads,g.ctypes.data)
        sme=bool(getattr(self.lib,'li_has_sme',lambda:0)())
        return dict(actual_threads=n,worker_cpus=cpu[:n].tolist(),worker_vl_bits=vl[:n].tolist(),topk_groups=int(g[0]),sve2_compiled=bool(self.lib.li_has_sve2()),sme_compiled=sme,score_impl='sme' if sme else 'sve2',blas_compiled=bool(self.lib.li_has_blas()),blas_thread_setter_used=self.blas_thread_setter)
    @exclusive
    def close(self):
        if self.handle:self.lib.li_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        if getattr(self,'handle',None):self.close()
