"""Canonical BF16 input format shared by native CPU and the CA ACLNN runner."""
import hashlib,json
from pathlib import Path
import numpy as np

INPUT_NAMES=['li_a2_meta.json','li_query.bf16','li_key.bf16','li_weights.bf16','li_a2_aq.i32','li_a2_ak.i32','li_a2_bt.i32',
             'li_a2_query.f32','li_a2_key.f32','li_a2_weights.f32']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fingerprint(meta,hashes):
    return hashlib.sha256(json.dumps({'meta':meta,'payload_sha256':hashes},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def to_bf16(values):
    u=np.ascontiguousarray(values,dtype='<f4').view('<u4')
    return ((u+np.uint32(0x7fff)+((u>>16)&np.uint32(1)))>>np.uint32(16)).astype('<u2')

def to_float(bits):
    return (np.asarray(bits,dtype='<u2').astype('<u4')<<16).view('<f4')

class Case:
    def __init__(self,path,verify_hashes=True):
        self.path=Path(path).resolve()
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest.get('schema_version')!=1:raise ValueError('Unsupported case schema')
        self.meta=json.loads((self.path/'li_a2_meta.json').read_text())
        if self.meta!=self.manifest['meta']:raise ValueError('Metadata changed after case generation')
        if set(self.manifest.get('sha256',{}))!=set(INPUT_NAMES):raise ValueError('Incomplete input hash manifest')
        if verify_hashes:
            for name,h in self.manifest['sha256'].items():
                if name not in INPUT_NAMES or sha256(self.path/name)!=h:raise ValueError(f'Input hash mismatch: {name}')
        if fingerprint(self.meta,self.manifest['sha256'])!=self.manifest['case_id']:raise ValueError('Invalid case_id')
        m=self.meta
        if m['layout_query']!='BSND' or m['layout_key']!='PA_BSND' or m['qk_dtype']!='bf16' or m['weight_dtype']!='bf16':
            raise ValueError('Project currently supports BSND/PA_BSND, BF16 Q/K/W')
        def read(name,dt,shape):
            if (self.path/name).stat().st_size!=int(np.prod(shape,dtype=np.int64))*np.dtype(dt).itemsize:raise ValueError(f'File byte size mismatch: {name}')
            a=np.fromfile(self.path/name,dtype=dt)
            if a.size!=int(np.prod(shape,dtype=np.int64)):raise ValueError(f'Shape/file mismatch: {name}')
            return a.reshape(shape)
        self.q=read('li_query.bf16','<u2',m['query_shape'])
        self.k=read('li_key.bf16','<u2',m['key_shape'])
        self.w=read('li_weights.bf16','<u2',m['weights_shape'])
        self.aq=read('li_a2_aq.i32','<i4',m['aq_shape']);self.ak=read('li_a2_ak.i32','<i4',m['ak_shape'])
        self.bt=read('li_a2_bt.i32','<i4',m['bt_shape'])
        B,S,H,D=self.q.shape
        if D!=128 or H not in (8,16,24,32,64) or self.k.shape[2:]!=(1,D) or self.w.shape!=(B,S,H):raise ValueError('Invalid Q/K/W shape')
        if self.aq.shape!=(B,) or self.ak.shape!=(B,) or self.bt.ndim!=2 or self.bt.shape[0]!=B:raise ValueError('Invalid lengths/table')
        if np.any(self.aq<0) or np.any(self.aq>S) or np.any(self.ak<0):raise ValueError('Invalid effective lengths')
        bs=self.k.shape[1]
        if bs<16 or bs>1024 or bs%16:raise ValueError('Invalid PA block_size')
        for b in range(B):
            n=(int(self.ak[b])+bs-1)//bs
            if n>self.bt.shape[1] or np.any(self.bt[b,:n]<0) or np.any(self.bt[b,:n]>=self.k.shape[0]):raise ValueError('Invalid active PA blocks')
        if m['sparse_mode'] not in (0,3) or not (1<=m['sparse_count']<=2048 or 3072<=m['sparse_count']<=8192 and m['sparse_count']%1024==0):raise ValueError('Invalid sparse attributes')
        self.B,self.S,self.H,self.D=B,S,H,D;self.T=max(1,int(self.ak.max()));self.K=int(m['sparse_count'])
        self.id=self.manifest['case_id']
    def visible(self,b,i):
        if i>=self.aq[b]:return 0
        return max(0,int(self.ak[b])-int(self.aq[b])+i+1) if self.meta['sparse_mode']==3 else int(self.ak[b])
    def logical_key(self,b):
        bs=self.k.shape[1];n=int(self.ak[b]);ids=self.bt[b,:(n+bs-1)//bs]
        return to_float(self.k[ids].reshape(-1,self.D)[:n])
    def shape_info(self):
        return dict(B=self.B,S1=self.S,N1=self.H,N2=1,D=self.D,K=self.K,lengths_q=self.aq.tolist(),lengths_k=self.ak.tolist(),dtype='bf16',key_shape=list(self.k.shape),mask=self.meta['sparse_mode'])
    @property
    def resident_bytes(self):return sum(a.nbytes for a in (self.q,self.k,self.w,self.aq,self.ak,self.bt))

def save_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
