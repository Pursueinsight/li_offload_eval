"""Generate one immutable shared case. Model-shaped random data, not a model trace."""
import argparse,json
from pathlib import Path
import numpy as np
from caseio import INPUT_NAMES,to_bf16,to_float,sha256,fingerprint,save_json,Case

PRESETS={
 'smoke':dict(B=2,S=3,T=257,H=8,K=32),
 **{f'decode_{n}k':dict(B=1,S=1,T=n*1024,H=64,K=2048) for n in (16,32,64,128)},
 'batch8_64k':dict(B=8,S=1,T=65536,H=64,K=2048),
 'batch32_64k':dict(B=32,S=1,T=65536,H=64,K=2048),
 'prefill128_16k':dict(B=1,S=128,T=16384,H=64,K=2048),
 'regression_a2':dict(B=18,S=3,T=3072,H=16,K=2048,lengths=[3016]*17+[3072],physical_blocks=469)}

def generate(out,preset='decode_64k',seed=0,distribution='normal',block_size=128,**overrides):
    cfg=PRESETS[preset].copy();cfg.update({k:v for k,v in overrides.items() if v is not None})
    B,S,T,H,K=(int(cfg[k]) for k in ('B','S','T','H','K'));D=128
    if min(B,S,T,K)<1 or H not in (8,16,24,32,64) or not(K<=2048 or K<=8192 and K%1024==0):raise ValueError('Invalid shape/K for original A5 LI')
    if block_size<16 or block_size>1024 or block_size%16:raise ValueError('PA block_size must be 16..1024, multiple of16')
    if overrides.get('B') is not None or overrides.get('T') is not None:cfg.pop('lengths',None)
    lengths=cfg.get('lengths',[T]*B)
    if len(lengths)!=B or min(lengths)<0 or max(lengths)>T:raise ValueError('Invalid lengths')
    needed=sum((n+block_size-1)//block_size for n in lengths)
    physical=int(cfg.get('physical_blocks',needed))
    if physical<needed:raise ValueError('physical_blocks too small')
    root=Path(out)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a new empty case directory; existing case/results are preserved')
    root.mkdir(parents=True,exist_ok=True)
    m=dict(query_shape=[B,S,H,D],key_shape=[max(1,physical),block_size,1,D],weights_shape=[B,S,H],aq_shape=[B],ak_shape=[B],bt_shape=[B,max(1,(T+block_size-1)//block_size)],layout_query='BSND',layout_key='PA_BSND',sparse_count=K,sparse_mode=3,qk_dtype='bf16',weight_dtype='bf16',return_values=False)
    rng=np.random.default_rng(seed)
    for name,shape in [('query',m['query_shape']),('key',m['key_shape']),('weights',m['weights_shape'])]:
        # Stream generation in bounded chunks; large batch cases avoid multi-GB FP32 temporaries.
        remaining=int(np.prod(shape,dtype=np.int64))
        with (root/f'li_{name}.bf16').open('wb') as raw_file,(root/f'li_a2_{name}.f32').open('wb') as fp_file:
            while remaining:
                n=min(remaining,262144)
                values=rng.standard_normal(n,dtype=np.float32) if distribution=='normal' else (rng.random(n,dtype=np.float32)*2-1)*(130 if name=='weights' else 1)
                bits=to_bf16(values);bits.tofile(raw_file);to_float(bits).tofile(fp_file);remaining-=n
    np.full(B,S,dtype='<i4').tofile(root/'li_a2_aq.i32');np.array(lengths,dtype='<i4').tofile(root/'li_a2_ak.i32')
    table=np.full(m['bt_shape'],-1,dtype='<i4');ids=rng.permutation(max(1,physical));pos=0
    for b,n in enumerate(lengths):
        count=(n+block_size-1)//block_size;table[b,:count]=ids[pos:pos+count];pos+=count
    table.tofile(root/'li_a2_bt.i32');save_json(root/'li_a2_meta.json',m)
    hashes={name:sha256(root/name) for name in INPUT_NAMES}
    manifest=dict(schema_version=1,case_id=fingerprint(m,hashes),preset=preset,seed=seed,distribution=distribution,input_origin='synthetic_model_shape',meta=m,sha256=hashes,
        model_reference='DeepSeek-V3.2 Indexer: H=64,D=128,K=2048. BF16 path here differs from official FP8/scales.',
        output_contract=dict(indices_dtype='int32',padding=-1,selection='weighted_relu_qk_topk',order='target_repository_order_unverified'))
    save_json(root/'manifest.json',manifest)
    return Case(root)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preset',choices=PRESETS,default='decode_64k');p.add_argument('--out',required=True)
    for n in ('batch','queries','context','heads','topk'):p.add_argument('--'+n,type=int)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--block-size',type=int,default=128);p.add_argument('--distribution',choices=['normal','legacy_uniform'],default='normal')
    a=p.parse_args();c=generate(a.out,a.preset,a.seed,a.distribution,a.block_size,B=a.batch,S=a.queries,T=a.context,H=a.heads,K=a.topk)
    print(json.dumps(dict(case_id=c.id,shape=c.shape_info(),resident_bytes=c.resident_bytes),indent=2))
if __name__=='__main__':main()
