"""Import existing li_a2_*.f32/.i32 files into canonical shared BF16 case format."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from caseio import Case,INPUT_NAMES,to_bf16,to_float,sha256,fingerprint,save_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--allow-bf16-rounding',action='store_true');a=p.parse_args()
    source=Path(a.dir);dest=Path(a.out)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Use a new empty case directory')
    dest.mkdir(parents=True,exist_ok=True);m=json.loads((source/'li_a2_meta.json').read_text())
    if m['layout_query']!='BSND' or m['layout_key']!='PA_BSND':raise ValueError('Importer supports BSND/PA_BSND original format')
    for field in ('dtype','qk_dtype','query_dtype','key_dtype','weight_dtype','weights_dtype'):
        if field in m and m[field] not in ('bf16','bfloat16','torch.bfloat16'):raise ValueError('Importer is BF16-only; dtype mismatch in '+field)
    for name in ('query','key','weights'):
        shape=m[name+'_shape'];v=np.fromfile(source/f'li_a2_{name}.f32',dtype='<f4')
        if v.size!=int(np.prod(shape)) or not np.isfinite(v).all():raise ValueError(f'Invalid {name}')
        raw=to_bf16(v);restored=to_float(raw)
        if not np.isfinite(restored).all():raise ValueError('BF16 overflow')
        if not a.allow_bf16_rounding and not np.array_equal(v.view('<u4'),restored.view('<u4')):raise ValueError('Not already BF16-valued; use explicit --allow-bf16-rounding only if appropriate')
        raw.tofile(dest/f'li_{name}.bf16');restored.tofile(dest/f'li_a2_{name}.f32')
    for name in ('aq','ak','bt'):shutil.copyfile(source/f'li_a2_{name}.i32',dest/f'li_a2_{name}.i32')
    m.update(qk_dtype='bf16',weight_dtype='bf16',return_values=False);save_json(dest/'li_a2_meta.json',m)
    hashes={name:sha256(dest/name) for name in INPUT_NAMES}
    save_json(dest/'manifest.json',dict(schema_version=1,case_id=fingerprint(m,hashes),preset='imported',input_origin='imported_existing_case',source_directory=str(source.resolve()),meta=m,sha256=hashes,output_contract={'order':'target_repository_order_unverified'}))
    c=Case(dest);print(c.id)
if __name__=='__main__':main()
