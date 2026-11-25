import random
from pathlib import Path
from .data import find_pairs,read_rgb,read_mask,mask_to_box
from .utils import save_json

def prepare_splits(image_dir,mask_dir,split_dir,sample_size=0,seed=42,train=.7,val=.15):
    if train <= 0 or val < 0 or train + val >= 1.0:
        raise ValueError('Tỷ lệ split không hợp lệ: train phải > 0, val >= 0 và train + val < 1.')
    pairs,_,_=find_pairs(image_dir,mask_dir)
    valid=[]
    for k,ip,mp in pairs:
        try:
            image=read_rgb(ip); mask=read_mask(mp)
            if image.shape[:2] != mask.shape: continue
            valid.append({'key':k,'image':str(ip.resolve()),'mask':str(mp.resolve())})
        except Exception: continue
    rng=random.Random(seed); rng.shuffle(valid)
    if sample_size and sample_size>0: valid=valid[:min(sample_size,len(valid))]
    n=len(valid); ntr=int(n*train); nval=int(n*val)
    split={'train':valid[:ntr],'val':valid[ntr:ntr+nval],'test':valid[ntr+nval:],'seed':seed,'sample_size':n,'policy':'image-level random split; lesion-level grouping not inferred'}
    Path(split_dir).mkdir(parents=True,exist_ok=True); save_json(split,Path(split_dir)/'split.json')
    return split

def load_records(split_path,part):
    import json
    d=json.loads(Path(split_path).read_text(encoding='utf-8'))
    return [(x['key'],Path(x['image']),Path(x['mask'])) for x in d[part]]
