from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

IMAGE_EXTS={'.jpg','.jpeg','.png'}

def image_key(path):
    stem=Path(path).stem
    if stem.lower().endswith('_segmentation'):
        stem=stem[:-13]
    return stem.lower()

def find_pairs(image_dir, mask_dir):
    image_dir,mask_dir=Path(image_dir),Path(mask_dir)
    imgs={image_key(p):p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.stem.lower().endswith('_segmentation')}
    masks={image_key(p):p for p in mask_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.lower().endswith('_segmentation')}
    return [(k,imgs[k],masks[k]) for k in sorted(imgs.keys() & masks.keys())], sorted(imgs.keys()-masks.keys()), sorted(masks.keys()-imgs.keys())

def read_rgb(path):
    with Image.open(path) as im: return np.asarray(im.convert('RGB'))

def read_mask(path):
    with Image.open(path) as im: arr=np.asarray(im.convert('L'))
    return (arr>0).astype(np.uint8)

def letterbox(image, size, mask=None, image_fill=(0,0,0)):
    h,w=image.shape[:2]; scale=min(size/w,size/h)
    nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))
    interp=cv2.INTER_LINEAR
    resized=cv2.resize(image,(nw,nh),interpolation=interp)
    left=(size-nw)//2; top=(size-nh)//2
    canvas=np.full((size,size,3),image_fill,dtype=np.uint8)
    canvas[top:top+nh,left:left+nw]=resized
    meta={'original_width':w,'original_height':h,'scale':scale,'left':left,'top':top,'new_width':nw,'new_height':nh,'size':size}
    if mask is None: return canvas,meta
    m=cv2.resize(mask.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)
    mc=np.zeros((size,size),dtype=np.uint8); mc[top:top+nh,left:left+nw]=m
    return canvas,mc,meta

def mask_to_box(mask):
    binary=(np.asarray(mask)>0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1: return None
    # Chỉ lấy thành phần liên thông lớn nhất làm tổn thương chính.
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[component]
    return np.array([x, y, x+w, y+h], dtype=np.float32)

class SegmentationDataset(Dataset):
    def __init__(self, records, size=256): self.records=records; self.size=size
    def __len__(self): return len(self.records)
    def __getitem__(self,idx):
        _,ip,mp=self.records[idx]
        image=read_rgb(ip); mask=read_mask(mp)
        image,mask,_=letterbox(image,self.size,mask)
        x=torch.from_numpy(image.copy()).permute(2,0,1).float()/255.0
        y=torch.from_numpy(mask.copy()).unsqueeze(0).float()
        return x,y

class DetectionDataset(Dataset):
    def __init__(self, records, size=512):
        self.records=[]; self.size=size
        for r in records:
            try:
                m=read_mask(r[2]);
                if mask_to_box(m) is not None: self.records.append(r)
            except Exception: pass
    def __len__(self): return len(self.records)
    def __getitem__(self,idx):
        _,ip,mp=self.records[idx]
        image=read_rgb(ip); mask=read_mask(mp)
        image,mask,meta=letterbox(image,self.size,mask)
        box=mask_to_box(mask)
        x=torch.from_numpy(image.copy()).permute(2,0,1).float()/255.0
        target={'boxes':torch.from_numpy(box[None, :].copy()),'labels':torch.ones((1,),dtype=torch.int64),'image_id':torch.tensor([idx]),'area':torch.tensor([(box[2]-box[0])*(box[3]-box[1])]),'iscrowd':torch.zeros((1,),dtype=torch.int64)}
        return x,target

def detection_collate(batch): return tuple(zip(*batch))
