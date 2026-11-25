from pathlib import Path
import numpy as np
from PIL import Image
from src.data import find_pairs
from src.prepare import prepare_splits

def test_pair_and_split(tmp_path):
    im=tmp_path/'images'; ma=tmp_path/'masks'; im.mkdir(); ma.mkdir()
    for i in range(10):
        Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(im/f'ISIC_{i:07d}.jpg')
        mask=np.zeros((16,16),dtype=np.uint8); mask[3:8,3:8]=255
        Image.fromarray(mask).save(ma/f'ISIC_{i:07d}_segmentation.png')
    pairs,mi,orphan=find_pairs(im,ma); assert len(pairs)==10 and not mi and not orphan
    s=prepare_splits(im,ma,tmp_path/'splits',sample_size=8,seed=7)
    assert len(s['train'])+len(s['val'])+len(s['test'])==8

def test_detection_dataset_skips_empty_masks(tmp_path):
    from src.data import DetectionDataset
    im=tmp_path/'images'; ma=tmp_path/'masks'; im.mkdir(); ma.mkdir()
    for i in range(2):
        Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(im/f'ISIC_{i:07d}.jpg')
        mask=np.zeros((16,16),dtype=np.uint8)
        if i == 1: mask[4:10,5:11]=255
        Image.fromarray(mask).save(ma/f'ISIC_{i:07d}_segmentation.png')
    pairs,_,_=find_pairs(im,ma)
    ds=DetectionDataset(pairs,size=32)
    assert len(ds)==1
    _,target=ds[0]
    assert target['boxes'].shape==(1,4)
    assert target['labels'].tolist()==[1]
