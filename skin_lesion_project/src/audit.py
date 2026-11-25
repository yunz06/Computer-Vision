from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from .data import find_pairs,read_mask,read_rgb
from .utils import save_json

def run_audit(image_dir,mask_dir,report_dir):
    pairs,missing_masks,orphan_masks=find_pairs(image_dir,mask_dir)
    bad_images=[]; bad_masks=[]; size_mismatch=[]; empty=[]; sizes=Counter(); components=Counter()
    for key,ip,mp in pairs:
        try:
            with Image.open(ip) as im: sizes[str(im.size)]+=1; im.verify()
        except Exception as e: bad_images.append({'key':key,'error':str(e)}); continue
        try:
            m=read_mask(mp)
            image_shape=read_rgb(ip).shape[:2]
            if m.shape != image_shape: size_mismatch.append({'key':key,'image_shape':list(image_shape),'mask_shape':list(m.shape)})
            if not m.any(): empty.append(key)
            n,_,stats,_=__import__('cv2').connectedComponentsWithStats(m,8)
            components[str(max(0,n-1))]+=1
        except Exception as e: bad_masks.append({'key':key,'error':str(e)})
    result={'image_count':sum(1 for p in Path(image_dir).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'} and not p.stem.lower().endswith('_segmentation')),'mask_count':sum(1 for p in Path(mask_dir).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'} and p.stem.lower().endswith('_segmentation')),'paired_count':len(pairs),'missing_masks':missing_masks,'orphan_masks':orphan_masks,'bad_images':bad_images,'bad_masks':bad_masks,'mask_size_mismatch':size_mismatch,'empty_masks':empty,'image_sizes':dict(sizes),'connected_component_count':dict(components),'detection_usable_count':sum(1 for key, _, _ in pairs if key not in set(empty) and key not in {x['key'] for x in bad_images} and key not in {x['key'] for x in bad_masks} and key not in {x['key'] for x in size_mismatch})}
    Path(report_dir).mkdir(parents=True,exist_ok=True); save_json(result,Path(report_dir)/'audit.json')
    lines=[f"Cặp ảnh-mask: {len(pairs)}",f"Thiếu mask: {len(missing_masks)}",f"Mask không có ảnh: {len(orphan_masks)}",f"Ảnh lỗi: {len(bad_images)}",f"Mask lỗi: {len(bad_masks)}",f"Sai kích thước ảnh-mask: {len(size_mismatch)}",f"Mask rỗng: {len(empty)}",f"Mẫu detection dùng được (ước tính): {result['detection_usable_count']}"]
    (Path(report_dir)/'audit.txt').write_text('\n'.join(lines),encoding='utf-8')
    return result
