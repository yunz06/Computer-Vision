from pathlib import Path
import cv2, numpy as np
from sklearn.cluster import KMeans
from .data import find_pairs,read_rgb,read_mask,letterbox,mask_to_box
from .metrics import binary_metrics, box_iou
from .utils import save_json

def run(image_dir,mask_dir,out_dir,limit=0,size=256,records=None):
    pairs = records if records is not None else find_pairs(image_dir,mask_dir)[0]
    pairs=pairs[:limit] if limit else pairs
    results=[]
    for key,ip,mp in pairs:
        image=read_rgb(ip); gt=read_mask(mp); img,gt,meta=letterbox(image,size,gt)
        lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        valid=np.zeros((size,size),dtype=bool); l,t=meta['left'],meta['top']; valid[t:t+meta['new_height'],l:l+meta['new_width']]=True
        _,otsu=cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Chọn vùng tối hơn trên phần ảnh thật, không dùng ground truth để chọn cực tính.
        kernel=np.ones((5,5),np.uint8)
        otsu_candidates=[otsu.astype(bool)&valid,(~otsu.astype(bool))&valid]
        otsu_pred=min(otsu_candidates,key=lambda p:float(gray[p].mean()) if p.any() else float('inf'))
        otsu_pred=cv2.morphologyEx((otsu_pred*255).astype('uint8'),cv2.MORPH_CLOSE,kernel)>0; otsu_pred &= valid
        pixels=lab[valid].reshape(-1,3).astype(np.float32)
        km=KMeans(n_clusters=2,random_state=42,n_init=5).fit(pixels)
        labels=np.full((size,size),-1,dtype=np.int32); labels[valid]=km.labels_
        lightness=[float(lab[:,:,0][labels==i].mean()) if np.any(labels==i) else float('inf') for i in (0,1)]
        km_pred=(labels==int(np.argmin(lightness)))
        gt_box = mask_to_box(gt)
        h, w = gt.shape
        center_box = np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75], dtype=np.float32)
        center_iou = box_iou(center_box, gt_box) if gt_box is not None else 0.0
        results.append({'key':key,'otsu_dice':binary_metrics(otsu_pred,gt)['dice'],'kmeans_lab_dice':binary_metrics(km_pred,gt)['dice'],'center_box_iou':center_iou})
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    save_json(results,Path(out_dir)/'baseline_results.json')
    summary={key:float(np.mean([row[key] for row in results])) for key in ('otsu_dice','kmeans_lab_dice','center_box_iou')} if results else {'count':0}
    summary['count']=len(results)
    save_json(summary,Path(out_dir)/'baseline_summary.json')
    return results
