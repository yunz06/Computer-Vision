from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .prepare import load_records
from .data import SegmentationDataset,DetectionDataset,detection_collate,read_mask,mask_to_box
from .models import UNet,build_detector
from .metrics import binary_metrics,box_iou,detection_ap,detection_precision_recall,segmentation_summary
from .utils import device,save_json

def eval_seg(cfg):
    ck=Path(cfg['paths']['checkpoints'])/'unet_best.pt'
    if not ck.exists(): raise FileNotFoundError('Chưa có outputs/checkpoints/unet_best.pt')
    records=load_records(Path(cfg['paths']['splits'])/'split.json','test'); ds=SegmentationDataset(records,cfg['image_size']); loader=DataLoader(ds,batch_size=1)
    dev=device(); model=UNet().to(dev); model.load_state_dict(torch.load(ck,map_location=dev,weights_only=False)['model']); model.eval(); rows=[]
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            p=torch.sigmoid(model(x.to(dev))).cpu().numpy()[0,0]>=.5; rows.append(binary_metrics(p,y.numpy()[0,0]))
    result=segmentation_summary(rows); save_json(result,Path(cfg['paths']['results'])/'seg_test_metrics.json'); return result

def eval_det(cfg):
    ck=Path(cfg['paths']['checkpoints'])/'fasterrcnn_best.pt'
    if not ck.exists(): ck=Path(cfg['paths']['checkpoints'])/'fasterrcnn_last.pt'
    if not ck.exists(): raise FileNotFoundError('Chưa có outputs/checkpoints/fasterrcnn_last.pt')
    records=load_records(Path(cfg['paths']['splits'])/'split.json','test'); ds=DetectionDataset(records,cfg['detection_size']); loader=DataLoader(ds,batch_size=1,collate_fn=detection_collate)
    dev=device(); state=torch.load(ck,map_location=dev,weights_only=False); model=build_detector(False,state.get('model_name',cfg['detection']['model']),cfg['detection_size']).to(dev); model.load_state_dict(state['model']); model.eval(); preds=[]; gts=[]; top_ious=[]
    with torch.no_grad():
        for images,targets in loader:
            out=model([images[0].to(dev)])[0]; preds.append({k:v.detach().cpu() for k,v in out.items()}); gt=targets[0]['boxes'].cpu(); gts.append(gt)
            if len(out['boxes']): top_ious.append(box_iou(out['boxes'][0].cpu().numpy(),gt[0].numpy()))
            else: top_ious.append(0.)
    result={'count':len(gts),'ap50':detection_ap(preds,gts,.5),'ap5095':float(np.mean([detection_ap(preds,gts,t) for t in np.arange(.5,.96,.05)])),'top1_box_iou':float(np.mean(top_ious)) if top_ious else 0.}; result.update(detection_precision_recall(preds,gts,.5,.5))
    save_json(result,Path(cfg['paths']['results'])/'det_test_metrics.json'); return result
