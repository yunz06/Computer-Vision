from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
from .data import SegmentationDataset
from .prepare import load_records
from .models import UNet
from .losses import segmentation_loss
from .metrics import binary_metrics
from .utils import device,save_json,save_loss_plot

def run(cfg,epochs=None,limit=0):
    split_path=Path(cfg['paths']['splits'])/'split.json'; tr=load_records(split_path,'train'); va=load_records(split_path,'val')
    if limit: tr=tr[:limit]; va=va[:max(1,limit//4)]
    if not tr: raise RuntimeError('Train split rỗng. Hãy chạy lệnh prepare trước.')
    size=cfg['image_size']; bs=cfg['segmentation']['batch_size']; dev=device()
    train_loader=DataLoader(SegmentationDataset(tr,size),batch_size=bs,shuffle=True,num_workers=0)
    val_loader=DataLoader(SegmentationDataset(va,size),batch_size=1,shuffle=False,num_workers=0) if va else []
    model=UNet().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=cfg['segmentation']['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=(dev.type=='cuda' and cfg['segmentation'].get('amp',True)))
    best=-1; history=[]; n_epochs=epochs or cfg['segmentation']['epochs']
    for ep in range(1,n_epochs+1):
        model.train(); losses=[]
        for x,y in train_loader:
            x=x.to(dev); y=y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=(dev.type=='cuda')):
                logits=model(x); loss=segmentation_loss(logits,y,cfg['segmentation']['loss'])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(loss.item())
        model.eval(); vals=[]
        with torch.no_grad():
            for x,y in val_loader:
                logits=model(x.to(dev)); pred=(torch.sigmoid(logits).cpu().numpy()[0,0]>=.5); vals.append(binary_metrics(pred,y.numpy()[0,0]))
        dice=sum(v['dice'] for v in vals)/len(vals) if vals else 0.0
        row={'epoch':ep,'train_loss':sum(losses)/len(losses),'val_dice':dice}; history.append(row); print(row)
        if dice>best:
            best=dice; Path(cfg['paths']['checkpoints']).mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'epoch':ep,'val_dice':dice,'image_size':size},Path(cfg['paths']['checkpoints'])/'unet_best.pt')
    save_json(history,Path(cfg['paths']['results'])/'seg_training.json')
    save_loss_plot(history,Path(cfg['paths']['figures'])/'seg_training.png','U-Net training')
    return history
