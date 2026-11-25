import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,base=32):
        super().__init__(); self.e1=DoubleConv(in_channels,base); self.e2=DoubleConv(base,base*2); self.e3=DoubleConv(base*2,base*4); self.e4=DoubleConv(base*4,base*8); self.pool=nn.MaxPool2d(2); self.mid=DoubleConv(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=DoubleConv(base*16,base*8); self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=DoubleConv(base*8,base*4); self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=DoubleConv(base*4,base*2); self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=DoubleConv(base*2,base); self.out=nn.Conv2d(base,out_channels,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); d=self.e4(self.pool(c)); z=self.mid(self.pool(d))
        z=self.d4(torch.cat([self.u4(z),d],1)); z=self.d3(torch.cat([self.u3(z),c],1)); z=self.d2(torch.cat([self.u2(z),b],1)); z=self.d1(torch.cat([self.u1(z),a],1)); return self.out(z)

def build_detector(pretrained=True,model_name='fasterrcnn_resnet50_fpn_v2',image_size=512):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    if model_name=='fasterrcnn_mobilenet_v3_large_fpn':
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model=fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
    else:
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model=fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    in_features=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,2)
    # Tránh torchvision tự phóng ảnh 512 lên 800 px, giảm VRAM trên GPU 4 GB.
    model.transform.min_size=(image_size,)
    model.transform.max_size=image_size
    return model
