import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.e1 = DoubleConv(in_channels, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.mid(self.pool(d))
        z = self.d4(torch.cat([self.u4(z), d], 1))
        z = self.d3(torch.cat([self.u3(z), c], 1))
        z = self.d2(torch.cat([self.u2(z), b], 1))
        z = self.d1(torch.cat([self.u1(z), a], 1))
        return self.out(z)


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        att = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        return x * att


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32):
        super().__init__()
        self.e1 = DoubleConv(in_channels, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.ag4 = AttentionGate(base * 8, base * 8, base * 4)
        self.d4 = DoubleConv(base * 16, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.ag3 = AttentionGate(base * 4, base * 4, base * 2)
        self.d3 = DoubleConv(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.ag2 = AttentionGate(base * 2, base * 2, base)
        self.d2 = DoubleConv(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.ag1 = AttentionGate(base, base, max(1, base // 2))
        self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.mid(self.pool(d))
        g4 = self.u4(z)
        z = self.d4(torch.cat([g4, self.ag4(g4, d)], 1))
        g3 = self.u3(z)
        z = self.d3(torch.cat([g3, self.ag3(g3, c)], 1))
        g2 = self.u2(z)
        z = self.d2(torch.cat([g2, self.ag2(g2, b)], 1))
        g1 = self.u1(z)
        z = self.d1(torch.cat([g1, self.ag1(g1, a)], 1))
        return self.out(z)


class UNetResNet34(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        enc = resnet34(weights=weights)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.l1 = enc.layer1
        self.l2 = enc.layer2
        self.l3 = enc.layer3
        self.l4 = enc.layer4

        self.u4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d4 = DoubleConv(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = DoubleConv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = DoubleConv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.d1 = DoubleConv(128, 64)
        self.u0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d0 = DoubleConv(32, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        s0 = self.stem(x_norm)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        z = self.d4(torch.cat([self.u4(s4), s3], 1))
        z = self.d3(torch.cat([self.u3(z), s2], 1))
        z = self.d2(torch.cat([self.u2(z), s1], 1))
        z = self.d1(torch.cat([self.u1(z), s0], 1))
        z = self.d0(self.u0(z))
        if z.shape[2:] != x.shape[2:]:
            z = F.interpolate(z, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.out(z)


def build_segmentation_model(model_name='unet', pretrained=True, in_channels=3, out_channels=1, base=32):
    name = (model_name or 'unet').lower()
    if name == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels, base=base)
    if name == 'attention_unet':
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, base=base)
    if name == 'unet_resnet34':
        return UNetResNet34(out_channels=out_channels, pretrained=pretrained)
    raise ValueError(f'Mô hình segmentation không hỗ trợ: {model_name}')


def build_detector(
    pretrained=True,
    model_name='fasterrcnn_resnet50_fpn_v2',
    image_size=512,
    trainable_backbone_layers=None,
    box_nms_thresh=0.5,
    box_detections_per_img=5,
):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights

    kwargs = {}
    if pretrained and trainable_backbone_layers is not None:
        kwargs['trainable_backbone_layers'] = int(trainable_backbone_layers)

    if model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None, **kwargs)
    else:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None, **kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.roi_heads.nms_thresh = float(box_nms_thresh)
    model.roi_heads.detections_per_img = int(box_detections_per_img)
    # Tránh torchvision tự phóng ảnh 512 lên 800 px, giảm VRAM trên GPU 4 GB.
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    return model
