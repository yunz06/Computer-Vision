import numpy as np, torch
from PIL import Image
from src.data import image_key,letterbox,mask_to_box
from src.metrics import binary_metrics,box_iou
from src.losses import segmentation_loss
from src.models import UNet

def test_image_key(): assert image_key('ISIC_0001_segmentation.png')=='isic_0001'
def test_letterbox_keeps_shape_and_mask_binary():
    im=np.zeros((100,200,3),dtype=np.uint8); m=np.zeros((100,200),dtype=np.uint8); m[20:60,30:80]=1
    x,y,meta=letterbox(im,256,m); assert x.shape==(256,256,3); assert y.shape==(256,256); assert set(np.unique(y))<={0,1}; assert meta['new_width']==256

def test_mask_box():
    m=np.zeros((10,10),dtype=np.uint8); m[2:5,3:8]=1
    assert np.allclose(mask_to_box(m),[3,2,8,5]); assert mask_to_box(np.zeros_like(m)) is None

def test_metrics():
    a=np.array([[1,1],[0,0]]); assert binary_metrics(a,a)['dice']>0.999; assert abs(box_iou([0,0,2,2],[1,1,3,3])-1/7)<1e-6

def test_loss_and_unet_forward():
    model=UNet(base=8); x=torch.rand(2,3,64,64); y=torch.randint(0,2,(2,1,64,64)).float(); z=model(x)
    assert z.shape==y.shape; loss=segmentation_loss(z,y); assert torch.isfinite(loss)

def test_detection_ap_and_pr():
    from src.metrics import detection_ap, detection_precision_recall
    pred = [{'boxes': torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]), 'scores': torch.tensor([0.9, 0.2])}]
    gt = [torch.tensor([[0., 0., 10., 10.]])]
    assert detection_ap(pred, gt, 0.5) > 0.99
    pr = detection_precision_recall(pred, gt, 0.5, 0.5)
    assert pr['precision_at_score_0_5_iou_0_5'] == 1.0
    assert pr['recall_at_score_0_5_iou_0_5'] == 1.0


def test_empty_mask_metrics():
    empty = np.zeros((8, 8), dtype=np.uint8)
    result = binary_metrics(empty, empty)
    assert result['dice'] == 1.0
    assert result['iou'] == 1.0

def test_mask_box_uses_largest_component_only():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:15, 6:18] = 1
    mask[25:27, 25:27] = 1
    assert np.allclose(mask_to_box(mask), [6, 5, 18, 15])
