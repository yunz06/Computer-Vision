import numpy as np
import torch
from src.data import (
    image_key,
    read_mask,
    letterbox,
    unletterbox_prob_map,
    unletterbox_mask,
    unletterbox_boxes,
    postprocess_mask,
    mask_to_box,
    mask_to_boxes,
    augment_pair,
)
from src.metrics import (
    binary_metrics,
    boundary_f1,
    box_iou,
    detection_ap,
    detection_ap5095,
    detection_precision_recall,
    segmentation_summary,
    top1_box_iou_summary,
)
from src.losses import segmentation_loss
from src.models import UNet, AttentionUNet, UNetResNet34, build_segmentation_model, build_detector


def test_image_key():
    assert image_key('ISIC_0001_segmentation.png') == 'isic_0001'


def test_read_mask_threshold_127(tmp_path):
    from PIL import Image
    arr = np.array([[0, 100, 127, 128, 255]], dtype=np.uint8)
    p = tmp_path / 'm.png'
    Image.fromarray(arr).save(p)
    m = read_mask(p, threshold=127)
    assert m.tolist() == [[0, 0, 0, 1, 1]]


def test_letterbox_and_unletterbox_roundtrip():
    im = np.zeros((100, 200, 3), dtype=np.uint8)
    m = np.zeros((100, 200), dtype=np.uint8)
    m[20:60, 30:80] = 1
    x, y, meta = letterbox(im, 256, m)
    assert x.shape == (256, 256, 3)
    assert y.shape == (256, 256)
    assert set(np.unique(y)) <= {0, 1}
    assert meta['new_width'] == 256

    prob_orig = unletterbox_prob_map(y.astype(np.float32), meta)
    assert prob_orig.shape == (100, 200)
    assert prob_orig.min() >= 0.0 and prob_orig.max() <= 1.0
    assert binary_metrics(prob_orig >= 0.5, m)['iou'] > 0.93

    restored = unletterbox_mask(y, meta)
    assert restored.shape == (100, 200)
    assert binary_metrics(restored, m)['iou'] > 0.90

    box_lb = mask_to_box(y)
    boxes_orig = unletterbox_boxes(box_lb[None, :], meta)
    assert boxes_orig.shape == (1, 4)
    assert box_iou(boxes_orig[0], [30, 20, 80, 60]) > 0.9


def test_mask_box_and_multi_component_filter():
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:40, 15:55] = 1   # area = 1200 (12%)
    m[60:80, 60:80] = 1   # area = 400 (4%)
    m[2:4, 2:4] = 1       # area = 4 (0.04% < 0.1%)
    boxes = mask_to_boxes(m, min_area_ratio=0.001, fallback_largest=False)
    assert len(boxes) == 2
    assert np.allclose(boxes[0], [15, 10, 55, 40])
    assert np.allclose(boxes[1], [60, 60, 80, 80])
    assert np.allclose(mask_to_box(m), [15, 10, 55, 40])
    assert mask_to_box(np.zeros_like(m)) is None


def test_postprocess_mask_largest_and_hole_filling():
    m = np.zeros((30, 30), dtype=np.uint8)
    m[5:20, 5:20] = 1
    m[10:14, 10:14] = 0  # hole inside main component
    m[25:27, 25:27] = 1  # small stray speck
    out = postprocess_mask(m, largest_component=True, hole_filling=True)
    assert out[11, 11] == 1  # hole filled
    assert out[26, 26] == 0  # stray speck removed


def test_augment_pair_keeps_sync():
    rng = np.random.default_rng(123)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    img[10:30, 20:40] = 200
    mask[10:30, 20:40] = 1
    aug_img, aug_mask = augment_pair(img, mask, rng, allow_affine=False)
    assert aug_img.shape == (64, 64, 3)
    assert aug_mask.shape == (64, 64)
    assert set(np.unique(aug_mask)) <= {0, 1}
    # Foreground pixels remain bright after geometric flip/rot90 + mild brightness
    assert aug_img[aug_mask == 1].mean() > aug_img[aug_mask == 0].mean() + 50


def test_metrics_and_boundary_f1():
    a = np.zeros((20, 20), dtype=np.uint8)
    a[4:16, 4:16] = 1
    m = binary_metrics(a, a)
    assert m['dice'] > 0.999
    assert m['iou'] > 0.999
    assert m['boundary_f1'] > 0.999
    assert abs(box_iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1 / 7) < 1e-6

    s = segmentation_summary([
        {'dice': 0.8, 'iou': 0.7, 'thresholded_jaccard_065': 0.7, 'boundary_f1': 0.9},
        {'dice': 0.6, 'iou': 0.5, 'thresholded_jaccard_065': 0.0, 'boundary_f1': 0.5},
    ])
    assert abs(s['failure_rate_jaccard_below_0_65'] - 0.5) < 1e-6
    assert 'std_dice' in s and 'mean_boundary_f1' in s


def test_loss_and_segmentation_models_forward():
    x = torch.rand(2, 3, 64, 64)
    y = torch.randint(0, 2, (2, 1, 64, 64)).float()
    for name in ('unet', 'attention_unet', 'unet_resnet34'):
        model = build_segmentation_model(name, pretrained=False, base=8)
        z = model(x)
        assert z.shape == y.shape
        for kind in ('bce', 'dice', 'bce_dice'):
            loss = segmentation_loss(z, y, kind=kind)
            assert torch.isfinite(loss)


def test_detection_metrics_and_detectors():
    pred = [{'boxes': torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]), 'scores': torch.tensor([0.9, 0.2])}]
    gt = [torch.tensor([[0., 0., 10., 10.]])]
    assert detection_ap(pred, gt, 0.5) > 0.99
    assert detection_ap5095(pred, gt) > 0.99
    pr = detection_precision_recall(pred, gt, 0.5, 0.5)
    assert pr['precision_at_score_0_5_iou_0_5'] == 1.0
    assert pr['recall_at_score_0_5_iou_0_5'] == 1.0
    top1 = top1_box_iou_summary([1.0, 0.6])
    assert top1['top1_iou_at_0_5'] == 1.0
    assert top1['top1_iou_at_0_75'] == 0.5

    for arch in ('fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn_v2'):
        det = build_detector(pretrained=False, model_name=arch, image_size=128)
        det.eval()
        with torch.no_grad():
            out = det([torch.rand(3, 128, 128)])
        assert len(out) == 1 and 'boxes' in out[0] and 'scores' in out[0]
