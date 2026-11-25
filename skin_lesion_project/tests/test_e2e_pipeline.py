import json
import numpy as np
from PIL import Image
from src.audit import run_audit
from src.eda import run_eda
from src.prepare import prepare_splits
from src.baselines import run as run_baselines
from src.train_seg import run as run_train_seg
from src.train_det import run as run_train_det
from src.evaluate import select_on_validation, eval_seg, eval_det, eval_unet_box
from src.integration import load_models, run_integration


def test_full_end_to_end_pipeline(tmp_path):
    im_dir = tmp_path / 'raw' / 'images'
    ma_dir = tmp_path / 'raw' / 'masks'
    im_dir.mkdir(parents=True)
    ma_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    for i in range(12):
        arr = rng.integers(150, 220, (72, 96, 3), dtype=np.uint8)
        mask = np.zeros((72, 96), dtype=np.uint8)
        r0, c0 = 14 + (i % 6) * 2, 20 + (i % 5) * 3
        arr[r0:r0 + 28, c0:c0 + 34] = 35
        mask[r0:r0 + 28, c0:c0 + 34] = 255
        Image.fromarray(arr).save(im_dir / f'ISIC_{i:07d}.jpg')
        Image.fromarray(mask).save(ma_dir / f'ISIC_{i:07d}_segmentation.png')

    cfg = {
        'seed': 42,
        'paths': {
            'images': str(im_dir),
            'masks': str(ma_dir),
            'processed': str(tmp_path / 'processed'),
            'splits': str(tmp_path / 'splits'),
            'reports': str(tmp_path / 'reports'),
            'checkpoints': str(tmp_path / 'checkpoints'),
            'results': str(tmp_path / 'results'),
            'figures': str(tmp_path / 'figures'),
            'demo': str(tmp_path / 'demo'),
        },
        'image_size': 64,
        'detection_size': 128,
        'data': {
            'mask_threshold': 127,
            'min_box_area_ratio': 0.001,
            'multi_component_boxes': True,
            'use_phash_grouping': False,
            'phash_max_distance': 4,
        },
        'segmentation': {
            'model': 'unet',
            'pretrained': False,
            'epochs': 1,
            'batch_size': 4,
            'learning_rate': 0.001,
            'num_workers': 0,
            'loss': 'bce_dice',
            'threshold': 0.5,
            'amp': True,
            'augment': True,
            'largest_component': False,
            'hole_filling': False,
        },
        'detection': {
            'epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.0005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'num_workers': 0,
            'score_threshold': 0.1,
            'nms_iou_threshold': 0.5,
            'max_detections': 5,
            'trainable_backbone_layers': 3,
            'amp': True,
            'augment': True,
            'model': 'fasterrcnn_mobilenet_v3_large_fpn',
            'pretrained': False,
        },
        'evaluation': {
            'boundary_tolerance_px': 2,
            'consistency_iou_threshold': 0.5,
            'qualitative_count': 2,
        },
    }

    audit_res = run_audit(im_dir, ma_dir, cfg['paths']['reports'])
    assert audit_res['paired_count'] == 12

    eda_res = run_eda(im_dir, ma_dir, cfg['paths']['reports'], cfg['paths']['figures'])
    assert eda_res['sample_count'] == 12

    split_res = prepare_splits(im_dir, ma_dir, cfg['paths']['splits'], seed=42)
    assert len(split_res['train']) == 8
    assert len(split_res['val']) == 1
    assert len(split_res['test']) == 3

    base_res = run_baselines(im_dir, ma_dir, cfg['paths']['results'], size=64)
    assert len(base_res) == 12

    seg_hist = run_train_seg(cfg, epochs=1)
    assert len(seg_hist) == 1
    assert (tmp_path / 'checkpoints' / 'unet_best.pt').exists()

    det_hist = run_train_det(cfg, epochs=1, pretrained=False)
    assert len(det_hist) == 1
    assert 'val_ap5095' in det_hist[0]
    assert (tmp_path / 'checkpoints' / 'fasterrcnn_last.pt').exists()

    val_sel = select_on_validation(cfg)
    assert 'segmentation_best' in val_sel and 'detection_best' in val_sel

    seg_eval = eval_seg(cfg, split='test')
    assert seg_eval['count'] == 3
    assert 'mean_dice' in seg_eval and 'std_dice' in seg_eval and 'mean_boundary_f1' in seg_eval
    assert (tmp_path / 'figures' / 'seg_test_qualitative.png').exists()

    det_eval = eval_det(cfg, split='test')
    assert det_eval['count'] == 3
    assert 'ap5095' in det_eval and 'top1_iou_at_0_5' in det_eval
    assert (tmp_path / 'figures' / 'det_test_qualitative.png').exists()

    unet_box_eval = eval_unet_box(cfg, split='test')
    assert unet_box_eval['count'] == 3

    seg_m, det_m, dev, errs = load_models(cfg)
    assert not errs and seg_m is not None and det_m is not None
    sample_img = Image.open(im_dir / 'ISIC_0000000.jpg')
    integ = run_integration(sample_img, seg_m, det_m, dev, cfg)
    assert integ['binary_mask'].shape == (72, 96)
    assert json.loads((tmp_path / 'results' / 'seg_test_metrics.json').read_text(encoding='utf-8'))['count'] == 3
