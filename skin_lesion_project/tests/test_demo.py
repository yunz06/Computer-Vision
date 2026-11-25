import numpy as np
import pytest
import torch
from PIL import Image
from src.integration import validate_input_image, run_integration
from src.visualization import render_panels
from src.demo import infer, format_status, save_result_files, build_app


class DummySeg(torch.nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1, 64, 64), device=x.device) * 8.0


class EmptySeg(torch.nn.Module):
    def forward(self, x):
        return torch.ones((x.shape[0], 1, 64, 64), device=x.device) * -8.0


class BrokenModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError('Simulated model failure')


class DummyDet(torch.nn.Module):
    def forward(self, images):
        return [{
            'boxes': torch.tensor([[10., 40., 110., 90.]], device=images[0].device),
            'scores': torch.tensor([0.9], device=images[0].device),
            'labels': torch.tensor([1], device=images[0].device),
        }]


class EmptyDet(torch.nn.Module):
    def forward(self, images):
        return [{
            'boxes': torch.zeros((0, 4), device=images[0].device),
            'scores': torch.zeros((0,), device=images[0].device),
            'labels': torch.zeros((0,), dtype=torch.int64, device=images[0].device),
        }]


def _cfg(tmp_path):
    return {
        'image_size': 64,
        'detection_size': 128,
        'segmentation': {'threshold': 0.5, 'largest_component': False, 'hole_filling': False},
        'detection': {'score_threshold': 0.5},
        'evaluation': {'consistency_iou_threshold': 0.5},
        'paths': {'demo': str(tmp_path)},
    }


def test_demo_inference_and_saving(tmp_path):
    cfg = _cfg(tmp_path)
    arr = np.full((100, 200, 3), 120, dtype=np.uint8)
    image = Image.fromarray(arr.copy())
    overlay, mask, status = infer(image, DummySeg(), DummyDet(), torch.device('cpu'), cfg)
    assert overlay.size == image.size
    assert mask.size == image.size
    assert 'Confidence' in status and 'Box–Mask IoU' in status
    assert np.array_equal(np.asarray(image), arr)  # original image untouched
    assert len(list(tmp_path.glob('*.png'))) == 3  # combined, overlay, mask


def test_integration_standard_output_and_panels(tmp_path):
    cfg = _cfg(tmp_path)
    image = Image.fromarray(np.full((100, 200, 3), 120, dtype=np.uint8))
    res = run_integration(image, DummySeg(), DummyDet(), torch.device('cpu'), cfg)
    for k in (
        'original_image', 'boxes', 'labels', 'scores', 'prob_map',
        'binary_mask', 'mask_bbox', 'box_mask_iou', 'consistency_status', 'inference_time_ms',
    ):
        assert k in res
    assert res['prob_map'].shape == (100, 200)
    assert res['binary_mask'].shape == (100, 200)
    assert res['box_mask_iou'] is not None
    orig_p, det_p, mask_p, ov_p, comb_p = render_panels(res)
    assert orig_p.size == (200, 100)
    assert det_p.size == (200, 100)
    assert mask_p.size == (200, 100)
    assert ov_p.size == (200, 100)
    assert comb_p.size == (400, 200)
    saved = save_result_files(res, tmp_path / 'explicit_save')
    assert len(saved) == 3


def test_empty_and_broken_branches_and_invalid_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    image = Image.fromarray(np.full((80, 80, 3), 100, dtype=np.uint8))

    res_empty_det = run_integration(image, DummySeg(), EmptyDet(), torch.device('cpu'), cfg)
    assert res_empty_det['box_mask_iou'] is None
    assert 'Detection rỗng' in res_empty_det['consistency_status']

    res_empty_mask = run_integration(image, EmptySeg(), DummyDet(), torch.device('cpu'), cfg)
    assert res_empty_mask['box_mask_iou'] is None
    assert 'Mask rỗng' in res_empty_mask['consistency_status']

    res_broken = run_integration(image, BrokenModel(), DummyDet(), torch.device('cpu'), cfg)
    assert res_broken['binary_mask'] is None
    assert 'segmentation' in res_broken['errors']
    assert 'thiếu nhánh Segmentation' in res_broken['consistency_status']
    assert 'Lỗi nhánh Segmentation' in format_status(res_broken)

    with pytest.raises(ValueError):
        validate_input_image(None)
    bad_file = tmp_path / 'bad.txt'
    bad_file.write_text('hello', encoding='utf-8')
    with pytest.raises(ValueError):
        validate_input_image(bad_file)
    tiny = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_input_image(tiny)


def test_gradio_app_builds(tmp_path):
    cfg = _cfg(tmp_path)
    cfg['paths']['checkpoints'] = str(tmp_path / 'checkpoints')
    app = build_app(cfg)
    assert app is not None
