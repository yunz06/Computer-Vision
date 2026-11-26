from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.integration import validate_input_image, run_integration
from src.visualization import render_panels
from src.demo import format_status, build_report_text, save_result_files, build_app
import src.demo as demo_module


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
        'detection': {'score_threshold': 0.5, 'model': 'fasterrcnn_mobilenet_v3_large_fpn'},
        'evaluation': {'consistency_iou_threshold': 0.5},
        'paths': {'demo': str(tmp_path), 'results': str(tmp_path / 'results'), 'checkpoints': str(tmp_path / 'checkpoints')},
    }


def _result(tmp_path):
    cfg = _cfg(tmp_path)
    image = Image.fromarray(np.full((100, 200, 3), 120, dtype=np.uint8))
    return image, run_integration(image, DummySeg(), DummyDet(), torch.device('cpu'), cfg)


def test_demo_report_and_file_export(tmp_path):
    image, result = _result(tmp_path)
    status = format_status(result)
    report = build_report_text(result)
    assert 'Confidence' in status and 'IoU bbox Detection' in status
    assert 'BÁO CÁO PHÂN TÍCH HÌNH ẢNH' in report
    assert 'không phải pixel IoU' in report
    assert np.asarray(image).shape == (100, 200, 3)

    saved = save_result_files(result, tmp_path / 'exported')
    assert len(saved) == 7
    assert sum(p.endswith('.png') for p in saved) == 5
    pdfs = [p for p in saved if p.endswith('.pdf')]
    assert len(pdfs) == 1 and (tmp_path / 'exported' / Path(pdfs[0]).name).stat().st_size > 1000
    assert any(p.endswith('_report.txt') for p in saved)


def test_integration_standard_output_and_panels(tmp_path):
    _, result = _result(tmp_path)
    for key in (
        'original_image', 'boxes', 'labels', 'scores', 'prob_map',
        'binary_mask', 'mask_bbox', 'box_mask_iou', 'consistency_status', 'inference_time_ms',
    ):
        assert key in result
    assert result['prob_map'].shape == (100, 200)
    assert result['binary_mask'].shape == (100, 200)
    assert result['box_mask_iou'] is not None
    orig, det, mask, overlay, combined = render_panels(result)
    assert orig.size == (200, 100)
    assert det.size == mask.size == overlay.size == (200, 100)
    assert combined.size == (400, 200)


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


def test_gradio_app_builds(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(demo_module, '_load_integration_models', lambda _cfg: (DummySeg(), DummyDet(), torch.device('cpu'), {}))
    app = build_app(cfg)
    assert app is not None
