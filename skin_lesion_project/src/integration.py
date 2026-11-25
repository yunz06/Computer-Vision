import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .models import build_segmentation_model, build_detector
from .data import (
    IMAGE_EXTS,
    letterbox,
    unletterbox_prob_map,
    postprocess_mask,
    mask_to_box,
)
from .metrics import box_iou
from .utils import device


def validate_input_image(image):
    if image is None:
        raise ValueError('Vui lòng chọn ảnh đầu vào (JPG/PNG).')
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists() or not p.is_file():
            raise ValueError(f'Không tìm thấy file ảnh: {p}')
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f'Định dạng không hỗ trợ ({p.suffix}). Chỉ chấp nhận JPG/PNG.')
        try:
            with Image.open(p) as im:
                im.load()
                rgb = np.asarray(im.convert('RGB')).copy()
        except Exception as exc:
            raise ValueError(f'Không đọc được file ảnh: {exc}') from exc
    elif isinstance(image, Image.Image):
        try:
            rgb = np.asarray(image.convert('RGB')).copy()
        except Exception as exc:
            raise ValueError(f'Ảnh không hợp lệ: {exc}') from exc
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1).astype(np.uint8)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            rgb = np.asarray(image[:, :, :3], dtype=np.uint8).copy()
        else:
            raise ValueError(f'Số kênh ảnh không hợp lệ: shape={image.shape}')
    else:
        raise ValueError('Kiểu dữ liệu ảnh đầu vào không hợp lệ.')

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Ảnh đầu vào phải có 3 kênh RGB.')
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(f'Kích thước ảnh quá nhỏ ({w}x{h}). Yêu cầu tối thiểu 16x16.')
    return rgb


def load_models(cfg):
    dev = device()
    ckpt_dir = Path(cfg['paths']['checkpoints'])
    seg_path = ckpt_dir / 'unet_best.pt'
    det_path = ckpt_dir / 'fasterrcnn_best.pt'
    if not det_path.exists():
        det_path = ckpt_dir / 'fasterrcnn_last.pt'

    seg = det = None
    load_errors = {}
    if seg_path.exists():
        try:
            state = torch.load(seg_path, map_location=dev, weights_only=False)
            arch = state.get('model_name', cfg.get('segmentation', {}).get('model', 'unet'))
            seg = build_segmentation_model(arch, pretrained=False).to(dev)
            seg.load_state_dict(state['model'])
            seg.eval()
        except Exception as exc:
            load_errors['segmentation'] = str(exc)

    if det_path.exists():
        try:
            state = torch.load(det_path, map_location=dev, weights_only=False)
            arch = state.get('model_name', cfg['detection']['model'])
            det_size = state.get('image_size', cfg['detection_size'])
            det_cfg = cfg.get('detection', {})
            det = build_detector(
                False,
                arch,
                det_size,
                trainable_backbone_layers=state.get('trainable_backbone_layers'),
                box_nms_thresh=det_cfg.get('nms_iou_threshold', 0.5),
                box_detections_per_img=det_cfg.get('max_detections', 5),
            ).to(dev)
            det.load_state_dict(state['model'])
            det.eval()
        except Exception as exc:
            load_errors['detection'] = str(exc)

    return seg, det, dev, load_errors


def _map_boxes_to_original(raw_boxes, raw_scores, raw_labels, meta, score_threshold):
    scale, left, top = meta['scale'], meta['left'], meta['top']
    ow, oh = meta['original_width'], meta['original_height']
    boxes, scores, labels = [], [], []
    for idx in range(len(raw_boxes)):
        sc = float(raw_scores[idx])
        if sc < score_threshold:
            continue
        b = raw_boxes[idx]
        x1 = float(np.clip((b[0] - left) / scale, 0.0, ow))
        y1 = float(np.clip((b[1] - top) / scale, 0.0, oh))
        x2 = float(np.clip((b[2] - left) / scale, 0.0, ow))
        y2 = float(np.clip((b[3] - top) / scale, 0.0, oh))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            scores.append(sc)
            labels.append(int(raw_labels[idx]) if raw_labels is not None and idx < len(raw_labels) else 1)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    order = np.argsort(-np.asarray(scores))
    return (
        np.asarray(boxes, dtype=np.float32)[order],
        np.asarray(scores, dtype=np.float32)[order],
        np.asarray(labels, dtype=np.int64)[order],
    )


def run_integration(image, seg, det, dev, cfg):
    t0 = time.perf_counter()
    original = validate_input_image(image)
    errors = {}

    seg_cfg = cfg.get('segmentation', {})
    det_cfg = cfg.get('detection', {})
    eval_cfg = cfg.get('evaluation', {})
    seg_thresh = float(seg_cfg.get('threshold', 0.5))
    det_thresh = float(det_cfg.get('score_threshold', 0.5))
    iou_thresh = float(eval_cfg.get('consistency_iou_threshold', 0.5))

    prob_map = None
    binary_mask = None
    mask_bbox = None

    if seg is None:
        errors['segmentation'] = 'Chưa nạp mô hình Segmentation.'
    else:
        try:
            lb, meta = letterbox(original, cfg['image_size'])
            tensor = torch.from_numpy(lb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            with torch.no_grad():
                logits = seg(tensor)
                prob_lb = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            prob_map = unletterbox_prob_map(prob_lb, meta)
            binary_mask = (prob_map >= seg_thresh).astype(np.uint8)
            if seg_cfg.get('largest_component', False) or seg_cfg.get('hole_filling', False):
                binary_mask = postprocess_mask(
                    binary_mask,
                    largest_component=seg_cfg.get('largest_component', False),
                    hole_filling=seg_cfg.get('hole_filling', False),
                )
            mask_bbox = mask_to_box(binary_mask)
        except Exception as exc:
            errors['segmentation'] = f'Lỗi nhánh Segmentation: {exc}'

    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    labels = np.zeros((0,), dtype=np.int64)

    if det is None:
        errors['detection'] = 'Chưa nạp mô hình Detection.'
    else:
        try:
            det_lb, det_meta = letterbox(original, cfg['detection_size'])
            tensor = torch.from_numpy(det_lb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            with torch.no_grad():
                output = det(tensor)[0]
            raw_boxes = output['boxes'].detach().cpu().numpy()
            raw_scores = output['scores'].detach().cpu().numpy()
            raw_labels = output['labels'].detach().cpu().numpy() if 'labels' in output else None
            boxes, scores, labels = _map_boxes_to_original(raw_boxes, raw_scores, raw_labels, det_meta, det_thresh)
        except Exception as exc:
            errors['detection'] = f'Lỗi nhánh Detection: {exc}'

    box_mask_iou = None
    has_det = len(boxes) > 0
    has_mask = mask_bbox is not None

    if 'detection' in errors and 'segmentation' in errors:
        consistency_status = 'Thiếu cả hai nhánh'
    elif 'detection' in errors:
        consistency_status = 'Bất đồng / thiếu nhánh Detection'
    elif 'segmentation' in errors:
        consistency_status = 'Bất đồng / thiếu nhánh Segmentation'
    elif not has_det and not has_mask:
        consistency_status = 'Bất đồng (cả hai nhánh rỗng)'
    elif not has_det:
        consistency_status = 'Bất đồng (Detection rỗng)'
    elif not has_mask:
        consistency_status = 'Bất đồng (Mask rỗng)'
    else:
        box_mask_iou = float(box_iou(boxes[0], mask_bbox))
        consistency_status = 'Nhất quán' if box_mask_iou >= iou_thresh else 'Bất đồng'

    inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        'original_image': original,
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'prob_map': prob_map,
        'binary_mask': binary_mask,
        'mask_bbox': mask_bbox,
        'box_mask_iou': box_mask_iou,
        'consistency_status': consistency_status,
        'inference_time_ms': inference_time_ms,
        'errors': errors,
    }
