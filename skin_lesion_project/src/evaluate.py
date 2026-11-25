from pathlib import Path
import numpy as np
import torch
from .prepare import load_records
from .data import (
    read_rgb,
    read_mask,
    load_or_cache_letterbox,
    unletterbox_prob_map,
    postprocess_mask,
    mask_to_boxes,
)
from .models import build_segmentation_model, build_detector
from .metrics import (
    binary_metrics,
    box_iou,
    detection_ap,
    detection_ap5095,
    detection_precision_recall,
    segmentation_summary,
    top1_box_iou_summary,
)
from .integration import _map_boxes_to_original
from .visualization import save_seg_qualitative_grid, save_det_qualitative_grid
from .utils import device, save_json


def _load_seg_model(cfg, checkpoint=None):
    ck = Path(checkpoint) if checkpoint else Path(cfg['paths']['checkpoints']) / 'unet_best.pt'
    if not ck.exists():
        raise FileNotFoundError(f'Chưa có checkpoint segmentation: {ck}')
    dev = device()
    state = torch.load(ck, map_location=dev, weights_only=False)
    arch = state.get('model_name', cfg['segmentation'].get('model', 'unet'))
    size = state.get('image_size', cfg['image_size'])
    model = build_segmentation_model(arch, pretrained=False).to(dev)
    model.load_state_dict(state['model'])
    model.eval()
    return model, dev, size, state


def _load_det_model(cfg, checkpoint=None):
    if checkpoint:
        ck = Path(checkpoint)
    else:
        ck = Path(cfg['paths']['checkpoints']) / 'fasterrcnn_best.pt'
        if not ck.exists():
            ck = Path(cfg['paths']['checkpoints']) / 'fasterrcnn_last.pt'
    if not ck.exists():
        raise FileNotFoundError(f'Chưa có checkpoint detection: {ck}')
    dev = device()
    state = torch.load(ck, map_location=dev, weights_only=False)
    arch = state.get('model_name', cfg['detection']['model'])
    size = state.get('image_size', cfg['detection_size'])
    det_cfg = cfg['detection']
    model = build_detector(
        False,
        arch,
        size,
        trainable_backbone_layers=state.get('trainable_backbone_layers'),
        box_nms_thresh=det_cfg.get('nms_iou_threshold', 0.5),
        box_detections_per_img=det_cfg.get('max_detections', 5),
    ).to(dev)
    model.load_state_dict(state['model'])
    model.eval()
    return model, dev, size, state


def _predict_seg_prob_maps(model, dev, size, records, mask_threshold=127, cache_dir=None):
    items = []
    with torch.no_grad():
        for key, ip, mp in records:
            gt_orig = read_mask(mp, threshold=mask_threshold)
            lb, meta = load_or_cache_letterbox(
                key, ip, None, size, mask_threshold=mask_threshold, cache_dir=cache_dir
            )
            x = torch.from_numpy(lb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            logits = model(x)
            prob_lb = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prob_orig = unletterbox_prob_map(prob_lb, meta)
            items.append((key, ip, gt_orig, prob_orig))
    return items



def eval_seg(cfg, split='test', checkpoint=None, threshold=None,
             largest_component=None, hole_filling=None,
             save_qualitative=True):
    import gc

    records = load_records(
        Path(cfg['paths']['splits']) / 'split.json', split
    )
    if not records:
        raise RuntimeError(f'Split "{split}" rỗng.')

    model, dev, size, state = _load_seg_model(cfg, checkpoint)
    seg_cfg = cfg['segmentation']
    data_cfg = cfg.get('data', {})
    eval_cfg = cfg.get('evaluation', {})

    selection_path = Path(cfg['paths']['results']) / 'val_selection.json'
    selection = {}

    if threshold is None and selection_path.exists():
        try:
            import json
            selection = (
                json.loads(selection_path.read_text(encoding='utf-8'))
                .get('segmentation_best') or {}
            )
        except (OSError, ValueError):
            selection = {}

    thresh = float(
        threshold if threshold is not None
        else selection.get('threshold', seg_cfg.get('threshold', 0.5))
    )
    use_lc = (
        bool(largest_component) if largest_component is not None
        else bool(selection.get(
            'largest_component',
            seg_cfg.get('largest_component', False)
        ))
    )
    use_hf = (
        bool(hole_filling) if hole_filling is not None
        else bool(selection.get(
            'hole_filling',
            seg_cfg.get('hole_filling', False)
        ))
    )

    mask_thresh = data_cfg.get('mask_threshold', 127)
    tol_px = eval_cfg.get('boundary_tolerance_px', 2)
    cache_dir = cfg['paths'].get('processed')

    rows = []
    # Chỉ lưu thông tin đường dẫn và điểm số, không giữ mask ảnh trong RAM.
    candidates = []

    for idx, (key, ip, mp) in enumerate(records):
        gt_orig = read_mask(mp, threshold=mask_thresh)

        lb, meta = load_or_cache_letterbox(
            key, ip, None, size,
            mask_threshold=mask_thresh,
            cache_dir=cache_dir
        )

        x = (
            torch.from_numpy(lb.copy())
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(dev) / 255.0
        )

        with torch.no_grad():
            logits = model(x)
            prob_lb = torch.sigmoid(logits)[0, 0].cpu().numpy()

        del x, logits, lb

        prob_orig = unletterbox_prob_map(prob_lb, meta)
        del prob_lb

        pred = (prob_orig >= thresh).astype(np.uint8)

        if use_lc or use_hf:
            pred = postprocess_mask(
                pred,
                largest_component=use_lc,
                hole_filling=use_hf
            )

        m = binary_metrics(
            pred, gt_orig,
            boundary_tolerance_px=tol_px
        )
        m['key'] = key
        rows.append(m)

        candidates.append({
            'index': idx,
            'key': key,
            'ip': ip,
            'mp': mp,
            'dice': m['dice'],
            'iou': m['iou'],
        })

        # Giải phóng các mảng lớn trước khi đọc ảnh tiếp theo.
        del gt_orig, prob_orig, pred
        gc.collect()

        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        print(
            f'\r[eval-seg] {idx + 1}/{len(records)} '
            f'Dice={m["dice"]:.4f} IoU={m["iou"]:.4f}',
            end='',
            flush=True
        )

    print()

    result = segmentation_summary(rows)
    result.update({
        'split': split,
        'model_name': state.get('model_name', 'unet'),
        'image_size': size,
        'threshold': thresh,
        'largest_component': use_lc,
        'hole_filling': use_hf,
        'boundary_tolerance_px': tol_px,
    })

    out_name = (
        'seg_test_metrics.json'
        if split == 'test'
        else f'seg_{split}_metrics.json'
    )

    save_json(
        result,
        Path(cfg['paths']['results']) / out_name
    )
    save_json(
        rows,
        Path(cfg['paths']['results']) / f'seg_{split}_per_image.json'
    )

    # Chỉ nạp lại một số ảnh cần thiết để tạo hình minh họa.
    if save_qualitative and candidates:
        k_qual = eval_cfg.get('qualitative_count', 3)
        ordered = sorted(candidates, key=lambda r: r['dice'])
        n = len(ordered)

        selected_idx = (
            set(range(min(k_qual, n)))
            | {n // 2}
            | set(range(max(0, n - min(k_qual, n)), n))
        )

        qualitative_records = []

        for idx in sorted(selected_idx):
            item = ordered[idx]
            gt_orig = read_mask(
                item['mp'], threshold=mask_thresh
            )

            lb, meta = load_or_cache_letterbox(
                item['key'], item['ip'], None, size,
                mask_threshold=mask_thresh,
                cache_dir=cache_dir
            )

            x = (
                torch.from_numpy(lb.copy())
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(dev) / 255.0
            )

            with torch.no_grad():
                logits = model(x)
                prob_lb = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob_orig = unletterbox_prob_map(prob_lb, meta)
            pred = (prob_orig >= thresh).astype(np.uint8)

            if use_lc or use_hf:
                pred = postprocess_mask(
                    pred,
                    largest_component=use_lc,
                    hole_filling=use_hf
                )

            qualitative_records.append({
                'key': item['key'],
                'ip': item['ip'],
                'image': read_rgb(item['ip']),
                'gt': gt_orig,
                'pred': pred,
                'dice': item['dice'],
                'iou': item['iou'],
            })

            del gt_orig, prob_orig, pred, x, logits, lb
            gc.collect()

        save_seg_qualitative_grid(
            qualitative_records,
            Path(cfg['paths']['figures']) / f'seg_{split}_qualitative.png',
            k=k_qual,
        )

    return result


def eval_det(cfg, split='test', checkpoint=None, score_threshold=None, save_qualitative=True):
    records = load_records(Path(cfg['paths']['splits']) / 'split.json', split)
    if not records:
        raise RuntimeError(f'Split "{split}" rỗng.')
    model, dev, size, state = _load_det_model(cfg, checkpoint)
    det_cfg = cfg['detection']
    data_cfg = cfg.get('data', {})
    eval_cfg = cfg.get('evaluation', {})

    selection_path = Path(cfg['paths']['results']) / 'val_selection.json'
    selected_score = None
    if score_threshold is None and selection_path.exists():
        try:
            import json
            selected_score = (json.loads(selection_path.read_text(encoding='utf-8')).get('detection_best') or {}).get('score_threshold')
        except (OSError, ValueError):
            selected_score = None
    score_thresh = float(score_threshold) if score_threshold is not None else float(
        selected_score if selected_score is not None else det_cfg.get('score_threshold', 0.5)
    )
    mask_thresh = data_cfg.get('mask_threshold', 127)
    min_area = data_cfg.get('min_box_area_ratio', 0.001)
    multi_comp = data_cfg.get('multi_component_boxes', True)

    preds, gts, top_ious = [], [], []
    per_image = []
    qualitative_records = []

    cache_dir = cfg['paths'].get('processed')
    with torch.no_grad():
        for key, ip, mp in records:
            gt_mask = read_mask(mp, threshold=mask_thresh)
            gt_boxes = mask_to_boxes(gt_mask, min_area_ratio=min_area, fallback_largest=True)
            if len(gt_boxes) == 0:
                continue
            if not multi_comp and len(gt_boxes) > 1:
                gt_boxes = gt_boxes[:1]

            lb, meta = load_or_cache_letterbox(key, ip, None, size, mask_threshold=mask_thresh, cache_dir=cache_dir)
            x = torch.from_numpy(lb.copy()).permute(2, 0, 1).float().to(dev) / 255.0
            out = model([x])[0]
            raw_boxes = out['boxes'].detach().cpu().numpy()
            raw_scores = out['scores'].detach().cpu().numpy()
            raw_labels = out['labels'].detach().cpu().numpy() if 'labels' in out else None
            boxes_arr, scores_arr, _ = _map_boxes_to_original(raw_boxes, raw_scores, raw_labels, meta, score_threshold=0.0)

            preds.append({'boxes': boxes_arr, 'scores': scores_arr})
            gts.append(gt_boxes)
            top1_iou = max(box_iou(boxes_arr[0], gb) for gb in gt_boxes) if len(boxes_arr) else 0.0
            top_ious.append(top1_iou)
            per_image.append({
                'key': key,
                'top1_box_iou': float(top1_iou),
                'top1_score': float(scores_arr[0]) if len(scores_arr) else 0.0,
                'num_preds_above_thresh': int(np.sum(scores_arr >= score_thresh)) if len(scores_arr) else 0,
            })
            qualitative_records.append({
                'key': key,
                'ip': ip,
                'gt_boxes': gt_boxes,
                'pred_boxes': boxes_arr[scores_arr >= score_thresh] if len(boxes_arr) else boxes_arr,
                'pred_scores': scores_arr[scores_arr >= score_thresh] if len(scores_arr) else scores_arr,
                'top1_iou': float(top1_iou),
            })

    result = {
        'split': split,
        'model_name': state.get('model_name', det_cfg['model']),
        'image_size': size,
        'count': len(gts),
        'ap50': detection_ap(preds, gts, 0.5),
        'ap5095': detection_ap5095(preds, gts),
        **top1_box_iou_summary(top_ious),
        **detection_precision_recall(preds, gts, score_thresh, 0.5),
    }
    out_name = 'det_test_metrics.json' if split == 'test' else f'det_{split}_metrics.json'
    save_json(result, Path(cfg['paths']['results']) / out_name)
    save_json(per_image, Path(cfg['paths']['results']) / f'det_{split}_per_image.json')
    if save_qualitative and qualitative_records:
        k_qual = eval_cfg.get('qualitative_count', 3)
        ordered = sorted(qualitative_records, key=lambda r: r['top1_iou'])
        n = len(ordered)
        selected_idx = {0, n // 2, n - 1}
        if k_qual > 1 and n >= 5:
            selected_idx.add(1)
        for idx in selected_idx:
            ordered[idx]['image'] = read_rgb(ordered[idx]['ip'])
        save_det_qualitative_grid(
            [ordered[idx] for idx in sorted(selected_idx)],
            Path(cfg['paths']['figures']) / f'det_{split}_qualitative.png',
            k=k_qual,
        )
    return result


def eval_unet_box(cfg, split='test', checkpoint=None):
    # Experiment E4: lấy bounding box từ mask dự đoán của U-Net để so sánh với Faster R-CNN.
    records = load_records(Path(cfg['paths']['splits']) / 'split.json', split)
    if not records:
        raise RuntimeError(f'Split "{split}" rỗng.')
    model, dev, size, state = _load_seg_model(cfg, checkpoint)
    seg_cfg = cfg['segmentation']
    data_cfg = cfg.get('data', {})
    selection_path = Path(cfg['paths']['results']) / 'val_selection.json'
    selected_seg = {}
    if selection_path.exists():
        try:
            import json
            selected_seg = json.loads(selection_path.read_text(encoding='utf-8')).get('segmentation_best') or {}
        except (OSError, ValueError):
            selected_seg = {}
    thresh = float(selected_seg.get('threshold', seg_cfg.get('threshold', 0.5)))
    use_lc = bool(selected_seg.get('largest_component', seg_cfg.get('largest_component', False)))
    use_hf = bool(selected_seg.get('hole_filling', seg_cfg.get('hole_filling', False)))
    mask_thresh = data_cfg.get('mask_threshold', 127)
    min_area = data_cfg.get('min_box_area_ratio', 0.001)

    items = _predict_seg_prob_maps(
        model, dev, size, records, mask_threshold=mask_thresh, cache_dir=cfg['paths'].get('processed')
    )
    preds, gts, top_ious = [], [], []
    for _, _, gt_orig, prob_orig in items:
        gt_boxes = mask_to_boxes(gt_orig, min_area_ratio=min_area, fallback_largest=True)
        if len(gt_boxes) == 0:
            continue
        pred_mask = (prob_orig >= thresh).astype(np.uint8)
        if use_lc or use_hf:
            pred_mask = postprocess_mask(pred_mask, largest_component=use_lc, hole_filling=use_hf)
        pred_boxes = mask_to_boxes(pred_mask, min_area_ratio=min_area, fallback_largest=True)
        scores = []
        for bx in pred_boxes:
            x1, y1, x2, y2 = map(int, bx)
            patch = prob_orig[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
            scores.append(float(patch.mean()) if patch.size else 0.0)
        scores_arr = np.asarray(scores, dtype=np.float32)
        preds.append({'boxes': pred_boxes, 'scores': scores_arr})
        gts.append(gt_boxes)
        top_ious.append(max(box_iou(pred_boxes[0], gb) for gb in gt_boxes) if len(pred_boxes) else 0.0)

    result = {
        'split': split,
        'source': 'unet_mask_box',
        'count': len(gts),
        'ap50': detection_ap(preds, gts, 0.5) if gts else 0.0,
        'ap5095': detection_ap5095(preds, gts) if gts else 0.0,
        **top1_box_iou_summary(top_ious),
        **(detection_precision_recall(preds, gts, 0.5, 0.5) if gts else {}),
    }
    save_json(result, Path(cfg['paths']['results']) / f'unet_box_{split}_metrics.json')
    return result


def select_on_validation(cfg, limit=0):
    # D6: Quét threshold và hậu xử lý trên Validation ONLY (không đụng Test).
    records = load_records(Path(cfg['paths']['splits']) / 'split.json', 'val')
    if limit and limit > 0:
        records = records[:limit]
    if not records:
        raise RuntimeError('Validation split rỗng.')

    data_cfg = cfg.get('data', {})
    eval_cfg = cfg.get('evaluation', {})
    mask_thresh = data_cfg.get('mask_threshold', 127)
    tol_px = eval_cfg.get('boundary_tolerance_px', 2)
    min_area = data_cfg.get('min_box_area_ratio', 0.001)
    selection = {'split': 'val', 'count': len(records)}

    seg_ckpt = Path(cfg['paths']['checkpoints']) / 'unet_best.pt'
    if seg_ckpt.exists():
        seg_model, dev, size, _ = _load_seg_model(cfg, seg_ckpt)
        items = _predict_seg_prob_maps(
            seg_model, dev, size, records, mask_threshold=mask_thresh, cache_dir=cfg['paths'].get('processed')
        )
        best_seg = None
        seg_sweep = []
        for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            for lc, hf in ((False, False), (True, False), (True, True)):
                rows = []
                for _, _, gt_orig, prob_orig in items:
                    p = (prob_orig >= th).astype(np.uint8)
                    if lc or hf:
                        p = postprocess_mask(p, largest_component=lc, hole_filling=hf)
                    rows.append(binary_metrics(p, gt_orig, boundary_tolerance_px=tol_px))
                s = segmentation_summary(rows)
                entry = {
                    'threshold': th,
                    'largest_component': lc,
                    'hole_filling': hf,
                    'mean_dice': s['mean_dice'],
                    'mean_iou': s['mean_iou'],
                    'mean_thresholded_jaccard_065': s['mean_thresholded_jaccard_065'],
                }
                seg_sweep.append(entry)
                if best_seg is None or entry['mean_dice'] > best_seg['mean_dice']:
                    best_seg = entry
        selection['segmentation_best'] = best_seg
        selection['segmentation_sweep'] = seg_sweep

    det_ckpt = Path(cfg['paths']['checkpoints']) / 'fasterrcnn_best.pt'
    if not det_ckpt.exists():
        det_ckpt = Path(cfg['paths']['checkpoints']) / 'fasterrcnn_last.pt'
    if det_ckpt.exists():
        det_model, dev, det_size, _ = _load_det_model(cfg, det_ckpt)
        cache_dir = cfg['paths'].get('processed')
        preds, gts = [], []
        with torch.no_grad():
            for key, ip, mp in records:
                gt_mask = read_mask(mp, threshold=mask_thresh)
                gt_boxes = mask_to_boxes(gt_mask, min_area_ratio=min_area, fallback_largest=True)
                if len(gt_boxes) == 0:
                    continue
                lb, meta = load_or_cache_letterbox(
                    key, ip, None, det_size, mask_threshold=mask_thresh, cache_dir=cache_dir
                )
                x = torch.from_numpy(lb.copy()).permute(2, 0, 1).float().to(dev) / 255.0
                out = det_model([x])[0]
                raw_boxes = out['boxes'].detach().cpu().numpy()
                raw_scores = out['scores'].detach().cpu().numpy()
                raw_labels = out['labels'].detach().cpu().numpy() if 'labels' in out else None
                boxes_orig, scores_arr, _ = _map_boxes_to_original(raw_boxes, raw_scores, raw_labels, meta, score_threshold=0.0)
                preds.append({'boxes': boxes_orig, 'scores': scores_arr})
                gts.append(gt_boxes)
        best_det = None
        det_sweep = []
        for sth in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pr = detection_precision_recall(preds, gts, sth, 0.5)
            p, r = pr['precision'], pr['recall']
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            entry = {'score_threshold': sth, 'precision': p, 'recall': r, 'f1': float(f1)}
            det_sweep.append(entry)
            if best_det is None or entry['f1'] > best_det['f1']:
                best_det = entry
        selection['detection_best'] = best_det
        selection['detection_sweep'] = det_sweep

    save_json(selection, Path(cfg['paths']['results']) / 'val_selection.json')
    return selection
