from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import KMeans
from .data import (
    find_pairs,
    read_rgb,
    read_mask,
    letterbox,
    unletterbox_mask,
    postprocess_mask,
    mask_to_boxes,
)
from .metrics import (
    binary_metrics,
    box_iou,
    detection_ap,
    detection_ap5095,
    segmentation_summary,
    top1_box_iou_summary,
)
from .utils import save_json


def _otsu_predict(img, valid):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = [otsu.astype(bool) & valid, (~otsu.astype(bool)) & valid]
    pred = min(candidates, key=lambda p: float(gray[p].mean()) if p.any() else float('inf'))
    kernel = np.ones((5, 5), np.uint8)
    m = (pred * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = (m > 0) & valid
    return postprocess_mask(m, largest_component=True, hole_filling=False)


def _kmeans_lab_predict(img, valid):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab[valid].reshape(-1, 3).astype(np.float32)
    if len(pixels) < 2:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    km = KMeans(n_clusters=2, random_state=42, n_init=5).fit(pixels)
    labels = np.full(img.shape[:2], -1, dtype=np.int32)
    labels[valid] = km.labels_
    lightness = [float(lab[:, :, 0][labels == i].mean()) if np.any(labels == i) else float('inf') for i in (0, 1)]
    pred = (labels == int(np.argmin(lightness))).astype(np.uint8)
    return pred


def _watershed_predict(img, valid):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = [otsu.astype(bool) & valid, (~otsu.astype(bool)) & valid]
    fg_init = min(candidates, key=lambda p: float(gray[p].mean()) if p.any() else float('inf')).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(fg_init, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    if max_d <= 0:
        return postprocess_mask(opening & valid, largest_component=True)
    sure_fg = (dist >= 0.4 * max_d).astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    ws_img = img.copy()
    cv2.watershed(ws_img, markers)
    pred = ((markers > 1) & valid).astype(np.uint8)
    return postprocess_mask(pred, largest_component=True, hole_filling=True)


def run(
    image_dir,
    mask_dir,
    out_dir,
    limit=0,
    size=256,
    records=None,
    mask_threshold=127,
    min_area_ratio=0.001,
    boundary_tolerance_px=2,
):
    pairs = records if records is not None else find_pairs(image_dir, mask_dir)[0]
    pairs = pairs[:limit] if limit else pairs

    results = []
    otsu_rows, km_rows, ws_rows = [], [], []
    center_preds, gt_boxes_list, top_ious = [], [], []

    for key, ip, mp in pairs:
        image = read_rgb(ip)
        gt_orig = read_mask(mp, threshold=mask_threshold)
        img_lb, _, meta = letterbox(image, size, gt_orig)
        valid = np.zeros((size, size), dtype=bool)
        l, t = meta['left'], meta['top']
        valid[t:t + meta['new_height'], l:l + meta['new_width']] = True

        otsu_orig = unletterbox_mask(_otsu_predict(img_lb, valid), meta)
        km_orig = unletterbox_mask(_kmeans_lab_predict(img_lb, valid), meta)
        ws_orig = unletterbox_mask(_watershed_predict(img_lb, valid), meta)

        m_otsu = binary_metrics(otsu_orig, gt_orig, boundary_tolerance_px=boundary_tolerance_px)
        m_km = binary_metrics(km_orig, gt_orig, boundary_tolerance_px=boundary_tolerance_px)
        m_ws = binary_metrics(ws_orig, gt_orig, boundary_tolerance_px=boundary_tolerance_px)
        otsu_rows.append(m_otsu)
        km_rows.append(m_km)
        ws_rows.append(m_ws)

        h, w = gt_orig.shape
        center_box = np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75], dtype=np.float32)
        gt_boxes = mask_to_boxes(gt_orig, min_area_ratio=min_area_ratio, fallback_largest=True)
        if len(gt_boxes) > 0:
            center_iou = max(box_iou(center_box, gb) for gb in gt_boxes)
            center_preds.append({'boxes': center_box[None, :], 'scores': np.array([1.0], dtype=np.float32)})
            gt_boxes_list.append(gt_boxes)
            top_ious.append(center_iou)
        else:
            center_iou = 0.0

        results.append({
            'key': key,
            'otsu_dice': m_otsu['dice'],
            'otsu_iou': m_otsu['iou'],
            'kmeans_lab_dice': m_km['dice'],
            'kmeans_lab_iou': m_km['iou'],
            'watershed_dice': m_ws['dice'],
            'watershed_iou': m_ws['iou'],
            'center_box_iou': float(center_iou),
        })

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_json(results, Path(out_dir) / 'baseline_results.json')

    summary = {
        'count': len(results),
        'otsu_dice': float(np.mean([r['otsu_dice'] for r in results])) if results else 0.0,
        'kmeans_lab_dice': float(np.mean([r['kmeans_lab_dice'] for r in results])) if results else 0.0,
        'watershed_dice': float(np.mean([r['watershed_dice'] for r in results])) if results else 0.0,
        'center_box_iou': float(np.mean([r['center_box_iou'] for r in results])) if results else 0.0,
        'otsu_morphology': segmentation_summary(otsu_rows),
        'kmeans_lab': segmentation_summary(km_rows),
        'watershed': segmentation_summary(ws_rows),
        'center_box': {
            'count': len(gt_boxes_list),
            'ap50': detection_ap(center_preds, gt_boxes_list, 0.5) if gt_boxes_list else 0.0,
            'ap5095': detection_ap5095(center_preds, gt_boxes_list) if gt_boxes_list else 0.0,
            **top1_box_iou_summary(top_ious),
        },
    }
    save_json(summary, Path(out_dir) / 'baseline_summary.json')
    return results
