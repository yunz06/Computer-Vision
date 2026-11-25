import cv2
import numpy as np


def _extract_boundary(binary_u8):
    """Lấy đường biên của ảnh nhị phân."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary_u8, kernel, iterations=1)
    return (binary_u8 ^ eroded).astype(np.uint8)


def boundary_f1(pred, target, tolerance_px=2, eps=1e-7, max_side=1536):
    """
    Boundary F1 trên ảnh thu nhỏ đồng tỷ lệ để giảm RAM.
    Dice và IoU vẫn được tính trên ảnh gốc trong binary_metrics.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    if pred.shape != target.shape:
        raise ValueError(f"Kích thước mask không khớp: {pred.shape} != {target.shape}")
    if pred.ndim != 2:
        raise ValueError("boundary_f1 chỉ hỗ trợ mask 2 chiều.")

    h, w = pred.shape
    scale = min(1.0, float(max_side) / max(h, w))

    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        p_small = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        t_small = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        tol = max(1, int(round(float(tolerance_px) * scale)))
    else:
        p_small, t_small = pred, target
        tol = max(1, int(tolerance_px))

    p = np.asarray(p_small > 0, dtype=np.uint8)
    t = np.asarray(t_small > 0, dtype=np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    bp = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
    bt = cv2.morphologyEx(t, cv2.MORPH_GRADIENT, kernel)

    n_p = int(cv2.countNonZero(bp))
    n_t = int(cv2.countNonZero(bt))

    if n_p == 0 and n_t == 0:
        return 1.0
    if n_p == 0 or n_t == 0:
        return 0.0

    ksize = 2 * tol + 1
    match_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated_t = cv2.dilate(bt, match_kernel)
    dilated_p = cv2.dilate(bp, match_kernel)

    matched_p = cv2.bitwise_and(bp, dilated_t)
    matched_t = cv2.bitwise_and(bt, dilated_p)

    precision = float(cv2.countNonZero(matched_p)) / n_p
    recall = float(cv2.countNonZero(matched_t)) / n_t
    return float(2.0 * precision * recall / (precision + recall + eps))


def binary_metrics(pred, target, boundary_tolerance_px=2, eps=1e-7):
    """Dice/IoU tính trên ảnh gốc theo từng khối; Boundary F1 tính ở ảnh thu nhỏ."""
    p = np.asarray(pred)
    t = np.asarray(target)

    if p.shape != t.shape:
        raise ValueError(f"Kích thước mask không khớp: {p.shape} != {t.shape}")
    if p.ndim != 2:
        raise ValueError("binary_metrics chỉ hỗ trợ mask 2 chiều.")

    intersection = union = pred_sum = target_sum = 0
    rows_per_chunk = max(1, min(256, p.shape[0]))

    for y in range(0, p.shape[0], rows_per_chunk):
        pc = p[y:y + rows_per_chunk] > 0
        tc = t[y:y + rows_per_chunk] > 0
        intersection += int(np.count_nonzero(pc & tc))
        union += int(np.count_nonzero(pc | tc))
        pred_sum += int(np.count_nonzero(pc))
        target_sum += int(np.count_nonzero(tc))

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    bf1 = boundary_f1(p, t, tolerance_px=boundary_tolerance_px, eps=eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "thresholded_jaccard_065": float(iou if iou >= 0.65 else 0.0),
        "boundary_f1": float(bf1),
    }


def box_iou(a, b, eps=1e-7):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ab - inter + eps)


def detection_ap(predictions, targets, iou_threshold=0.5):
    """AP 101 điểm nội suy, một lớp đối tượng."""
    preds = []
    total_gt = 0
    gt_arrays = [
        target.detach().cpu().numpy()
        if hasattr(target, "detach") else np.asarray(target)
        for target in targets
    ]

    for image_idx, (pred, gt) in enumerate(zip(predictions, gt_arrays)):
        total_gt += len(gt)
        boxes = (
            pred["boxes"].detach().cpu().numpy()
            if hasattr(pred["boxes"], "detach") else np.asarray(pred["boxes"])
        )
        scores = (
            pred["scores"].detach().cpu().numpy()
            if hasattr(pred["scores"], "detach") else np.asarray(pred["scores"])
        )
        for box, score in zip(boxes, scores):
            preds.append((float(score), image_idx, box))

    if total_gt == 0:
        return 0.0

    preds.sort(key=lambda item: item[0], reverse=True)
    matched = set()
    tp, fp = [], []

    for _, image_idx, box in preds:
        gt = gt_arrays[image_idx]
        best_idx = -1
        best_iou = 0.0
        for j, gt_box in enumerate(gt):
            if (image_idx, j) in matched:
                continue
            value = box_iou(box, gt_box)
            if value > best_iou:
                best_iou, best_idx = value, j

        if best_idx >= 0 and best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched.add((image_idx, best_idx))
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = np.linspace(0.0, 1.0, 101)
    values = [
        float(precision_envelope[recall >= r].max())
        if np.any(recall >= r) else 0.0
        for r in samples
    ]
    return float(np.mean(values))


def detection_ap5095(predictions, targets):
    thresholds = np.arange(0.5, 0.96, 0.05)
    return float(np.mean([
        detection_ap(predictions, targets, float(threshold))
        for threshold in thresholds
    ]))


def detection_precision_recall(predictions, targets, score_threshold=0.5, iou_threshold=0.5):
    tp = fp = total_gt = 0

    for pred, target in zip(predictions, targets):
        boxes = (
            pred["boxes"].detach().cpu().numpy()
            if hasattr(pred["boxes"], "detach") else np.asarray(pred["boxes"])
        )
        scores = (
            pred["scores"].detach().cpu().numpy()
            if hasattr(pred["scores"], "detach") else np.asarray(pred["scores"])
        )
        gt = (
            target.detach().cpu().numpy()
            if hasattr(target, "detach") else np.asarray(target)
        )

        total_gt += len(gt)
        order = np.argsort(-scores)
        used = set()

        for idx in order:
            if scores[idx] < score_threshold:
                continue
            best_idx = -1
            best_iou = 0.0
            for j, gt_box in enumerate(gt):
                if j in used:
                    continue
                value = box_iou(boxes[idx], gt_box)
                if value > best_iou:
                    best_iou, best_idx = value, j

            if best_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                used.add(best_idx)
            else:
                fp += 1

    fn = max(0, total_gt - tp)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    result = {
        "precision": precision,
        "recall": recall,
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
    }

    if abs(score_threshold - 0.5) < 1e-6 and abs(iou_threshold - 0.5) < 1e-6:
        result["precision_at_score_0_5_iou_0_5"] = precision
        result["recall_at_score_0_5_iou_0_5"] = recall
    return result


def top1_box_iou_summary(top_ious):
    if not top_ious:
        return {
            "top1_box_iou": 0.0,
            "top1_box_iou_std": 0.0,
            "top1_iou_at_0_5": 0.0,
            "top1_iou_at_0_75": 0.0,
            "top1_iou_at_0_9": 0.0,
        }

    arr = np.asarray(top_ious, dtype=np.float64)
    return {
        "top1_box_iou": float(arr.mean()),
        "top1_box_iou_std": float(arr.std()),
        "top1_iou_at_0_5": float((arr >= 0.5).mean()),
        "top1_iou_at_0_75": float((arr >= 0.75).mean()),
        "top1_iou_at_0_9": float((arr >= 0.9).mean()),
    }


def segmentation_summary(rows):
    if not rows:
        return {"count": 0}

    keys = ("dice", "iou", "thresholded_jaccard_065", "boundary_f1")
    result = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values:
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"std_{key}"] = float(np.std(values))

    result["failure_rate_jaccard_below_0_65"] = float(
        np.mean([row["iou"] < 0.65 for row in rows])
    )
    result["failure_rate_dice_below_0_65"] = float(
        np.mean([row["dice"] < 0.65 for row in rows])
    )
    result["count"] = len(rows)
    return result
