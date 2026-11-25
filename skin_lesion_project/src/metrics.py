import cv2
import numpy as np


def _extract_boundary(binary_u8):
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary_u8, kernel, iterations=1)
    return (binary_u8 ^ eroded).astype(np.uint8)


def boundary_f1(pred, target, tolerance_px=2, eps=1e-7):
    p = (np.asarray(pred) > 0).astype(np.uint8)
    t = (np.asarray(target) > 0).astype(np.uint8)
    p_any, t_any = bool(p.any()), bool(t.any())
    if not p_any and not t_any:
        return 1.0
    if p_any != t_any:
        return 0.0
    bp = _extract_boundary(p)
    bt = _extract_boundary(t)
    if not bp.any() and not bt.any():
        return 1.0
    if not bp.any() or not bt.any():
        return 0.0
    dt_t = cv2.distanceTransform((1 - bt).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((1 - bp).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((dt_t[bp > 0] <= tolerance_px).mean())
    rec = float((dt_p[bt > 0] <= tolerance_px).mean())
    return float((2.0 * prec * rec + eps) / (prec + rec + eps)) if (prec + rec) > 0 else 0.0


def binary_metrics(pred, target, boundary_tolerance_px=2, eps=1e-7):
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    ps = p.sum()
    ts = t.sum()
    dice = (2 * inter + eps) / (ps + ts + eps)
    iou = (inter + eps) / (union + eps)
    bf1 = boundary_f1(p, t, tolerance_px=boundary_tolerance_px, eps=eps)
    return {
        'dice': float(dice),
        'iou': float(iou),
        'thresholded_jaccard_065': float(iou if iou >= 0.65 else 0.0),
        'boundary_f1': float(bf1),
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
    # AP 101 điểm nội suy, một lớp đối tượng.
    preds = []
    total_gt = 0
    for image_idx, (p, t) in enumerate(zip(predictions, targets)):
        gt = t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
        total_gt += len(gt)
        boxes = p['boxes'].detach().cpu().numpy() if hasattr(p['boxes'], 'detach') else np.asarray(p['boxes'])
        scores = p['scores'].detach().cpu().numpy() if hasattr(p['scores'], 'detach') else np.asarray(p['scores'])
        for box, score in zip(boxes, scores):
            preds.append((float(score), image_idx, box))
    if total_gt == 0:
        return 0.0
    preds.sort(key=lambda z: z[0], reverse=True)
    matched = set()
    tp = []
    fp = []
    for _, i, box in preds:
        gt = targets[i].detach().cpu().numpy() if hasattr(targets[i], 'detach') else np.asarray(targets[i])
        best = -1
        best_iou = 0.0
        for j, g in enumerate(gt):
            if (i, j) in matched:
                continue
            value = box_iou(box, g)
            if value > best_iou:
                best_iou = value
                best = j
        if best >= 0 and best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched.add((i, best))
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
    values = [float(precision_envelope[recall >= r].max()) if np.any(recall >= r) else 0.0 for r in samples]
    return float(np.mean(values))


def detection_ap5095(predictions, targets):
    return float(np.mean([detection_ap(predictions, targets, float(t)) for t in np.arange(0.5, 0.96, 0.05)]))


def detection_precision_recall(predictions, targets, score_threshold=0.5, iou_threshold=0.5):
    tp = fp = 0
    total_gt = 0
    for pred, target in zip(predictions, targets):
        boxes = pred['boxes'].detach().cpu().numpy() if hasattr(pred['boxes'], 'detach') else np.asarray(pred['boxes'])
        scores = pred['scores'].detach().cpu().numpy() if hasattr(pred['scores'], 'detach') else np.asarray(pred['scores'])
        gt = target.detach().cpu().numpy() if hasattr(target, 'detach') else np.asarray(target)
        total_gt += len(gt)
        order = np.argsort(-scores)
        used = set()
        for idx in order:
            if scores[idx] < score_threshold:
                continue
            best = -1
            best_iou = 0.0
            for j, g in enumerate(gt):
                if j in used:
                    continue
                v = box_iou(boxes[idx], g)
                if v > best_iou:
                    best_iou = v
                    best = j
            if best >= 0 and best_iou >= iou_threshold:
                tp += 1
                used.add(best)
            else:
                fp += 1
    fn = max(0, total_gt - tp)
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    out = {
        'precision': prec,
        'recall': rec,
        'score_threshold': float(score_threshold),
        'iou_threshold': float(iou_threshold),
    }
    if abs(score_threshold - 0.5) < 1e-6 and abs(iou_threshold - 0.5) < 1e-6:
        out['precision_at_score_0_5_iou_0_5'] = prec
        out['recall_at_score_0_5_iou_0_5'] = rec
    return out


def top1_box_iou_summary(top_ious):
    if not top_ious:
        return {
            'top1_box_iou': 0.0,
            'top1_box_iou_std': 0.0,
            'top1_iou_at_0_5': 0.0,
            'top1_iou_at_0_75': 0.0,
            'top1_iou_at_0_9': 0.0,
        }
    arr = np.asarray(top_ious, dtype=np.float64)
    return {
        'top1_box_iou': float(arr.mean()),
        'top1_box_iou_std': float(arr.std()),
        'top1_iou_at_0_5': float((arr >= 0.5).mean()),
        'top1_iou_at_0_75': float((arr >= 0.75).mean()),
        'top1_iou_at_0_9': float((arr >= 0.9).mean()),
    }


def segmentation_summary(rows):
    if not rows:
        return {'count': 0}
    keys = ('dice', 'iou', 'thresholded_jaccard_065', 'boundary_f1')
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[f'mean_{k}'] = float(np.mean(vals))
            out[f'std_{k}'] = float(np.std(vals))
    out['failure_rate_jaccard_below_0_65'] = float(np.mean([r['iou'] < 0.65 for r in rows]))
    out['failure_rate_dice_below_0_65'] = float(np.mean([r['dice'] < 0.65 for r in rows]))
    out['count'] = len(rows)
    return out
