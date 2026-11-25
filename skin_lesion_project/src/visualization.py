from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def render_detection_panel(result):
    canvas = result['original_image'].copy()
    boxes = result.get('boxes')
    scores = result.get('scores')
    if boxes is not None and len(boxes):
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            score = float(scores[idx]) if scores is not None and idx < len(scores) else 0.0
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 80), 2)
            label_txt = f'lesion {score:.2f}'
            ty = max(18, y1 - 6)
            cv2.putText(canvas, label_txt, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2, cv2.LINE_AA)
    return canvas


def render_mask_panel(result):
    orig = result['original_image']
    mask = result.get('binary_mask')
    if mask is None:
        return np.zeros(orig.shape[:2], dtype=np.uint8)
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def render_overlay_panel(result):
    overlay = result['original_image'].copy()
    mask = result.get('binary_mask')
    if mask is not None and np.any(mask):
        tint = overlay.copy()
        tint[mask > 0] = (235, 50, 50)
        overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
    mask_bbox = result.get('mask_bbox')
    if mask_bbox is not None:
        mx1, my1, mx2, my2 = [int(round(v)) for v in mask_bbox]
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (0, 200, 255), 2)
    boxes = result.get('boxes')
    if boxes is not None and len(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[0]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


def render_combined_panel(result):
    orig = result['original_image']
    det_img = render_detection_panel(result)
    mask_gray = render_mask_panel(result)
    mask_rgb = np.stack([mask_gray] * 3, axis=-1)
    overlay = render_overlay_panel(result)
    top_row = np.concatenate([orig, det_img], axis=1)
    bot_row = np.concatenate([mask_rgb, overlay], axis=1)
    return np.concatenate([top_row, bot_row], axis=0)


def render_panels(result):
    orig = Image.fromarray(result['original_image'])
    det_img = Image.fromarray(render_detection_panel(result))
    mask_img = Image.fromarray(render_mask_panel(result))
    overlay_img = Image.fromarray(render_overlay_panel(result))
    combined_img = Image.fromarray(render_combined_panel(result))
    return orig, det_img, mask_img, overlay_img, combined_img


def save_seg_qualitative_grid(records, out_path, k=3):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not records:
            return
        ordered = sorted(records, key=lambda r: r['dice'])
        n = len(ordered)
        picks = []
        for idx in range(min(k, n)):
            picks.append(('Worst', ordered[idx]))
        mid = n // 2
        picks.append(('Median', ordered[mid]))
        for idx in range(max(0, n - min(k, n)), n):
            picks.append(('Best', ordered[idx]))

        fig, axes = plt.subplots(len(picks), 3, figsize=(9.5, 2.8 * len(picks)))
        if len(picks) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_i, (tag, item) in enumerate(picks):
            img = item['image']
            gt = item['gt']
            pred = item['pred']
            ov = img.copy()
            ov[pred > 0] = (0.6 * ov[pred > 0] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
            axes[row_i, 0].imshow(img)
            axes[row_i, 0].set_title(f"{tag}: {item['key']}")
            axes[row_i, 1].imshow(gt * 255, cmap='gray')
            axes[row_i, 1].set_title('Ground Truth')
            axes[row_i, 2].imshow(ov)
            axes[row_i, 2].set_title(f"Pred Dice={item['dice']:.3f}, IoU={item['iou']:.3f}")
            for c in range(3):
                axes[row_i, c].axis('off')
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f'Không lưu được qualitative segmentation: {exc}')


def save_det_qualitative_grid(records, out_path, k=3):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not records:
            return
        ordered = sorted(records, key=lambda r: r['top1_iou'])
        n = len(ordered)
        picks = [('Worst', ordered[0]), ('Median', ordered[n // 2]), ('Best', ordered[-1])]
        if k > 1 and n >= 5:
            picks = [('Worst', ordered[0]), ('Low', ordered[1]), ('Median', ordered[n // 2]), ('Best', ordered[-1])]

        fig, axes = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 4.0))
        if len(picks) == 1:
            axes = [axes]
        for ax, (tag, item) in zip(axes, picks):
            vis = item['image'].copy()
            for gb in item['gt_boxes']:
                x1, y1, x2, y2 = [int(round(v)) for v in gb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            for pb, sc in zip(item['pred_boxes'], item['pred_scores']):
                x1, y1, x2, y2 = [int(round(v)) for v in pb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f'{float(sc):.2f}', (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ax.imshow(vis)
            ax.set_title(f"{tag}: {item['key']}\nTop1 IoU={item['top1_iou']:.3f}")
            ax.axis('off')
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f'Không lưu được qualitative detection: {exc}')
