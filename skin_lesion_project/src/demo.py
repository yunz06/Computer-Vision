from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import cv2
from PIL import Image
from .models import UNet, build_detector
from .data import letterbox, mask_to_box
from .metrics import box_iou
from .utils import device


def load_models(cfg):
    dev = device()
    seg_path = Path(cfg['paths']['checkpoints']) / 'unet_best.pt'
    det_path = Path(cfg['paths']['checkpoints']) / 'fasterrcnn_best.pt'
    if not det_path.exists():
        det_path = Path(cfg['paths']['checkpoints']) / 'fasterrcnn_last.pt'
    seg = det = None
    if seg_path.exists():
        seg = UNet().to(dev)
        seg.load_state_dict(torch.load(seg_path, map_location=dev, weights_only=False)['model'])
        seg.eval()
    if det_path.exists():
        state = torch.load(det_path, map_location=dev, weights_only=False)
        det = build_detector(False, state.get('model_name', cfg['detection']['model']), cfg['detection_size']).to(dev)
        det.load_state_dict(state['model'])
        det.eval()
    return seg, det, dev


def infer(image, seg, det, dev, cfg):
    if image is None:
        raise ValueError('Vui lòng chọn ảnh.')
    original = np.asarray(image.convert('RGB'))
    overlay = original.copy()
    predicted_mask = None
    restored_mask = None
    if seg is not None:
        lb, meta = letterbox(original, cfg['image_size'])
        tensor = torch.from_numpy(lb.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
        with torch.no_grad():
            predicted_mask = (torch.sigmoid(seg(tensor))[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
        l, t = meta['left'], meta['top']
        nw, nh = meta['new_width'], meta['new_height']
        cropped = predicted_mask[t:t + nh, l:l + nw]
        restored_mask = cv2.resize(cropped, (meta['original_width'], meta['original_height']), interpolation=cv2.INTER_NEAREST)
        tint = overlay.copy()
        tint[restored_mask > 0] = (255, 40, 40)
        overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)

    status = 'Chưa có mô hình Detection'
    det_box_original = None
    if det is not None:
        det_image, meta = letterbox(original, cfg['detection_size'])
        tensor = torch.from_numpy(det_image.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
        with torch.no_grad():
            output = det(tensor)[0]
        threshold = cfg['detection'].get('score_threshold', 0.5)
        keep = torch.where(output['scores'] >= threshold)[0]
        if len(keep):
            idx = int(keep[0])
            box = output['boxes'][idx].cpu().numpy()
            scale, left, top = meta['scale'], meta['left'], meta['top']
            x1 = max(0, min(original.shape[1] - 1, int(round((box[0] - left) / scale))))
            y1 = max(0, min(original.shape[0] - 1, int(round((box[1] - top) / scale))))
            x2 = max(x1 + 1, min(original.shape[1], int(round((box[2] - left) / scale))))
            y2 = max(y1 + 1, min(original.shape[0], int(round((box[3] - top) / scale))))
            det_box_original = np.array([x1, y1, x2, y2], dtype=np.float32)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            status = f"Confidence: {float(output['scores'][idx]):.3f}"
        else:
            status = 'Không có bbox vượt ngưỡng confidence.'

    if det_box_original is not None and restored_mask is not None:
        mask_box = mask_to_box(restored_mask)
        if mask_box is None:
            status += ' | Box–Mask: mask rỗng'
        else:
            status += f' | Box–Mask IoU: {box_iou(det_box_original, mask_box):.3f}'

    out_dir = Path(cfg['paths']['demo'])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    Image.fromarray(overlay).save(out_dir / f'{stamp}_overlay.png')
    if restored_mask is not None:
        Image.fromarray((restored_mask * 255).astype(np.uint8)).save(out_dir / f'{stamp}_mask.png')
    mask_image = Image.fromarray((restored_mask * 255).astype(np.uint8)) if restored_mask is not None else None
    return Image.fromarray(overlay), mask_image, status


def launch(cfg):
    import gradio as gr
    seg, det, dev = load_models(cfg)

    def predict(image):
        return infer(image, seg, det, dev, cfg)

    with gr.Blocks(title='ISIC 2018 – Detection & Segmentation') as app:
        gr.Markdown('# ISIC 2018 – Phát hiện và phân đoạn tổn thương da\n**Công cụ học tập/nghiên cứu, không phải công cụ chẩn đoán y khoa.**')
        input_image = gr.Image(type='pil', label='Ảnh đầu vào')
        button = gr.Button('Phân tích')
        overlay = gr.Image(label='Overlay và bounding box')
        mask = gr.Image(label='Mask dự đoán')
        status = gr.Textbox(label='Detection và Box–Mask')
        button.click(predict, inputs=input_image, outputs=[overlay, mask, status])
    app.launch()
