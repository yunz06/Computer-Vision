from pathlib import Path
from datetime import datetime
from .integration import load_models as _load_integration_models, run_integration
from .visualization import render_panels


def load_models(cfg):
    seg, det, dev, _ = _load_integration_models(cfg)
    return seg, det, dev


def format_status(result):
    parts = []
    scores = result.get('scores')
    if scores is not None and len(scores):
        parts.append(f"Confidence top-1: {float(scores[0]):.3f} (tổng {len(scores)} box)")
    elif 'detection' in result.get('errors', {}):
        parts.append(result['errors']['detection'])
    else:
        parts.append('Không có bbox vượt ngưỡng confidence.')

    if 'segmentation' in result.get('errors', {}):
        parts.append(result['errors']['segmentation'])

    if result.get('box_mask_iou') is not None:
        parts.append(f"Box–Mask IoU: {result['box_mask_iou']:.3f}")
    parts.append(f"Trạng thái: {result['consistency_status']}")
    parts.append(f"Thời gian suy luận: {result['inference_time_ms']:.1f} ms")
    return ' | '.join(parts)


def save_result_files(result, out_dir):
    if result is None:
        raise ValueError('Chưa có kết quả để lưu. Vui lòng phân tích ảnh trước.')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, mask_img, overlay_img, combined_img = render_panels(result)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    combined_path = out_dir / f'{stamp}_combined.png'
    overlay_path = out_dir / f'{stamp}_overlay.png'
    combined_img.save(combined_path)
    overlay_img.save(overlay_path)
    saved = [str(combined_path), str(overlay_path)]
    if result.get('binary_mask') is not None:
        mask_path = out_dir / f'{stamp}_mask.png'
        mask_img.save(mask_path)
        saved.append(str(mask_path))
    return saved


def infer(image, seg, det, dev, cfg):
    result = run_integration(image, seg, det, dev, cfg)
    _, _, mask_img, overlay_img, _ = render_panels(result)
    save_result_files(result, cfg['paths']['demo'])
    mask_out = mask_img if result.get('binary_mask') is not None else None
    return overlay_img, mask_out, format_status(result)


def build_app(cfg):
    import gradio as gr
    seg, det, dev, load_errors = _load_integration_models(cfg)

    def predict(image):
        try:
            result = run_integration(image, seg, det, dev, cfg)
        except ValueError as exc:
            return None, None, None, None, f'Ảnh đầu vào không hợp lệ: {exc}', None
        orig_img, det_img, mask_img, overlay_img, _ = render_panels(result)
        mask_out = mask_img if result.get('binary_mask') is not None else None
        return orig_img, det_img, mask_out, overlay_img, format_status(result), result

    def save_action(result_state):
        try:
            paths = save_result_files(result_state, cfg['paths']['demo'])
            return 'Đã lưu:\n' + '\n'.join(paths)
        except Exception as exc:
            return f'Lỗi lưu kết quả: {exc}'

    init_note = ''
    if load_errors:
        init_note = '\n\n> Cảnh báo nạp checkpoint: ' + '; '.join(f'{k}: {v}' for k, v in load_errors.items())

    with gr.Blocks(title='ISIC 2018 – Detection & Segmentation') as app:
        gr.Markdown(
            '# ISIC 2018 – Phát hiện và phân đoạn tổn thương da\n'
            '**Công cụ học tập/nghiên cứu — Không phải công cụ chẩn đoán y tế.**' + init_note
        )
        result_state = gr.State(value=None)
        with gr.Row():
            input_image = gr.Image(type='pil', label='Ảnh đầu vào (JPG/PNG)')
        with gr.Row():
            analyze_btn = gr.Button('Phân tích', variant='primary')
            save_btn = gr.Button('Lưu kết quả')
        with gr.Row():
            panel_orig = gr.Image(label='Panel 1: Ảnh gốc')
            panel_det = gr.Image(label='Panel 2: Bounding Box + Confidence')
        with gr.Row():
            panel_mask = gr.Image(label='Panel 3: Binary Mask')
            panel_overlay = gr.Image(label='Panel 4: Overlay trên ảnh gốc')
        info_box = gr.Textbox(label='Box–Mask IoU, Trạng thái nhất quán & Thời gian inference', interactive=False)
        save_box = gr.Textbox(label='Trạng thái lưu file', interactive=False)

        analyze_btn.click(
            predict,
            inputs=input_image,
            outputs=[panel_orig, panel_det, panel_mask, panel_overlay, info_box, result_state],
        )
        save_btn.click(save_action, inputs=result_state, outputs=save_box)
    return app


def launch(cfg):
    app = build_app(cfg)
    app.launch()
