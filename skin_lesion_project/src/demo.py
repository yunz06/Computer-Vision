from pathlib import Path
from datetime import datetime
from io import BytesIO
import numpy as np
from .integration import load_models as _load_integration_models, run_integration
from .visualization import render_panels


def _stats(result):
    image = result.get('original_image')
    h, w = image.shape[:2] if image is not None else (None, None)
    mask = result.get('binary_mask')
    pixels = int(np.count_nonzero(mask)) if mask is not None else None
    total = int(mask.size) if mask is not None else None
    area = 100.0 * pixels / total if total else None
    boxes, scores = result.get('boxes'), result.get('scores')
    rows = []
    if boxes is not None:
        for i, b in enumerate(boxes):
            rows.append((i + 1, float(scores[i]) if scores is not None and i < len(scores) else 0.0,
                         *[float(v) for v in b]))
    return w, h, pixels, total, area, rows


def build_report_text(result):
    w, h, pixels, total, area, rows = _stats(result)
    iou = result.get('box_mask_iou')
    lines = [
        'BÁO CÁO PHÂN TÍCH HÌNH ẢNH', 'ISIC 2018 – Detection & Segmentation',
        f"Thời điểm: {datetime.now():%d/%m/%Y %H:%M:%S}",
        f'Kích thước ảnh: {w} x {h} px' if w and h else 'Kích thước ảnh: không xác định',
        '', '1. DETECTION', f'Số bounding box: {len(rows)}',
    ]
    if rows:
        lines.append(f'Confidence cao nhất: {rows[0][1]:.4f}')
        for i, score, x1, y1, x2, y2 in rows:
            lines.append(f'Box {i}: score={score:.4f}; tọa độ=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})')
    else:
        lines.append('Không có bounding box vượt ngưỡng confidence đã chọn.')
    lines += ['', '2. SEGMENTATION']
    if pixels is None:
        lines.append('Không có mask dự đoán.')
    else:
        lines += [f'Số pixel mask: {pixels:,}/{total:,}', f'Tỷ lệ diện tích mask: {area:.2f}%']
    lines += ['', '3. ĐỐI CHIẾU HÌNH HỌC']
    lines.append(f'IoU giữa bbox Detection và bbox bao quanh Mask: {iou:.4f}' if iou is not None else 'Không tính được IoU do thiếu kết quả.')
    lines += [f"Trạng thái: {result.get('consistency_status', 'Không xác định')}", '', '4. HIỆU NĂNG',
              f"Thời gian suy luận: {float(result.get('inference_time_ms', 0)):.2f} ms", '', '5. GHI CHÚ',
              '- Confidence là điểm số của mô hình, không phải xác suất mắc bệnh.',
              '- IoU trên so sánh hai bounding box, không phải pixel IoU giữa bbox và mask.',
              '- Kết quả chỉ phục vụ học tập/nghiên cứu, không dùng để chẩn đoán hoặc quyết định điều trị.']
    errors = result.get('errors') or {}
    if errors:
        lines += ['', 'CẢNH BÁO'] + [f'- {k}: {v}' for k, v in errors.items()]
    return '\n'.join(lines)


def format_status(result):
    w, h, pixels, total, area, rows = _stats(result)
    parts = [f'Kích thước ảnh: {w} x {h} px' if w and h else 'Không xác định kích thước ảnh',
             f'Detection: {len(rows)} box']
    if rows:
        parts.append(f'Confidence cao nhất: {rows[0][1]:.3f}')
        parts.append('Tọa độ: ' + '; '.join(f'#{r[0]} ({r[2]:.0f}, {r[3]:.0f}, {r[4]:.0f}, {r[5]:.0f})' for r in rows))
    parts.append(f'Segmentation: {pixels:,}/{total:,} pixel ({area:.2f}%)' if pixels is not None else 'Segmentation: không có mask')
    iou = result.get('box_mask_iou')
    if iou is not None:
        parts.append(f'IoU bbox Detection–bbox Mask: {float(iou):.3f}')
    parts += [f"Đối chiếu: {result.get('consistency_status', 'Không xác định')}",
              f"Thời gian suy luận: {float(result.get('inference_time_ms', 0)):.1f} ms",
              'Lưu ý: confidence không phải xác suất bệnh; kết quả không có giá trị chẩn đoán.']
    errors = result.get('errors') or {}
    if errors:
        parts.append('Cảnh báo: ' + ' | '.join(f'{k}: {v}' for k, v in errors.items()))
    return '\n'.join(parts)


def _export_pdf(result, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallNote', parent=styles['BodyText'], fontSize=8, leading=11))
    w, h, pixels, total, area, rows = _stats(result)
    iou = result.get('box_mask_iou')
    data = [['Hạng mục', 'Kết quả'], ['Kích thước ảnh', f'{w} x {h} px' if w and h else 'Không rõ'],
            ['Số bbox', str(len(rows))], ['Confidence cao nhất', f'{rows[0][1]:.4f}' if rows else 'Không có'],
            ['Pixel mask', f'{pixels:,}/{total:,}' if pixels is not None else 'Không có mask'],
            ['Diện tích mask', f'{area:.2f}%' if area is not None else 'Không rõ'],
            ['IoU bbox Detection–bbox Mask', f'{iou:.4f}' if iou is not None else 'Không tính được'],
            ['Trạng thái', str(result.get('consistency_status', 'Không xác định'))],
            ['Thời gian suy luận', f"{float(result.get('inference_time_ms', 0)):.2f} ms"]]
    table = Table(data, colWidths=[65*mm, 105*mm], repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#263746')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#C8D0D8')),
        ('FONTSIZE',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('PADDING',(0,0),(-1,-1),6)]))
    story = [Paragraph('BÁO CÁO PHÂN TÍCH HÌNH ẢNH', styles['Title']),
             Paragraph('ISIC 2018 – Detection &amp; Segmentation', styles['Heading2']),
             Paragraph(f"Thời điểm: {datetime.now():%d/%m/%Y %H:%M:%S}", styles['BodyText']), Spacer(1,8),
             Paragraph('1. TỔNG QUAN KẾT QUẢ', styles['Heading2']), table, Spacer(1,8)]
    if rows:
        box_data = [['Box','Score','x1','y1','x2','y2']]
        box_data += [[str(i),f'{s:.4f}',f'{x1:.1f}',f'{y1:.1f}',f'{x2:.1f}',f'{y2:.1f}'] for i,s,x1,y1,x2,y2 in rows]
        bt = Table(box_data, colWidths=[15*mm,25*mm,25*mm,25*mm,25*mm,25*mm], repeatRows=1)
        bt.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EDF2')),('FONTSIZE',(0,0),(-1,-1),7),('PADDING',(0,0),(-1,-1),5)]))
        story += [Paragraph('2. DANH SÁCH BOUNDING BOX', styles['Heading2']), bt, Spacer(1,8)]
    story.append(Paragraph('3. HÌNH ẢNH KẾT QUẢ', styles['Heading2']))
    panels = render_panels(result)
    for label, pil in zip(['Ảnh gốc','Detection','Binary Mask','Overlay'], panels[:4]):
        if pil is None: continue
        buf = BytesIO(); pil.convert('RGB').save(buf, format='PNG'); buf.seek(0)
        im = PDFImage(buf); scale = min((82*mm)/im.imageWidth, (65*mm)/im.imageHeight)
        im.drawWidth *= scale; im.drawHeight *= scale
        story += [Paragraph(label, styles['Heading3']), im, Spacer(1,5)]
    story += [Paragraph('4. GHI CHÚ VÀ GIỚI HẠN', styles['Heading2']),
              Paragraph('Confidence là điểm số của mô hình Detection, không phải xác suất mắc bệnh. IoU là IoU giữa bbox Detection và bbox bao quanh Mask, không phải pixel IoU giữa bounding box và mask. Công cụ chỉ phục vụ học tập/nghiên cứu, không dùng để chẩn đoán hoặc quyết định điều trị.', styles['SmallNote'])]
    doc.build(story)
    return str(path)


def save_result_files(result, out_dir):
    if result is None: raise ValueError('Chưa có kết quả để lưu. Vui lòng phân tích ảnh trước.')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    panels = render_panels(result); stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f'); paths=[]
    for name, panel in zip(['original','detection','mask','overlay','combined'], panels):
        if panel is None: continue
        p=out/f'{stamp}_{name}.png'; panel.save(p); paths.append(str(p))
    pdf=out/f'{stamp}_report.pdf'; paths.append(_export_pdf(result,pdf))
    txt=out/f'{stamp}_report.txt'; txt.write_text(build_report_text(result),encoding='utf-8'); paths.append(str(txt))
    return paths


def build_app(cfg):
    import gradio as gr
    seg, det, dev, load_errors = _load_integration_models(cfg)
    def predict(image):
        try: result=run_integration(image,seg,det,dev,cfg)
        except Exception as exc: return None,None,None,None,'Không thể phân tích ảnh: '+str(exc),None,None,None
        orig, det_img, mask, overlay, _ = render_panels(result)
        return orig,det_img,(mask if result.get('binary_mask') is not None else None),overlay,format_status(result),build_report_text(result),result,None
    def save_action(state):
        try:
            paths=save_result_files(state,cfg['paths']['demo'])
            pdf=next((p for p in paths if p.endswith('.pdf')),None)
            return 'Đã lưu báo cáo và hình ảnh:\n'+'\n'.join(paths),pdf
        except Exception as exc: return 'Lỗi xuất báo cáo: '+str(exc),None
    note=''
    if load_errors: note='\n\n**Cảnh báo nạp checkpoint:** '+'; '.join(f'{k}: {v}' for k,v in load_errors.items())
    with gr.Blocks(title='ISIC 2018 – Detection & Segmentation') as app:
        gr.Markdown('# ISIC 2018 – Phân tích hình ảnh bằng Detection và Segmentation\n**Công cụ học tập/nghiên cứu — Không phải công cụ chẩn đoán y tế.**'+note)
        state=gr.State(None)
        image=gr.Image(type='pil',label='Ảnh đầu vào (JPG/PNG)')
        with gr.Row(): run=gr.Button('Phân tích ảnh',variant='primary'); save=gr.Button('Xuất báo cáo PDF và lưu hình ảnh')
        with gr.Row(): orig=gr.Image(label='1. Ảnh gốc'); det_img=gr.Image(label='2. Detection – Bounding Box')
        with gr.Row(): mask=gr.Image(label='3. Segmentation – Binary Mask'); overlay=gr.Image(label='4. Overlay')
        info=gr.Textbox(label='Kết quả phân tích và nhận xét',lines=11,interactive=False)
        report=gr.Textbox(label='Nội dung báo cáo',lines=12,interactive=False)
        status=gr.Textbox(label='Trạng thái xuất file',lines=4,interactive=False)
        pdf=gr.File(label='Báo cáo PDF')
        run.click(predict,inputs=image,outputs=[orig,det_img,mask,overlay,info,report,state,pdf])
        save.click(save_action,inputs=state,outputs=[status,pdf])
    return app


def launch(cfg):
    build_app(cfg).launch()
