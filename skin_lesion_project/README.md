# ISIC 2018 – Phát hiện và phân đoạn tổn thương da

Project môn Thị giác máy tính: **Object Detection + Image Segmentation** trên ISIC 2018 Task 1. Đây là công cụ nghiên cứu/giảng dạy, **không phải công cụ chẩn đoán y khoa**.

## 1. Môi trường

- Windows 10/11, Python 3.10–3.12 (khuyến nghị Python 3.12).
- PyTorch + Torchvision phải tương thích với CUDA/driver trên máy. Không cài lại Torch tự động bằng `requirements.txt`.
- GPU 4 GB (ví dụ RTX 2050): U-Net dùng batch 4, ảnh 256×256; Faster R-CNN dùng ảnh 512 px, batch 1–2 + AMP.

Cài dependencies phụ trợ tại thư mục project:

```cmd
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Kiểm tra nhanh môi trường & GPU:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.healthcheck
```

## 2. Đặt dữ liệu

Giải nén ZIP ISIC 2018 Task 1 theo cấu trúc dưới đây (script audit tìm đệ quy cả trong thư mục con):

```text
data/raw/images/   # ISIC_XXXXXXX.jpg
data/raw/masks/    # ISIC_XXXXXXX_segmentation.png
```

Nếu có `HAM10000_metadata.csv`, đặt trong `data/raw/` để `prepare` tự ghép `lesion_id` cho group split chống rò rỉ dữ liệu.

## 3. Pipeline đầy đủ (theo Official Plan)

Chạy tại thư mục gốc `skin_lesion_project`:

```cmd
:: D1 - Data Audit
D:\Python\.venv\Scripts\python.exe -m src.cli audit

:: D2 - EDA (thống kê kích thước, diện tích tổn thương, connected components)
D:\Python\.venv\Scripts\python.exe -m src.cli eda

:: D3 & D4 - Tạo manifest.csv và split cố định (train/val/test)
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --seed 42

:: D5 - Baseline cổ điển (Otsu + Morphology, K-means Lab, Watershed, Center-box)
D:\Python\.venv\Scripts\python.exe -m src.cli baselines --split test

:: D5 - Huấn luyện Segmentation (hỗ trợ unet | attention_unet | unet_resnet34 và bce | dice | bce_dice)
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg

:: D5 - Huấn luyện Detection (mặc định chọn checkpoint theo Val mAP@0.5:0.95)
D:\Python\.venv\Scripts\python.exe -m src.cli train-det

:: D6 - Quét ngưỡng & hậu xử lý trên Validation ONLY
D:\Python\.venv\Scripts\python.exe -m src.cli select-val

:: D7 - Đánh giá cuối trên Test Set (ở kích thước ảnh gốc)
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task seg
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task det
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task unet_box

:: Chạy Gradio Demo (4 panel + Box-Mask IoU + nút lưu)
D:\Python\.venv\Scripts\python.exe app.py
```

Với smoke run nhanh 100 ảnh, có thể chạy `chay_thu_100_anh.cmd` (chỉ để kiểm tra pipeline, không dùng số liệu này cho báo cáo).

## 4. Dataset policy

- **Mask binarization**: xác minh quy ước pixel, nhị phân hóa với ngưỡng `> 127`.
- **Detection boxes**: tạo box cho từng connected component có diện tích `>= min_box_area_ratio * (H * W)` (cấu hình trong `configs/default.yaml`). Box là nhãn dẫn xuất từ segmentation mask, không phải annotation box độc lập do chuyên gia gán.
- **Mask rỗng**: giữ trong segmentation nếu hợp lệ; tự động loại khỏi tập training detection và ghi log trong `audit` / `manifest.csv`.
- **Hậu xử lý & Evaluation**: probability map của U-Net được nội suy tuyến tính (`INTER_LINEAR`) về kích thước ảnh gốc **trước** khi threshold; mọi metric Segmentation (`Dice`, `Mask IoU`, `Thresholded Jaccard@0.65`, `Failure Rate Jaccard < 0.65`, `Boundary F1`) và Detection (`AP@0.5`, `AP@0.5:0.95`, `Precision/Recall`, `Top-1 Box IoU`) đều tính ở tọa độ/kích thước gốc.

## 5. Kiểm thử

```cmd
D:\Python\.venv\Scripts\python.exe -m pytest -v
```

## 6. Kết quả và đường dẫn

- `reports/audit.json`, `reports/audit.txt`, `reports/eda_summary.json`, `reports/eda_report.md`
- `data/splits/split.json`, `data/splits/manifest.csv`, `data/splits/train.txt`, `val.txt`, `test.txt`
- `outputs/checkpoints/`: checkpoint tốt nhất (`unet_best.pt`, `fasterrcnn_best.pt`)
- `outputs/results/`: metric JSON, `val_selection.json`, `baseline_summary.json`
- `outputs/figures/`: learning curves, EDA plots, qualitative error analysis grids
- `outputs/demo/`: ảnh tổng hợp, overlay và mask được người dùng lưu từ Gradio UI
