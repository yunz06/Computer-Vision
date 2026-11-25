# ISIC 2018 – Phát hiện và phân đoạn tổn thương da

Project môn Thị giác máy tính: **Detection + Semantic Segmentation** trên ISIC 2018 Task 1. Đây là công cụ nghiên cứu/giảng dạy, **không phải thiết bị chẩn đoán y khoa**.

## 1. Môi trường

- Windows 10/11, Python 3.10–3.12 (khuyến nghị Python 3.12).
- PyTorch + Torchvision phải tương thích với CUDA/driver trên máy. Không cài lại Torch tự động bằng `requirements.txt`.
- GPU 4 GB: U-Net dùng batch nhỏ; Faster R-CNN mặc định batch 1, ảnh 512 px.

Mở CMD tại thư mục project và dùng Python dùng chung:

```cmd
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Nếu `torch` hoặc `torchvision` chưa có, cài theo hướng dẫn chính thức của PyTorch phù hợp máy. Project không tự nâng cấp các package đó.

## 2. Đặt dữ liệu

Giải nén ZIP theo cấu trúc dưới đây (các file có thể nằm trong thư mục con; script audit tìm đệ quy):

```text
data/raw/images/   # ISIC_XXXXXXX.jpg
data/raw/masks/    # ISIC_XXXXXXX_segmentation.png
```

Tên mask được ghép bằng cách bỏ hậu tố `_segmentation` khỏi tên file. Ảnh không có mask, mask không có ảnh, file hỏng và mask rỗng được thống kê. Không xóa dữ liệu gốc.

## 3. Các lệnh chính

Chạy tại thư mục gốc `skin_lesion_project`:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli audit
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --sample-size 100 --seed 42
D:\Python\.venv\Scripts\python.exe -m src.cli baselines --limit 100
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg --epochs 2 --limit 100
D:\Python\.venv\Scripts\python.exe -m src.cli train-det --epochs 2 --limit 100 --weights none --model fasterrcnn_mobilenet_v3_large_fpn
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task seg
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task det
D:\Python\.venv\Scripts\python.exe -m src.cli demo
```

`--limit 100` là smoke run để kiểm tra pipeline, không phải kết quả khoa học cuối cùng. Bỏ `--limit` để dùng toàn bộ split. `prepare` tạo split theo seed cố định; nếu có `lesion_id` đáng tin cậy trong metadata thì phiên bản hiện tại vẫn ghi rõ split ảnh-level và không giả định đã chống leakage theo bệnh nhân.

## 4. Dataset policy

- Segmentation: dùng mask nhị phân; mask resize bằng nearest-neighbor.
- Detection: tạo **một bounding box tight cho thành phần liên thông lớn nhất** trong mask, làm đại diện vùng tổn thương chính. Các thành phần còn lại được thống kê để kiểm tra; không tự động biến mọi đốm nhỏ thành object.
- Mask rỗng được giữ trong segmentation nếu hợp lệ; không dùng để tạo target detection dương. Các mẫu không tạo được box được loại khỏi split detection và ghi log.
- Ảnh được letterbox giữ tỷ lệ, mask dùng nearest-neighbor; padding được lưu trong metadata transform.

## 5. Kiểm thử

```cmd
D:\Python\.venv\Scripts\python.exe -m pytest -q
```

Các test kiểm tra ghép cặp, letterbox, mask nhị phân, metric, bbox và forward pass U-Net bằng dữ liệu tổng hợp. Test tổng hợp không thay thế test trên ISIC thật.

## 6. Kết quả và đường dẫn

- `reports/audit.json`, `reports/audit.txt`: audit dataset.
- `data/splits/*.json`: split cố định.
- `outputs/checkpoints/`: checkpoint tốt nhất.
- `outputs/results/`: metric JSON và kết quả baseline (Otsu, K-means Lab, center-box).
- `outputs/figures/`: đường cong loss và ảnh minh họa.
- `outputs/demo/`: ảnh kết quả demo.

## 7. Lưu ý thực nghiệm

1. Không dùng test split để chọn siêu tham số.
2. Detection ground truth được suy ra từ segmentation mask; cần nêu rõ giới hạn này trong báo cáo.
3. 100 ảnh chỉ để kiểm thử pipeline/thử nghiệm ban đầu; không đại diện cho hiệu năng tổng quát.
4. Faster R-CNN dùng pretrained weights nếu tải được. GPU 4 GB nên thử MobileNetV3-FPN trước; nếu dùng ResNet50-FPN v2 bị hết VRAM, chạy lại với `--model fasterrcnn_mobilenet_v3_large_fpn`. Khi offline có thể dùng `--weights none`, khi đó mô hình khởi tạo ngẫu nhiên và cần huấn luyện đủ lâu hơn.
5. Báo cáo cuối cùng phải ghi cấu hình, seed, split, checkpoint và metric thực chạy; không dùng số liệu mẫu trong README.
