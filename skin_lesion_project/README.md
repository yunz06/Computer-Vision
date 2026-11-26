# ISIC 2018 – Phát hiện và phân đoạn tổn thương da

Project môn **Thị giác máy tính (Computer Vision)**, thực hiện trên bộ dữ liệu **ISIC 2018 Task 1**, tập trung vào hai bài toán:

- **Image Segmentation:** Phân đoạn vùng tổn thương trên ảnh.
- **Object Detection:** Phát hiện và xác định vị trí vùng tổn thương bằng Bounding Box.

Project được xây dựng nhằm mục đích học tập, nghiên cứu và thực hành các phương pháp xử lý ảnh, học sâu và đánh giá mô hình. **Đây không phải là công cụ chẩn đoán y khoa và không thay thế ý kiến của chuyên gia y tế.**

---

## 1. Môi trường và công nghệ

### 1.1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10/11 |
| Python | 3.10–3.12 |
| Python khuyến nghị | 3.12 |
| Deep Learning | PyTorch, Torchvision |
| GPU | NVIDIA GPU hỗ trợ CUDA |
| RAM | Khuyến nghị từ 8 GB |
| VRAM | Có thể chạy trên GPU 4 GB với cấu hình phù hợp |

Các mô hình được sử dụng:

- **U-Net:** Phân đoạn vùng tổn thương.
- **Faster R-CNN:** Phát hiện vùng tổn thương bằng Bounding Box.
- **Otsu, K-means Lab, Watershed:** Các phương pháp phân đoạn cổ điển dùng làm baseline.
- **Center-box:** Baseline hình học cho bài toán Detection.

### 1.2. Cài đặt thư viện

Mở CMD tại thư mục project `skin_lesion_project`.

Cài đặt các thư viện phụ trợ:

```cmd
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

**Lưu ý:** PyTorch và Torchvision cần được cài đặt theo phiên bản tương thích với CUDA và driver trên máy. Không cài đặt lại hai thư viện này một cách tự động thông qua `requirements.txt`.

Kiểm tra môi trường và GPU:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.healthcheck
```

Nếu chương trình nhận diện được GPU và các thư viện cần thiết, có thể tiếp tục thực hiện pipeline.

---

## 2. Chuẩn bị dữ liệu

Sử dụng bộ dữ liệu **ISIC 2018 Task 1** gồm ảnh gốc và segmentation mask tương ứng.
1. Training Images – Ảnh gốc

https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip

2. Training Ground Truth – Segmentation Masks

https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip

Trang dữ liệu chính thức của ISIC:

https://challenge.isic-archive.com/data/

Sau khi giải nén dữ liệu, sắp xếp theo cấu trúc:

```text
skin_lesion_project/
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   │   ├── ISIC_XXXXXXX.jpg
│   │   │   └── ...
│   │   │
│   │   └── masks/
│   │       ├── ISIC_XXXXXXX_segmentation.png
│   │       └── ...
│
└── ...
```

Chương trình hỗ trợ tìm kiếm ảnh và mask đệ quy trong các thư mục con.

### Metadata

Nếu có file metadata chứa thông tin liên kết giữa ảnh và tổn thương, có thể truyền đường dẫn khi chuẩn bị dữ liệu:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --metadata <duong_dan_csv>
```

Chỉ sử dụng metadata khi đã xác minh được ánh xạ giữa ID ảnh và thông tin tổn thương. Metadata HAM10000 không mặc nhiên tương ứng với ID ảnh của ISIC 2018 Task 1.

Nếu không có metadata phù hợp, chương trình sử dụng pHash để hỗ trợ phát hiện ảnh gần trùng. Phương pháp này không chứng minh được các ảnh thuộc cùng một tổn thương.

---

## 3. Quy trình thực hiện

Pipeline được tổ chức thành các giai đoạn từ kiểm tra dữ liệu đến huấn luyện, đánh giá và chạy ứng dụng demo.

Thực hiện các lệnh tại thư mục gốc `skin_lesion_project`.

### Bước 1. Kiểm tra dữ liệu (Data Audit)

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli audit
```

Kiểm tra:

- Số lượng ảnh và segmentation mask.
- Khả năng ghép cặp ảnh với mask.
- File bị lỗi hoặc không đọc được.
- Kích thước ảnh và mask.
- Mask rỗng và giá trị pixel không hợp lệ.

### Bước 2. Phân tích dữ liệu (EDA)

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli eda
```

Thống kê và trực quan hóa:

- Phân bố kích thước ảnh.
- Tỷ lệ diện tích vùng tổn thương.
- Phân bố dữ liệu.
- Connected Components trong segmentation mask.

### Bước 3. Chuẩn bị dữ liệu và chia tập

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --seed 42
```

Tạo manifest và chia dữ liệu thành ba tập:

| Tập dữ liệu | Tỷ lệ |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Việc chia tập được cố định bằng seed để bảo đảm khả năng tái lập kết quả.

### Bước 4. Chạy các phương pháp Baseline

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli baselines --split test
```

Các phương pháp được sử dụng:

- Otsu kết hợp Morphology.
- K-means trên không gian màu Lab.
- Watershed.
- Center-box cho bài toán Detection.

Kết quả baseline được dùng để so sánh với mô hình học sâu.

### Bước 5. Huấn luyện Segmentation

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg
```

Các kiến trúc được hỗ trợ:

- U-Net.
- Attention U-Net.
- U-Net ResNet34.

Các hàm loss được hỗ trợ:

- BCE.
- Dice Loss.
- BCE + Dice Loss.

Mô hình được huấn luyện trên tập Train và theo dõi kết quả trên Validation. Checkpoint tốt nhất được lựa chọn dựa trên chỉ số đánh giá đã cấu hình.

Với GPU có VRAM khoảng 4 GB, cấu hình U-Net sử dụng ảnh 256 × 256 và batch size 4.

### Bước 6. Huấn luyện Detection

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli train-det
```

Mô hình Faster R-CNN được sử dụng để dự đoán Bounding Box của vùng tổn thương.

Checkpoint được lựa chọn dựa trên kết quả đánh giá Validation, mặc định ưu tiên mAP@0.5:0.95.

Với GPU VRAM khoảng 4 GB, có thể sử dụng ảnh 512 px, batch size 1–2 và AMP tùy theo cấu hình phần cứng.

### Bước 7. Lựa chọn ngưỡng trên Validation

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli select-val
```

Giai đoạn này lựa chọn:

- Ngưỡng threshold cho xác suất dự đoán của U-Net.
- Cấu hình hậu xử lý segmentation.
- Ngưỡng confidence cho Detection.

**Chỉ sử dụng tập Validation để lựa chọn ngưỡng và cấu hình. Không sử dụng kết quả Test để điều chỉnh mô hình.**

### Bước 8. Đánh giá cuối cùng trên Test

Đánh giá Segmentation:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task seg
```

Đánh giá Detection:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task det
```

Đánh giá Bounding Box được suy ra từ kết quả U-Net:

```cmd
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task unet_box
```

Các kết quả được lưu thành file JSON và hình ảnh trực quan hóa để phục vụ phân tích, so sánh và viết báo cáo.

---

## 4. Quy ước xử lý dữ liệu và đánh giá

### 4.1. Segmentation Mask

Mask được nhị phân hóa bằng ngưỡng:

```python
binary_mask = mask > 127
```

Các pixel thuộc vùng tổn thương được xác định là vùng trắng, phần nền là vùng đen.

### 4.2. Detection Bounding Box

Bounding Box được tạo từ segmentation mask.

Mặc định, chương trình lấy **thành phần trắng lớn nhất** để tạo một Bounding Box cho mỗi ảnh (`multi_component_boxes: false`).

Các thành phần nhỏ còn lại vẫn được giữ trong Ground Truth của bài toán Segmentation nhưng không được xem là đối tượng Detection độc lập.

Cần lưu ý rằng Bounding Box là nhãn được suy ra từ segmentation mask, không phải annotation Bounding Box độc lập do chuyên gia gán.

Mask rỗng được giữ lại trong dữ liệu Segmentation nếu hợp lệ. Những mẫu này được loại khỏi tập huấn luyện Detection và được ghi nhận trong quá trình audit/manifest.

### 4.3. Đánh giá ở kích thước ảnh gốc

Probability map của U-Net được nội suy tuyến tính (`INTER_LINEAR`) về kích thước ảnh gốc trước khi áp dụng threshold.

Các chỉ số đánh giá được tính trên tọa độ/kích thước ảnh gốc.

**Segmentation:**

- Dice.
- Mask IoU.
- Thresholded Jaccard@0.65.
- Failure Rate Jaccard < 0.65.
- Boundary F1.

**Detection:**

- AP@0.5.
- AP@0.5:0.95.
- Precision.
- Recall.
- Top-1 Box IoU.

---

## 5. Chạy ứng dụng Demo Gradio

Demo cho phép tải ảnh lên và xem kết quả dự đoán của hai mô hình Detection và Segmentation trên cùng một giao diện.

### 5.1. Khởi chạy Demo

Đảm bảo đã cài đặt dependencies:

```cmd
D:\Python\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Tại thư mục gốc `skin_lesion_project`, chạy:

```cmd
D:\Python\.venv\Scripts\python.exe app.py
```

Sau khi khởi động thành công, Gradio sẽ hiển thị địa chỉ truy cập trên cửa sổ CMD, thường có dạng:

```text
http://127.0.0.1:7860
```

Mở địa chỉ này trên trình duyệt để sử dụng ứng dụng.

### 5.2. Các bước sử dụng

**Bước 1 – Tải ảnh đầu vào**

Chọn ảnh cần phân tích bằng khu vực tải ảnh trên giao diện Gradio.

Ảnh đầu vào cần thuộc định dạng được hỗ trợ, chẳng hạn JPG hoặc PNG.

**Bước 2 – Chạy mô hình**

Nhấn nút **Phân tích ảnh**.

Chương trình thực hiện suy luận bằng hai nhánh:

- Faster R-CNN: phát hiện vùng tổn thương và tạo Bounding Box.
- U-Net: tạo probability map và segmentation mask.

**Bước 3 – Xem kết quả trực quan**

Giao diện hiển thị bốn khung hình:

| Khung hình | Nội dung |
|---|---|
| 1. Original Image | Ảnh đầu vào |
| 2. Detection | Bounding Box và confidence |
| 3. Segmentation | Binary Mask dự đoán |
| 4. Overlay | Ảnh gốc kết hợp vùng Mask |

**Bước 4 – Xem thông tin phân tích**

Khu vực kết quả cung cấp các thông tin:

- Kích thước ảnh đầu vào.
- Số lượng Bounding Box được phát hiện.
- Confidence của Detection.
- Tọa độ từng Bounding Box.
- Số pixel thuộc vùng Segmentation Mask.
- Tỷ lệ diện tích Mask trên ảnh.
- IoU giữa Bounding Box của Detection và Bounding Box bao quanh Mask.
- Trạng thái đối chiếu giữa hai kết quả.
- Thời gian suy luận.

**Lưu ý về chỉ số IoU:** Đây là IoU giữa hai Bounding Box, một Bounding Box từ Detection và một Bounding Box được tạo bao quanh Segmentation Mask. Chỉ số này không phải pixel IoU giữa Bounding Box và Mask.

Confidence là điểm số của mô hình Detection, không phải xác suất mắc bệnh.

### 5.3. Xuất báo cáo PDF

Sau khi phân tích ảnh, nhấn nút **Xuất báo cáo PDF và lưu hình ảnh**.

Chương trình tạo báo cáo gồm:

- Thông tin ảnh đầu vào.
- Thống kê kết quả Detection.
- Danh sách Bounding Box và confidence.
- Thống kê Segmentation Mask.
- Kết quả đối chiếu Detection và Segmentation.
- Thời gian suy luận.
- Hình ảnh gốc, Detection, Mask và Overlay.
- Ghi chú về giới hạn của kết quả.

File PDF được cung cấp trên giao diện để tải về. Các hình ảnh kết quả và file báo cáo cũng được lưu trong thư mục:

```text
outputs/
└── demo/
```

Demo chỉ hỗ trợ quan sát và phân tích kết quả mô hình trong phạm vi project học tập/nghiên cứu. Không sử dụng kết quả để tự chẩn đoán hoặc đưa ra quyết định điều trị.

---

## 6. Kiểm thử

Chạy toàn bộ kiểm thử bằng lệnh:

```cmd
D:\Python\.venv\Scripts\python.exe -m pytest -v
```

Các bài kiểm thử giúp kiểm tra những thành phần chính của project, bao gồm xử lý dữ liệu, các hàm tính toán metric và những chức năng liên quan.

---

## 7. Kết quả và thư mục lưu trữ

Các kết quả sau khi chạy pipeline được tổ chức theo các thư mục:

| Đường dẫn | Nội dung |
|---|---|
| `reports/audit.json` | Kết quả kiểm tra dữ liệu dạng JSON |
| `reports/audit.txt` | Báo cáo kiểm tra dữ liệu |
| `reports/eda_summary.json` | Thống kê dữ liệu |
| `reports/eda_report.md` | Báo cáo EDA |
| `data/splits/split.json` | Cấu hình chia tập |
| `data/splits/manifest.csv` | Danh sách và thông tin mẫu dữ liệu |
| `data/splits/train.txt` | Danh sách ảnh Train |
| `data/splits/val.txt` | Danh sách ảnh Validation |
| `data/splits/test.txt` | Danh sách ảnh Test |
| `outputs/checkpoints/` | Các checkpoint mô hình |
| `outputs/results/` | Metric, kết quả đánh giá và lựa chọn Validation |
| `outputs/figures/` | Biểu đồ và hình ảnh phân tích |
| `outputs/demo/` | Ảnh kết quả và báo cáo được lưu từ Demo |

Một số checkpoint chính:

```text
outputs/checkpoints/
├── unet_best.pt
├── unet_last.pt
├── fasterrcnn_best.pt
└── fasterrcnn_last.pt
```

Các file checkpoint có thể được sử dụng để nạp lại mô hình mà không cần huấn luyện từ đầu, với điều kiện cấu hình kiến trúc và môi trường tương thích.

---

## 8. Lưu ý khi sử dụng project

1. Giữ nguyên dữ liệu gốc trong `data/raw/` trong suốt quá trình thực hiện.
2. Sử dụng cùng một cấu hình chia tập khi huấn luyện và đánh giá để bảo đảm tính nhất quán.
3. Chỉ lựa chọn threshold và cấu hình hậu xử lý trên Validation.
4. Không sử dụng Test Set để điều chỉnh mô hình.
5. Bounding Box Detection được suy ra từ segmentation mask, vì vậy cần nêu rõ giới hạn này khi trình bày kết quả.
6. Không xem confidence là xác suất mắc bệnh.
7. Kết quả của project phục vụ mục đích học tập và nghiên cứu, không có giá trị chẩn đoán y khoa.

---

## 9. Tổng kết

Project xây dựng một quy trình tương đối đầy đủ cho hai bài toán Object Detection và Image Segmentation trên ISIC 2018 Task 1, bắt đầu từ kiểm tra dữ liệu, phân tích dữ liệu, xây dựng baseline, huấn luyện mô hình, lựa chọn cấu hình trên Validation và đánh giá cuối cùng trên Test.

Ứng dụng Gradio được tích hợp để trực quan hóa kết quả dự đoán, đối chiếu hai phương pháp và xuất báo cáo PDF, hỗ trợ quá trình kiểm tra và trình bày kết quả của project.

**Mục tiêu chính:** So sánh các phương pháp xử lý ảnh cổ điển với mô hình học sâu, đồng thời đánh giá khả năng phát hiện và phân đoạn vùng tổn thương trên bộ dữ liệu ISIC 2018 Task 1.