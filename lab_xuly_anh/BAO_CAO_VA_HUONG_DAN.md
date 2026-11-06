---
title: "Báo cáo & hướng dẫn sử dụng – Lab Xử lý ảnh (Chương 2: Canny, Chương 4: Wavelet hash / Khuôn mặt / Phân đoạn)"
lang: vi
---

# 1. Tổng quan

Bộ bài lab viết bằng Python, dùng cùng phong cách với các file có sẵn của bạn (OpenCV + NumPy + Matplotlib, hộp thoại `tkinter` để chọn ảnh, `imdecode/imencode` để đọc/ghi đường dẫn tiếng Việt, kết quả lưu vào thư mục `results/`).

| File | Nội dung | Đề bài |
|---|---|---|
| `common.py` | Hàm dùng chung: hộp thoại chọn ảnh / nhiều ảnh / thư mục / video, đọc-ghi ảnh Unicode, ghép lưới ảnh | – |
| `canny_so_sanh_opencv_skimage.py` | Canny bằng `cv2.Canny` và `skimage.feature.canny`; so sánh Sobel, Laplacian | Ch.2 – Thực hành 1 + Lý thuyết 1b |
| `canny_khao_sat_tham_so.py` | Thay đổi sigma, ngưỡng thấp/cao; so sánh với mặc định | Ch.2 – Thực hành 2 |
| `canny_cac_loai_anh.py` | Canny trên ảnh nhiễu / tương phản thấp / nhiều chi tiết | Ch.2 – Thực hành 3 |
| `canny_phan_doan_nhan_dang_hinh.py` | Canny + phân đoạn (contour) + nhận dạng hình dạng (approxPolyDP, Hough) | Ch.2 – Thực hành 4 |
| `canny_danh_gia_va_anh_mau.py` | Đánh giá chất lượng cạnh (Precision/Recall/F1/Pratt FOM), cải thiện hiệu suất, Canny cho ảnh màu | Ch.2 – Câu hỏi mở rộng 1, 2, 3 |
| `canny_video_webcam.py` | Canny cho video/webcam, có thanh trượt chỉnh ngưỡng, ghi video kết quả | Ch.2 – Câu hỏi mở rộng 4 |
| `hash_lib.py` | Thư viện băm ảnh: wavelet hash, aHash, dHash, pHash, khoảng cách Hamming | Ch.4 |
| `wavelet_hash_danh_gia_roc.py` | Băm wavelet, khoảng cách Hamming, Accuracy / Sensitivity / Specificity, ROC, so sánh các phương pháp băm | Ch.4 – Bài toán + Nâng cao 1 |
| `tim_kiem_anh_wavelet_hash.py` | Ứng dụng tìm kiếm ảnh bằng hàm băm wavelet | Ch.4 – Nâng cao 2 |
| `nhan_dien_khuon_mat_mtcnn_facenet.py` | Webcam + MTCNN + FaceNet nhận diện khuôn mặt thời gian thực | Ch.4 – phần khuôn mặt |
| `phan_doan_anh_cac_phuong_phap.py` | Phân đoạn ảnh: Otsu, Adaptive, K-means, Watershed, GrabCut, Canny+contour | Ch.4 – Phân đoạn ảnh |

# 2. Cài đặt và chạy

```
pip install opencv-python numpy matplotlib scikit-image scipy PyWavelets
# chỉ cần cho nhan_dien_khuon_mat_mtcnn_facenet.py (nhận diện khuôn mặt):
pip install torch torchvision facenet-pytorch
```

Chạy bất kỳ bài nào:

```
python canny_so_sanh_opencv_skimage.py                 # hiện hộp thoại chọn ảnh
python canny_so_sanh_opencv_skimage.py anh.jpg         # hoặc truyền thẳng đường dẫn
```

Nếu bấm **Cancel** ở hộp thoại, chương trình tự dùng ảnh mẫu (hoặc dữ liệu tổng hợp) để vẫn chạy được. Mọi kết quả được lưu trong `results/`. Đặt biến môi trường `LAB_NO_SHOW=1` nếu chỉ muốn lưu file mà không mở cửa sổ.

Ghi chú: `PyWavelets` là tùy chọn khi chạy `wavelet_hash_danh_gia_roc.py` — nếu chưa cài, code tự dùng phép biến đổi Haar viết bằng NumPy (cho kết quả tương đương với wavelet `haar`). Wavelet khác như `db4` cần cài `PyWavelets`.

# 3. Chương 2 – Canny Edge Detector

## 3.1 Lý thuyết

### 3.1.1 Các bước của thuật toán Canny

1. **Giảm nhiễu (Gaussian):** làm mờ ảnh xám bằng bộ lọc Gaussian (kích thước 5×5, sigma ~1–1.4). Đạo hàm rất nhạy với nhiễu nên bước này giúp giảm cạnh giả.
2. **Tính gradient:** dùng Sobel theo x, y được $G_x, G_y$; độ lớn $G=\sqrt{G_x^2+G_y^2}$ và hướng $\theta=\arctan(G_y/G_x)$ (làm tròn về 0°, 45°, 90°, 135°).
3. **Non-maximum suppression (NMS):** với mỗi pixel, so $G$ với hai láng giềng dọc theo hướng gradient; chỉ giữ nếu là cực đại cục bộ. Kết quả: cạnh mỏng 1 pixel.
4. **Ngưỡng kép (double threshold):** pixel $G >$ ngưỡng cao → cạnh mạnh; $G <$ ngưỡng thấp → loại; ở giữa → cạnh yếu (ứng viên).
5. **Theo dõi cạnh bằng độ trễ (hysteresis):** cạnh yếu chỉ được giữ nếu nối (8-lân cận) với một cạnh mạnh; còn lại bị loại. Giúp cạnh liền nét, bỏ nhiễu rời rạc.

### 3.1.2 So sánh với Sobel, Laplacian

| Tiêu chí | Sobel | Laplacian | Canny |
|---|---|---|---|
| Bản chất | Đạo hàm bậc 1, một hướng (x/y) | Đạo hàm bậc 2 (đẳng hướng) | Chuỗi nhiều bước dựa trên đạo hàm bậc 1 |
| Độ dày cạnh | Dày | Dày, có hai đường (zero-crossing) | Mỏng 1 pixel (nhờ NMS) |
| Chống nhiễu | Trung bình (có làm trơn nhẹ) | Kém (đạo hàm bậc 2 khuếch đại nhiễu) | Tốt (Gaussian + hysteresis) |
| Cạnh liền nét | Không | Không | Có |
| Tốc độ | Nhanh nhất | Nhanh | Chậm hơn (nhiều bước) |
| Tham số | Kích thước kernel | Kích thước kernel | sigma, ngưỡng thấp, ngưỡng cao |

`canny_so_sanh_opencv_skimage.py` in thời gian và tỷ lệ pixel cạnh của từng phương pháp để minh họa.

### 3.1.3 Tham số và ảnh hưởng

- **Sigma (Gaussian):** nhỏ → giữ chi tiết nhưng nhiều cạnh giả/nhiễu; lớn → mịn, ít nhiễu nhưng mất cạnh mỏng, cạnh bị dịch vị trí. Thông thường 1–2.
- **Ngưỡng thấp / cao:** ngưỡng cao quyết định cạnh "chắc chắn", ngưỡng thấp quyết định mức độ kéo dài cạnh. Ngưỡng thấp quá nhỏ → nhiều cạnh nhiễu; ngưỡng cao quá lớn → đứt cạnh, mất cạnh yếu. Quy tắc: cao ≈ 2–3 × thấp. Có thể tự động hóa theo trung vị ảnh (`auto_canny`) hoặc Otsu (đã cài trong `canny_danh_gia_va_anh_mau.py`).

### 3.1.4 Ưu điểm, nhược điểm, ứng dụng

**Ưu:** phát hiện tốt (ít bỏ sót), định vị chính xác, mỗi cạnh chỉ một đáp ứng, chống nhiễu tốt, cạnh liền nét.
**Nhược:** chậm hơn Sobel/Laplacian; phải chọn 3 tham số; Gaussian làm mờ nên làm tròn góc, mất cạnh rất mịn; mặc định chỉ xử lý ảnh xám; không hiểu ngữ nghĩa (cạnh vân vật thể cũng bị bắt).

**Lĩnh vực dùng nhiều nhất:** thị giác máy tính công nghiệp và robot, xe tự hành, y tế, nhận dạng/OCR.
**Ví dụ cụ thể:** phát hiện làn đường (Canny + Hough); kiểm tra lỗi/đo kích thước linh kiện; tách biên khối u trong ảnh X-quang/CT; trích biên chữ trước khi OCR; tiền xử lý cho nhận dạng hình dạng (bài 4 của lab).

**So sánh độ chính xác / tốc độ / chống nhiễu (tổng hợp):** độ chính xác định vị: Canny > Sobel > Laplacian; tốc độ: Sobel ≈ Laplacian > Canny; chống nhiễu: Canny > Sobel > Laplacian.

## 3.2 Câu hỏi mở rộng (đã cài đặt trong `canny_danh_gia_va_anh_mau.py`, `canny_video_webcam.py`)

1. **Đánh giá chất lượng cạnh:** khi có ảnh "cạnh chuẩn" (ground truth, ví dụ nhãn BSDS500) dùng Precision, Recall, F1 có dung sai 1–2 pixel và Pratt's Figure of Merit. Code dùng cạnh Canny của ảnh sạch làm chuẩn, rồi đánh giá trên ảnh đã thêm nhiễu. Khi không có ground truth có thể đánh giá định tính hoặc qua kết quả của tác vụ phía sau.
2. **Cải thiện hiệu suất:** lọc nhiễu tốt hơn (bilateral, Non-local Means) trước Canny; ngưỡng tự động (median/Otsu); chọn sigma theo mức nhiễu; đóng cạnh bằng morphology; giảm kích thước ảnh trước khi xử lý để tăng tốc.
3. **Ảnh màu:** có 3 cách – (a) chuyển xám rồi Canny (đơn giản, phổ biến), (b) Canny từng kênh B, G, R rồi OR, (c) chọn kênh có gradient lớn nhất tại mỗi pixel. Cả ba đều có trong code.
4. **Video:** xử lý từng khung (đọc bằng `cv2.VideoCapture`, Canny mỗi khung); có thanh trượt ngưỡng; để ổn định theo thời gian có thể làm mịn ngưỡng/giảm nhiễu giữa các khung; để nhanh hơn thì giảm độ phân giải hoặc dùng GPU (`cv2.cuda`).

## 3.3 Hướng dẫn chạy từng bài thực hành

| Lệnh | Bạn làm gì | Kết quả |
|---|---|---|
| `python canny_so_sanh_opencv_skimage.py` | Chọn 1 ảnh | Bảng thời gian + tỷ lệ pixel cạnh; ảnh so sánh Canny (OpenCV/skimage), Sobel, Laplacian |
| `python canny_khao_sat_tham_so.py` | Chọn 1 ảnh | 3 hình: ảnh hưởng của sigma, của ngưỡng, so sánh mặc định ↔ tự chỉnh |
| `python canny_cac_loai_anh.py` | Chọn lần lượt ảnh nhiễu / tương phản thấp / nhiều chi tiết (Cancel = tự tạo từ ảnh gốc) | Lưới 9 ảnh: không lọc vs Gaussian + ngưỡng tự động |
| `python canny_phan_doan_nhan_dang_hinh.py` | Chọn ảnh có vật thể hình học | Phân đoạn theo contour, nhãn hình (tam giác/vuông/chữ nhật/tròn/đa giác), HoughCircles |
| `python canny_danh_gia_va_anh_mau.py` | Chọn 1 ảnh | Bảng Precision/Recall/F1/Pratt của 6 cách xử lý nhiễu; so sánh Canny ảnh màu |
| `python canny_video_webcam.py` | Chọn video (Cancel = webcam) | Cửa sổ video + trackbar `low`/`high`; ghi `results/canny_video.mp4`. Phím: `q` thoát, `s` lưu khung, space tạm dừng |

# 4. Chương 4 – Băm wavelet (Wavelet hash)

## 4.1 Quy trình (khớp từng bước của đề)

1. **Chuẩn bị dữ liệu:** hộp thoại chọn **thư mục ảnh**. Chương trình tự tạo *cặp tương tự* (ảnh gốc + một biến đổi ngẫu nhiên: nhiễu, đổi sáng, đổi tương phản, xoay nhẹ, nén JPEG, co giãn, làm mờ, cắt lề) và *cặp không tương tự* (hai ảnh khác nhau). Nếu thư mục có file `pairs.csv` (3 cột: `ảnh1,ảnh2,nhãn` với nhãn 1 = tương tự, 0 = không) thì dùng cặp của bạn. Không chọn thư mục → dùng 40 ảnh tổng hợp.
2. **Trích xuất wavelet:** ảnh xám, resize 64×64, biến đổi wavelet nhiều cấp (`pywt.wavedec2`) đến khi hệ số xấp xỉ LL còn 8×8.
3. **Mã băm:** lượng tử hóa mỗi hệ số LL theo trung vị (> trung vị → 1) → chuỗi **64 bit**.
4. **So sánh:** khoảng cách Hamming giữa hai mã băm; hai ảnh "tương tự" nếu Hamming ≤ T.
5. **Đánh giá:** Accuracy, Sensitivity (độ nhạy = TP/(TP+FN)), Specificity (độ đặc hiệu = TN/(TN+FP)), đường cong ROC (quét T từ 0 đến 64) và AUC. Ngưỡng T tối ưu chọn theo chỉ số Youden (TPR − FPR lớn nhất).

Kết quả ghi trong `results/`: `c4_bao_cao.txt`, `c4_cap_anh.csv`, `c4_roc.png` (histogram Hamming + ROC), `c4_vi_du_cap_anh.png`, `c4_so_sanh_phuong_phap.png`.

## 4.2 Bài nâng cao

1. **Khảo sát các phương pháp băm:** cùng một tập cặp ảnh, so sánh wavelet-LL, wavelet-detail (năng lượng hệ số chi tiết), aHash, dHash, pHash (DCT) (+ `db4` nếu cài PyWavelets) bằng AUC/Accuracy/Sensitivity/Specificity và đồ thị ROC chung.
2. **Ứng dụng tìm kiếm ảnh** (`tim_kiem_anh_wavelet_hash.py`): chọn thư mục CSDL → chọn ảnh truy vấn (lặp lại đến khi Cancel) → hiện top-5 ảnh có Hamming nhỏ nhất; lưu `results/c4_tim_kiem_N.png` và cache mã băm `c4_index.npz`.

## 4.3 Ví dụ kết quả (chạy trên dữ liệu tổng hợp 40 ảnh, 160 cặp)

| | Giá trị |
|---|---|
| Hamming TB – cặp tương tự / khác nhau | 2.16 / 25.50 |
| Ngưỡng T tối ưu | 7 |
| Accuracy / Sensitivity / Specificity | 0.931 / 0.950 / 0.912 |
| AUC | 0.911 |

**Lưu ý khi đọc kết quả:** số liệu trên là của dữ liệu *tổng hợp* (ảnh vẽ bằng hình học có nền phẳng). Nền phẳng làm nhiều hệ số LL bằng nhau nên ngưỡng "trung vị" kém ổn định hơn và wavelet-LL thấp hơn aHash/dHash/pHash trong lần chạy này. Với ảnh thật của bạn kết quả sẽ khác — hãy chạy lại trên bộ ảnh thật và dùng chính số liệu đó trong báo cáo, đừng dùng số trên.

# 5. Chương 4 – Nhận diện khuôn mặt thời gian thực (OpenCV + MTCNN + FaceNet)

`nhan_dien_khuon_mat_mtcnn_facenet.py`:

1. Mở webcam bằng `cv2.VideoCapture(0)` (hoặc video/ảnh truyền qua dòng lệnh).
2. **MTCNN** (`facenet_pytorch.MTCNN`) phát hiện khuôn mặt, cắt và chuẩn hóa.
3. **FaceNet** (`InceptionResnetV1` pretrained VGGFace2) tạo vector đặc trưng 512 chiều; so khoảng cách L2 với các khuôn mặt đã đăng ký → hiển thị tên + khoảng cách, hoặc `Unknown` nếu khoảng cách ≥ ngưỡng (mặc định 0.9).

Cách dùng: chạy → hộp thoại chọn **thư mục khuôn mặt đã biết** (`thu_muc/Ten_Nguoi/1.jpg…` hoặc `thu_muc/Ten.jpg`; Cancel nếu muốn tự đăng ký). Trong cửa sổ webcam: `e` đăng ký khuôn mặt đang thấy (hộp thoại nhập tên), `+`/`-` đổi ngưỡng, `s` lưu khung, `q`/ESC thoát. Nếu chưa cài `torch`/`facenet-pytorch` chương trình tự chuyển sang chế độ Haar Cascade (chỉ vẽ khung, không nhận diện).

# 6. Chương 4 – Phân đoạn ảnh

`phan_doan_anh_cac_phuong_phap.py` chạy 6 phương pháp trên cùng ảnh và ghép thành một ảnh so sánh (`results/c4_phan_doan_<tên ảnh>.png`, cộng từng ảnh riêng):

| Phương pháp | Ý tưởng | Điểm mạnh / yếu |
|---|---|---|
| Otsu | Chọn ngưỡng toàn cục tối thiểu phương sai trong lớp | Nhanh; kém khi sáng không đều |
| Adaptive threshold | Ngưỡng theo vùng lân cận | Chịu được sáng không đều; nhiều nhiễu |
| K-means (K=2,3,4) | Gom cụm pixel theo màu | Không dùng thông tin không gian |
| Watershed | Marker từ distance transform | Tách vật dính nhau; dễ phân mảnh quá mức |
| GrabCut | Tách tiền cảnh trong khung chữ nhật (bạn kéo khung bằng chuột, hoặc mặc định khung giữa) | Chính xác nhất, cần khung ban đầu |
| Canny + contour | Đóng cạnh bằng morphology → `findContours` | Đơn giản; phụ thuộc chất lượng cạnh |

# 7. Điều kiện kiểm thử và hạn chế

- Tất cả file **Chương 2**, **wavelet_hash_danh_gia_roc.py**, **tim_kiem_anh_wavelet_hash.py**, **phan_doan_anh_cac_phuong_phap.py** đã được chạy thử thành công (với ảnh/dữ liệu mẫu tự sinh, không cửa sổ).
- Môi trường kiểm thử **không cài được** `PyWavelets`, `torch`, `facenet-pytorch` (không có mạng tới PyPI). Hệ quả: wavelet hash được kiểm thử bằng Haar tự viết (khi bạn cài PyWavelets, code tự dùng `pywt`, nhưng đường chạy này chưa được test); **`nhan_dien_khuon_mat_mtcnn_facenet.py` chưa được chạy với MTCNN/FaceNet thật** (chỉ kiểm tra cú pháp và nhánh Haar/không có webcam). Hãy chạy thử trên máy bạn và báo lỗi nếu có.
- Hộp thoại chọn ảnh (`tkinter`) và cửa sổ OpenCV chưa được kiểm thử trực tiếp vì môi trường không có màn hình; code chọn ảnh theo đúng mẫu trong `Gaussian.py`/`loc_trung_binh.py` của bạn.
- Ảnh trong `results/` đi kèm là kết quả từ ảnh mẫu tự sinh – hãy chạy lại với ảnh của bạn để có số liệu cho báo cáo.
