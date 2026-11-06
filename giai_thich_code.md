# Ghi chú thuyết trình – Xử lý ảnh với OpenCV

## Phần chung: cách chọn ảnh (chon_anh.py)
- `chon_anh()` mở hộp thoại để chọn ảnh bất kỳ, thay cho đường dẫn cố định như `images/anh.jpg`.
- Đọc ảnh bằng `np.fromfile` + `cv2.imdecode` để đường dẫn có dấu tiếng Việt vẫn đọc được (`cv2.imread` sẽ lỗi).
- `luu_anh()` tự tạo thư mục `results` nếu chưa có rồi lưu ảnh.
- Mọi file đều bắt đầu bằng: `img, path = chon_anh(...)`.

**Cách nói khi thuyết trình:** "Ảnh là một ma trận số. OpenCV đọc ảnh màu thành mảng 3 kênh B-G-R, mỗi giá trị từ 0 đến 255. Mọi phép xử lý sau đây đều là tính toán trên ma trận đó."

---

## Nhóm 1: Phép toán điểm ảnh (mỗi điểm xử lý độc lập)

### biendoiamanh.py – Ảnh âm bản
1. Chọn ảnh.
2. `255 - img`: đảo giá trị mọi điểm ảnh (đen thành trắng, sáng thành tối).
3. Lưu kết quả.
4. Resize hai ảnh cùng kích thước vì `np.hstack` chỉ ghép được ảnh cùng chiều cao.
5. Ghép ảnh gốc và âm bản cạnh nhau để so sánh.

### TangGiamAnh.py – Tăng/giảm độ sáng
- `cv2.add(img, 50)` cộng 50 vào mọi kênh; `cv2.subtract` trừ 50.
- Điểm cần nhấn mạnh: hai hàm này có **bão hòa** (kết quả >255 giữ 255, <0 giữ 0). Nếu dùng `img + 50` của numpy thì giá trị tràn số và quay vòng, ảnh bị lỗi màu.

### thayDoiTuongPhan.py – Độ tương phản
- Công thức `mới = alpha × cũ + beta`.
- `alpha = 1.5` làm vùng sáng sáng hơn, vùng tối tối hơn (tăng tương phản). `beta` chỉnh độ sáng chung.
- `convertScaleAbs` tự giới hạn kết quả trong 0..255.
- Đổi BGR sang RGB vì OpenCV dùng B-G-R còn matplotlib dùng R-G-B.

### Catnguong.py – Cắt ngưỡng (ảnh trong tin nhắn của bạn)
1. Chọn ảnh, đổi sang ảnh xám (1 kênh).
2. `cv2.threshold(gray, 127, 255, THRESH_BINARY)`: điểm > 127 thành trắng (255), còn lại thành đen (0).
3. Ghép ảnh xám (trái) và ảnh nhị phân (phải).
- Nhận xét theo ảnh của bạn: vùng tối như bục, thân hoa, viền đồng hồ thành đen; nền sáng thành trắng. Ngưỡng cố định rất nhạy với ánh sáng, vì vậy mặt bục xám đậm bị đen hoàn toàn và mất chi tiết.

---

## Nhóm 2: Chuyển đổi và cắt ảnh

### elearning1.py – Thao tác cơ bản
1. Đọc ảnh màu; `waitKey(1000)` hiển thị 1 giây, `waitKey(0)` chờ bấm phím.
2. Lưu ảnh sang tệp mới, `imwrite` quyết định định dạng theo đuôi file.
3. Ảnh xám: 1 kênh, nhẹ và dùng cho nhiều thuật toán.
4. HSV: Hue (sắc độ), Saturation (độ bão hòa), Value (độ sáng), tiện để tách màu.
5. LAB: L là độ sáng, a và b là hai trục màu; gần với cách mắt người cảm nhận.
6. Cắt ảnh: `img[y1:y2, x1:x2]` (hàng/y trước, cột/x sau).

---

## Nhóm 3: Lọc làm mờ / khử nhiễu

Ý tưởng chung: **tích chập (convolution)**. Trượt một ma trận nhỏ (kernel) trên ảnh, mỗi điểm ảnh mới là tổng có trọng số của các điểm xung quanh.

### loc_trung_binh.py – Lọc trung bình
- Tạo nhiễu Gaussian + muối tiêu để có dữ liệu thử.
- `cv2.blur` với kernel 3x3, 5x5, 9x9: mỗi điểm thành trung bình của vùng lân cận.
- Kernel càng lớn thì khử nhiễu càng mạnh nhưng ảnh càng mờ.
- `filter2D` với kernel toàn `1/25` cho kết quả tương đương `cv2.blur`.
- PSNR (dB) đo chất lượng so với ảnh gốc, càng cao càng giống.

### Gaussian.py – Lọc Gaussian
- `cv2.GaussianBlur(image, (15,15), 0)`: giống lọc trung bình nhưng **trọng số giảm dần từ tâm ra ngoài**, nên mờ tự nhiên hơn.
- Số `0` là để OpenCV tự tính sigma.

### Median.py – Lọc trung vị
- Thêm nhiễu muối tiêu: 2% điểm trắng và 2% điểm đen ở vị trí ngẫu nhiên.
- `cv2.medianBlur(noisy, 5)`: lấy **giá trị giữa** của vùng 5x5 sau khi sắp xếp.
- Điểm mạnh: loại bỏ điểm trắng/đen lẻ loi rất tốt và giữ biên rõ hơn lọc trung bình, vì điểm nhiễu cực trị bị loại khỏi giá trị giữa.

---

## Nhóm 4: Làm sắc nét và các kernel tùy chỉnh

### Sharpen.py – Làm sắc nét
- Kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` có tổng bằng 1 nên độ sáng chung không đổi.
- Điểm trung tâm được nhấn mạnh so với 4 điểm lân cận, nên biên rõ hơn.

### kernel.py – 4 loại kernel
| Kernel | Đặc điểm | Hiệu ứng |
|---|---|---|
| Trung bình 15x15 | Toàn `1/225`, tổng = 1 | Làm mờ mạnh |
| Làm sắc nét | Tâm 9, xung quanh -1, tổng = 1 | Tăng độ nét |
| Phát hiện cạnh | Tâm 8, xung quanh -1, tổng = 0 | Vùng phẳng thành đen, chỉ cạnh sáng |
| Emboss | Chênh lệch theo đường chéo | Hiệu ứng nổi 3D |

- Dùng `CV_32F` để giữ giá trị âm, rồi `abs` + `normalize` (cạnh) hoặc `+128` (emboss) để đưa về 0..255.

---

## Nhóm 5: Phát hiện cạnh bằng gradient

Cạnh = nơi độ sáng thay đổi đột ngột = đạo hàm (gradient) lớn.

### Sobel.py
1. Ảnh xám, làm mờ Gaussian 3x3 để giảm nhiễu.
2. `cv2.Sobel` theo x (cạnh dọc) và theo y (cạnh ngang).
3. Gradient tổng hợp: `sqrt(Gx² + Gy²)`.
4. Hiển thị 4 ảnh: gốc, Sobel X, Sobel Y, tổng hợp.

### prewitt.py
- Cùng ý tưởng với Sobel nhưng kernel có trọng số đều (`-1,0,1` ở cả 3 hàng), không nhấn mạnh hàng giữa.
- So sánh nhanh: Sobel có trọng số `1,2,1` nên khử nhiễu tốt hơn một chút; Prewitt đơn giản hơn.
- Thêm nhãn và tiêu đề rồi ghép lưới 2x2.

---

## Gợi ý khung thuyết trình
1. Ảnh số là ma trận 0..255 (chọn ảnh bằng hộp thoại).
2. Phép toán điểm ảnh: âm bản, sáng/tối, tương phản, cắt ngưỡng.
3. Không gian màu và cắt ảnh.
4. Lọc không gian (tích chập): trung bình, Gaussian, Median.
5. Kernel tùy chỉnh: làm nét, cạnh, emboss.
6. Phát hiện cạnh: Sobel, Prewitt.
7. Kết luận: mọi phép lọc đều là chọn kernel phù hợp mục đích.

## Lưu ý khi chạy
- Đặt `chon_anh.py` cùng thư mục với các file còn lại.
- Cần cài: `pip install opencv-python numpy matplotlib` (tkinter đi kèm Python).
- Kết quả được lưu vào thư mục `results` (tự tạo).
