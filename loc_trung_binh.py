import sys
import cv2
import numpy as np
from chon_anh import chon_anh, luu_anh


def make_sample_image(size=400):
    """Tạo ảnh mẫu đơn giản nếu người dùng không chọn ảnh."""
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (180, 180), (0, 0, 255), -1)
    cv2.circle(img, (280, 130), 70, (0, 160, 0), -1)
    cv2.line(img, (40, 300), (360, 260), (255, 0, 0), 5)
    cv2.putText(img, "Mean Filter", (80, 370), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 3)
    return img


def add_noise(img, sigma=25):
    """Thêm nhiễu Gaussian + một ít nhiễu muối tiêu."""
    # Nhiễu Gaussian: cộng số ngẫu nhiên phân phối chuẩn vào mọi điểm ảnh
    noisy = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Nhiễu muối tiêu: 1% điểm đen, 1% điểm trắng
    mask = np.random.rand(*img.shape[:2])
    noisy[mask < 0.01] = 0
    noisy[mask > 0.99] = 255
    return noisy


def label(img, text):
    """Vẽ thanh đen và ghi tiêu đề lên đầu ảnh."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    # 1. Đọc ảnh: ưu tiên đường dẫn truyền qua dòng lệnh, nếu không thì mở hộp thoại.
    #    Nếu bấm Cancel -> dùng ảnh mẫu tự tạo.
    if len(sys.argv) > 1:
        img = cv2.imdecode(np.fromfile(sys.argv[1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"Không đọc được ảnh: {sys.argv[1]}")
        img = cv2.resize(img, (400, 400))
    else:
        try:
            img, _ = chon_anh("Chọn ảnh để demo lọc trung bình")
            img = cv2.resize(img, (400, 400))
        except SystemExit:
            print("Không chọn ảnh -> dùng ảnh mẫu.")
            img = make_sample_image()

    # 2. Tạo ảnh nhiễu để có cái mà lọc
    noisy = add_noise(img)

    # 3. Lọc trung bình với nhiều kích thước kernel
    #    Kernel càng lớn -> khử nhiễu mạnh hơn nhưng ảnh càng mờ.
    blur3 = cv2.blur(noisy, (3, 3))
    blur5 = cv2.blur(noisy, (5, 5))
    blur9 = cv2.blur(noisy, (9, 9))

    # 4. Cách tương đương: tự tạo kernel trung bình rồi dùng filter2D
    k = 5
    kernel = np.ones((k, k), np.float32) / (k * k)
    blur5_manual = cv2.filter2D(noisy, -1, kernel)
    print("cv2.blur 5x5 == filter2D 5x5 ?",
          np.allclose(blur5, blur5_manual, atol=1))

    # 5. So sánh chất lượng bằng PSNR so với ảnh gốc (càng cao càng tốt)
    for name, res in [("Anh nhieu", noisy), ("Mean 3x3", blur3),
                      ("Mean 5x5", blur5), ("Mean 9x9", blur9)]:
        print(f"{name:10s} PSNR = {cv2.PSNR(img, res):.2f} dB")

    # 6. Ghép ảnh: hàng 1 (gốc, nhiễu), hàng 2 (3 ảnh đã lọc)
    row1 = np.hstack([label(img, "Goc"), label(noisy, "Co nhieu")])
    row2 = np.hstack([label(blur3, "Mean 3x3"), label(blur5, "Mean 5x5"),
                      label(blur9, "Mean 9x9")])
    # Hàng 1 hẹp hơn hàng 2 nên phải phóng to cho bằng chiều rộng
    row1 = cv2.resize(row1, (row2.shape[1],
                             int(row1.shape[0] * row2.shape[1] / row1.shape[1])))
    result = np.vstack([row1, row2])

    luu_anh("mean_filter_results.png", result)

    # 7. Thu nhỏ kết quả cho vừa màn hình rồi hiển thị
    screen_width, screen_height = 1200, 700
    h, w = result.shape[:2]
    scale = min(screen_width / w, screen_height / h, 1)
    result_small = cv2.resize(result, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Demo loc trung binh (Mean Filter)", cv2.WINDOW_NORMAL)
    cv2.imshow("Demo loc trung binh (Mean Filter)", result_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
