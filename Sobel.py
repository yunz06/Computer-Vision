import cv2
import numpy as np
import matplotlib.pyplot as plt
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn ảnh và chuyển sang ảnh xám
img, path = chon_anh("Chọn ảnh để phát hiện cạnh Sobel")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BƯỚC 2: Làm mờ nhẹ để giảm nhiễu (nhiễu dễ bị nhầm thành cạnh)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# BƯỚC 3: Sobel theo x và y
# CV_64F giữ được giá trị âm. (1,0) = đạo hàm theo x, (0,1) = đạo hàm theo y.
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

# BƯỚC 4: Gradient tổng hợp = sqrt(Gx^2 + Gy^2), rồi giới hạn 0..255
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

# BƯỚC 5: Hiển thị 4 ảnh trên lưới 2x2 và lưu hình tổng hợp
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1), plt.imshow(gray, cmap='gray'), plt.title('Anh goc')
plt.subplot(2, 2, 2), plt.imshow(np.abs(sobelx), cmap='gray'), plt.title('Sobel X')
plt.subplot(2, 2, 3), plt.imshow(np.abs(sobely), cmap='gray'), plt.title('Sobel Y')
plt.subplot(2, 2, 4), plt.imshow(sobel, cmap='gray'), plt.title('Sobel tong hop')

import os
os.makedirs('results', exist_ok=True)
plt.savefig('results/ketqua_sobel.png', dpi=150)
plt.show()
