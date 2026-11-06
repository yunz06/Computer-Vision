import cv2
import numpy as np
import matplotlib.pyplot as plt
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh gốc
image, path = chon_anh("Chọn ảnh để làm sắc nét")

# BƯỚC 2: Kernel làm sắc nét (tổng = 1 nên độ sáng chung không đổi)
# Tâm = 5, 4 điểm lân cận trên/dưới/trái/phải = -1:
# nhấn mạnh điểm đang xét so với xung quanh -> biên rõ hơn.
kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# BƯỚC 3: Áp kernel lên ảnh bằng tích chập
sharpened = cv2.filter2D(image, -1, kernel)

# BƯỚC 4: Lưu ảnh kết quả
luu_anh("input4_sharpen.jpg", sharpened)

# BƯỚC 5: Đổi BGR -> RGB để matplotlib hiển thị đúng màu
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

# BƯỚC 6: Hiển thị trước và sau
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Anh goc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened_rgb)
plt.title("Anh sau lam sac net")
plt.axis("off")

plt.tight_layout()
plt.show()
