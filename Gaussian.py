import cv2
import matplotlib.pyplot as plt
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh gốc
image, path = chon_anh("Chọn ảnh để lọc Gaussian")

# BƯỚC 2: Đổi BGR -> RGB
# OpenCV đọc ảnh theo thứ tự B-G-R, còn matplotlib hiển thị theo R-G-B,
# nếu không đổi thì màu sẽ bị sai.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# BƯỚC 3: Lọc Gaussian
# Kernel (15, 15): càng lớn càng mờ. Số 0 = để OpenCV tự tính độ lệch chuẩn sigma.
# Lọc Gaussian làm mờ có trọng số: điểm gần tâm ảnh hưởng nhiều hơn điểm xa.
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# BƯỚC 4: Lưu ảnh (đổi lại RGB -> BGR trước khi ghi)
luu_anh("input2_gaussian.jpg", cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))

# BƯỚC 5: Hiển thị ảnh trước và sau khi lọc
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Anh goc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred)
plt.title("Anh sau loc Gaussian")
plt.axis("off")

plt.tight_layout()
plt.show()
