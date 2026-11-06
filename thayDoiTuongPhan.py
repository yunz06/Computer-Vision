import cv2
import matplotlib.pyplot as plt
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh
img, path = chon_anh("Chọn ảnh để thay đổi độ tương phản")

# BƯỚC 2: Thay đổi độ tương phản
# Công thức: ảnh_mới = alpha * ảnh_cũ + beta
#   alpha > 1: tăng tương phản (vùng sáng sáng hơn, vùng tối tối hơn)
#   alpha < 1: giảm tương phản
#   beta     : cộng thêm để chỉnh độ sáng chung
# convertScaleAbs tự giới hạn kết quả trong 0..255.
alpha = 1.5
beta = 0
contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# BƯỚC 3: Đổi BGR -> RGB để matplotlib hiển thị đúng màu
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
contrast_rgb = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)

# BƯỚC 4: Lưu ảnh kết quả
luu_anh("TangDoTuongPhan.jpg", contrast)

# BƯỚC 5: Hiển thị trước và sau
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Anh goc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(contrast_rgb)
plt.title("Anh tang do tuong phan")
plt.axis("off")

plt.tight_layout()
plt.show()
