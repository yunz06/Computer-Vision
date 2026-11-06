import cv2
import numpy as np
import matplotlib.pyplot as plt
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh
img, path = chon_anh("Chọn ảnh để thử các kernel")

# BƯỚC 2: Kernel làm mờ trung bình 15x15
# Tất cả phần tử = 1/225 nên tổng = 1 -> mỗi điểm ảnh bằng trung bình vùng 15x15.
kernel_blur = np.ones((15, 15), np.float32) / 225
blur = cv2.filter2D(img, -1, kernel_blur)  # -1: giữ nguyên kiểu dữ liệu đầu ra

# BƯỚC 3: Kernel làm sắc nét
# Tâm = 9, xung quanh = -1, tổng = 1 -> giữ độ sáng, tăng tương phản ở biên.
kernel_sharpen = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)
sharpen = cv2.filter2D(img, -1, kernel_sharpen)

# BƯỚC 4: Kernel phát hiện cạnh
# Tổng = 0 -> vùng phẳng cho kết quả 0 (đen), chỉ chỗ thay đổi mạnh mới sáng.
kernel_edge = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Dùng CV_32F để giữ được giá trị âm, sau đó lấy trị tuyệt đối
edge_raw = cv2.filter2D(gray, cv2.CV_32F, kernel_edge)

# Chuẩn hóa về 0..255 để cạnh hiện rõ
edge = cv2.normalize(np.abs(edge_raw), None, 0, 255,
                     cv2.NORM_MINMAX).astype(np.uint8)

# BƯỚC 5: Kernel tạo hiệu ứng nổi (Emboss)
# Chênh lệch theo đường chéo tạo cảm giác ảnh được dập nổi.
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  0, 1],
    [ 0,  1, 2]
], dtype=np.float32)
emboss_raw = cv2.filter2D(gray, cv2.CV_32F, kernel_emboss)

# Cộng 128 để giá trị âm/dương đều nằm quanh mức xám trung bình
emboss = np.clip(emboss_raw + 128, 0, 255).astype(np.uint8)

# BƯỚC 6: Hiển thị 5 ảnh trên lưới 2x3
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(blur, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(sharpen, cv2.COLOR_BGR2RGB),
          edge, emboss]
titles = ["Anh goc", "Kernel lam mo 15x15", "Kernel lam sac net",
          "Kernel phat hien canh", "Kernel Emboss"]

# Lưu ảnh trước khi plt.show() (show chặn chương trình đến khi đóng cửa sổ)
luu_anh("LocTuyenTinh_KernelLamMo.jpg", blur)
luu_anh("LocTuyenTinh_KernelSacNet.jpg", sharpen)
luu_anh("KernelTuyChinh_PhatHienCanh.jpg", edge)
luu_anh("KernelTuyChinh_Emboss.jpg", emboss)

plt.figure(figsize=(15, 9))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    if i in [3, 4]:   # 2 ảnh xám cần cmap="gray"
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
