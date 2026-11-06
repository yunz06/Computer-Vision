import cv2
import numpy as np
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh (thay cho đường dẫn cố định "images/anh.jpg")
img, path = chon_anh("Chọn ảnh để tạo ảnh âm bản")

# BƯỚC 2: Tạo ảnh âm bản
# Mỗi điểm ảnh có giá trị 0..255. Âm bản = 255 - giá trị gốc
# (đen 0 -> trắng 255, sáng -> tối, màu -> màu bù).
negative = 255 - img

# BƯỚC 3: Lưu ảnh âm bản vào thư mục results
luu_anh("anh_am_ban.png", negative)

# BƯỚC 4: Resize 2 ảnh về cùng kích thước
# np.hstack chỉ ghép được khi hai ảnh cao bằng nhau.
size = (500, 350)  # (rộng, cao)
img_small = cv2.resize(img, size)
negative_small = cv2.resize(negative, size)

# BƯỚC 5: Ghép ảnh gốc (trái) và âm bản (phải) theo chiều ngang rồi hiển thị
result = np.hstack((img_small, negative_small))
cv2.imshow("Anh goc va anh am ban", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
