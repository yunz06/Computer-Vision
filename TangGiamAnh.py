import cv2
import numpy as np
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh gốc
img, path = chon_anh("Chọn ảnh để tăng/giảm độ sáng")

# BƯỚC 2: Tăng độ sáng: cộng thêm 50 cho mỗi kênh B, G, R
# cv2.add có "bão hòa": kết quả > 255 sẽ bị giữ ở 255 (không bị tràn số).
brighter_img = cv2.add(img, (50, 50, 50, 0))

# BƯỚC 3: Giảm độ sáng: trừ đi 50, kết quả < 0 sẽ giữ ở 0
darker_img = cv2.subtract(img, (50, 50, 50, 0))

# BƯỚC 4: Lưu ảnh kết quả vào thư mục results
luu_anh("anhthemsang.png", brighter_img)
luu_anh("anhgiamsang.png", darker_img)

# BƯỚC 5: Hiển thị so sánh: Tối | Gốc | Sáng
size = (400, 300)
result = np.hstack([cv2.resize(darker_img, size),
                    cv2.resize(img, size),
                    cv2.resize(brighter_img, size)])
cv2.imshow("Toi (-50) | Goc | Sang (+50)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
