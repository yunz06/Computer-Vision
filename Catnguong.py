import cv2
import numpy as np
from chon_anh import chon_anh, luu_anh

THRESH = 127  # ngưỡng: điểm ảnh > 127 thành trắng, còn lại thành đen

# BƯỚC 1: Chọn và đọc ảnh (hỗ trợ đường dẫn tiếng Việt)
img, path = chon_anh("Chọn ảnh để cắt ngưỡng")

# BƯỚC 2: Chuyển sang ảnh xám (1 kênh, giá trị 0..255)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BƯỚC 3: Cắt ngưỡng (nhị phân hóa)
# Điểm ảnh > THRESH -> 255 (trắng); ngược lại -> 0 (đen)
_, binary = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)

# BƯỚC 4: Ghép ảnh xám (trái) và ảnh nhị phân (phải) rồi hiển thị
result = np.hstack([gray, binary])
cv2.imshow("Trai: Anh xam | Phai: Anh nhi phan (thresh=127)", result)

# BƯỚC 5: Lưu kết quả
luu_anh("ket_qua_cat_nguong.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
