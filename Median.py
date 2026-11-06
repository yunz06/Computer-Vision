import cv2
import numpy as np
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn và đọc ảnh gốc
img, path = chon_anh("Chọn ảnh để thử lọc Median")

# BƯỚC 2: Tạo nhiễu muối tiêu (salt & pepper)
noisy = img.copy()
num_pixels = int(0.02 * img.shape[0] * img.shape[1])  # 2% tổng số điểm ảnh

# Nhiễu trắng (muối): chọn ngẫu nhiên vị trí rồi gán 255
y = np.random.randint(0, img.shape[0], num_pixels)
x = np.random.randint(0, img.shape[1], num_pixels)
noisy[y, x] = 255

# Nhiễu đen (tiêu): chọn ngẫu nhiên vị trí rồi gán 0
y = np.random.randint(0, img.shape[0], num_pixels)
x = np.random.randint(0, img.shape[1], num_pixels)
noisy[y, x] = 0

# BƯỚC 3: Lọc Median kích thước 5x5
# Thay mỗi điểm ảnh bằng GIÁ TRỊ TRUNG VỊ của vùng 5x5 -> loại bỏ điểm
# trắng/đen lẻ loi rất tốt mà vẫn giữ được biên (tốt hơn lọc trung bình).
median_img = cv2.medianBlur(noisy, 5)

# BƯỚC 4: Lưu ảnh
luu_anh("median_nhieu.png", noisy)
luu_anh("median_ketqua.png", median_img)

# BƯỚC 5: Resize 3 ảnh về cùng kích thước để ghép
img_small = cv2.resize(img, (350, 350))
noisy_small = cv2.resize(noisy, (350, 350))
median_small = cv2.resize(median_img, (350, 350))

# BƯỚC 6: Tạo thanh nhãn đen phía trên mỗi ảnh
label_height = 30
img_label = np.zeros((label_height, 350, 3), dtype=np.uint8)
noisy_label = np.zeros((label_height, 350, 3), dtype=np.uint8)
median_label = np.zeros((label_height, 350, 3), dtype=np.uint8)

cv2.putText(img_label, "Anh goc", (75, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(noisy_label, "Anh nhieu", (65, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(median_label, "Sau Median Filter", (30, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# BƯỚC 7: Ghép nhãn + ảnh (dọc), rồi ghép 3 cột (ngang) và hiển thị
img_final = np.vstack((img_label, img_small))
noisy_final = np.vstack((noisy_label, noisy_small))
median_final = np.vstack((median_label, median_small))
result = np.hstack((img_final, noisy_final, median_final))

cv2.imshow("So sanh Median Filter", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
