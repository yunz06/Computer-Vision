import cv2
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn ảnh màu (thay cho đường dẫn cố định cat.jpg)
# cv2.IMREAD_COLOR (hoặc số 1) = đọc ảnh màu BGR
img, path = chon_anh("Chọn ảnh để thực hành")

# BƯỚC 2: Hiển thị ảnh trong 1 giây
# waitKey(1000) = chờ 1000 ms; waitKey(0) = chờ đến khi bấm phím
cv2.imshow('tai anh', img)
cv2.waitKey(1000)

# BƯỚC 3: Lưu ảnh sang tệp mới
luu_anh("1_anhsaukhiluu.png", img)

# BƯỚC 4: Đọc lại dưới dạng ảnh xám (cờ 0 = IMREAD_GRAYSCALE) và lưu
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
luu_anh("1_anhdentrang.png", img_gray)

# BƯỚC 5: Chuyển không gian màu
# HSV: Hue (sắc độ) - Saturation (độ bão hòa) - Value (độ sáng)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
luu_anh("1_anhHSV.png", img_hsv)

# LAB: L (độ sáng) - a (xanh lá <-> đỏ) - b (xanh dương <-> vàng)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
luu_anh("1_anhLAB.png", img_lab)

# BƯỚC 6: Cắt ảnh bằng slicing numpy: img[y1:y2, x1:x2]
# Nếu ảnh nhỏ hơn vùng cắt thì tự động thu nhỏ vùng cắt cho hợp lệ
h, w = img.shape[:2]
y1, y2 = 50, min(250, h)
x1, x2 = 100, min(300, w)
cropped_img = img[y1:y2, x1:x2]

cv2.imshow('Anh da cat', cropped_img)
cv2.waitKey(0)
luu_anh("1_anhdacat.png", cropped_img)

cv2.destroyAllWindows()
