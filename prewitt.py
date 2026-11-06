import cv2
import numpy as np
from chon_anh import chon_anh, luu_anh

# BƯỚC 1: Chọn ảnh và chuyển sang ảnh xám (phát hiện cạnh làm trên ảnh xám)
img, path = chon_anh("Chọn ảnh để phát hiện cạnh Prewitt")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BƯỚC 2: Hai kernel Prewitt
# kernel_x: phát hiện cạnh DỌC (thay đổi theo chiều ngang)
kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# kernel_y: phát hiện cạnh NGANG (thay đổi theo chiều dọc)
kernel_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# BƯỚC 3: Tích chập ảnh với 2 kernel (dùng CV_32F để giữ giá trị âm)
prewitt_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
prewitt_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

# BƯỚC 4: Độ lớn gradient = sqrt(Gx^2 + Gy^2) -> cạnh theo mọi hướng
prewitt = cv2.magnitude(prewitt_x, prewitt_y)

# BƯỚC 5: Đưa về uint8 (lấy trị tuyệt đối, giới hạn 0..255) để hiển thị
prewitt_x = cv2.convertScaleAbs(prewitt_x)
prewitt_y = cv2.convertScaleAbs(prewitt_y)
prewitt = cv2.convertScaleAbs(prewitt)

# BƯỚC 6: Resize 4 ảnh về cùng kích thước
size = (400, 300)
img_small = cv2.resize(img, size)
x_small = cv2.resize(prewitt_x, size)
y_small = cv2.resize(prewitt_y, size)
prewitt_small = cv2.resize(prewitt, size)

# Ảnh xám 1 kênh -> 3 kênh để ghép chung với ảnh màu
x_small = cv2.cvtColor(x_small, cv2.COLOR_GRAY2BGR)
y_small = cv2.cvtColor(y_small, cv2.COLOR_GRAY2BGR)
prewitt_small = cv2.cvtColor(prewitt_small, cv2.COLOR_GRAY2BGR)


# BƯỚC 7: Hàm thêm tiêu đề phía trên mỗi ảnh
def add_label(image, text):
    label = np.zeros((40, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return np.vstack((label, image))


img_final = add_label(img_small, "Anh goc")
x_final = add_label(x_small, "Prewitt X")
y_final = add_label(y_small, "Prewitt Y")
prewitt_final = add_label(prewitt_small, "Ket hop Prewitt X va Y")

# BƯỚC 8: Ghép thành lưới 2x2 và thêm tiêu đề lớn
top = np.hstack((img_final, x_final))
bottom = np.hstack((y_final, prewitt_final))
result = np.vstack((top, bottom))

title = np.zeros((60, result.shape[1], 3), dtype=np.uint8)
cv2.putText(title, "PHAT HIEN CANH BANG PHUONG PHAP PREWITT",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
final_result = np.vstack((title, result))

luu_anh("ketqua_prewitt.png", final_result)

cv2.imshow("Ket qua Prewitt", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
