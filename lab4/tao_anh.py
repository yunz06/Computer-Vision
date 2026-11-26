import numpy as np
from PIL import Image, ImageFilter

# --- Phần 1: Chỉnh cat_1.png -> lưu thành cat_2.png ---
img = Image.open("cat_1.png")

# 1. Xoay góc lệch nhẹ
img = img.rotate(6, resample=Image.Resampling.BILINEAR)

# 2. Làm hơi mờ
img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

# 3. Thêm hạt nhẹ
arr = np.asarray(img, dtype=np.float32)
noise = np.random.normal(0.0, 12.0, size=arr.shape)
img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

img.save("cat_2.png")


# --- Phần 2: Đổi màu đỏ trên car_1.png -> lưu thành car_2.png ---
car = Image.open("car_1.png").convert("RGB")
car_arr = np.asarray(car, dtype=np.float32)

r, g, b = car_arr[:, :, 0], car_arr[:, :, 1], car_arr[:, :, 2]

# Nhận diện vùng màu đỏ (R cao hơn rõ rệt so với G và B)
red_mask = (r > 110) & ((r - g) > 45) & ((r - b) > 45)

# Đổi sang màu xanh dương (giữ lại độ sáng tương đối)
new_arr = car_arr.copy()
new_arr[red_mask, 0] = b[red_mask] * 0.4
new_arr[red_mask, 1] = g[red_mask] * 0.8 + 40
new_arr[red_mask, 2] = np.clip(r[red_mask] * 1.1, 0, 255)

Image.fromarray(np.clip(new_arr, 0, 255).astype(np.uint8)).save("car_2.png")