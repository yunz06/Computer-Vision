"""
CHUONG 2 - Bai 1: Thuc hien thuat toan Canny bang OpenCV va scikit-image,
dong thoi so sanh voi Sobel va Laplacian (ly thuyet muc 1b).

Chay:  python canny_so_sanh_opencv_skimage.py            (hien hop thoai chon anh)
       python canny_so_sanh_opencv_skimage.py anh.jpg
"""
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

from common import lay_anh, ket_qua, hien_thi_matplotlib, luu_anh

img, name = lay_anh("Chon anh de phat hien canh Canny", resize_max=800)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- 1. OpenCV: cv2.Canny ----------
t0 = time.perf_counter()
canny_cv = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), 50, 150)
t_cv = time.perf_counter() - t0

# ---------- 2. scikit-image: skimage.feature.canny ----------
t0 = time.perf_counter()
canny_sk = feature.canny(gray, sigma=1.4, low_threshold=50, high_threshold=150)
canny_sk = (canny_sk * 255).astype(np.uint8)
t_sk = time.perf_counter() - t0

# ---------- 3. So sanh voi Sobel va Laplacian ----------
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

t0 = time.perf_counter()
sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sx, sy)
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
t_sobel = time.perf_counter() - t0

t0 = time.perf_counter()
lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
lap = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
t_lap = time.perf_counter() - t0

# Do do "day" cua canh: ty le pixel canh (Canny manh -> mong, Sobel/Laplacian -> day)
def ty_le_canh(e, thr=60):
    return 100.0 * np.count_nonzero(e > thr) / e.size

print("=" * 60)
print(f"Anh: {name}  kich thuoc {gray.shape[1]}x{gray.shape[0]}")
print(f"{'Phuong phap':<22}{'Thoi gian (ms)':>16}{'Ty le pixel canh (%)':>24}")
for ten, e, t in [("Canny OpenCV", canny_cv, t_cv), ("Canny scikit-image", canny_sk, t_sk),
                  ("Sobel", sobel, t_sobel), ("Laplacian", lap, t_lap)]:
    print(f"{ten:<22}{t * 1000:>16.2f}{ty_le_canh(e):>24.2f}")
print("=" * 60)
print("Nhan xet: Canny cho canh mong (1 pixel), lien tuc; Sobel/Laplacian canh day, de nhieu.")

# ---------- 4. Hien thi + luu ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
items = [
    (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Anh goc"),
    (canny_cv, "Canny - OpenCV (50,150)"),
    (canny_sk, "Canny - scikit-image (sigma=1.4)"),
    (sobel, "Sobel (tong hop)"),
    (lap, "Laplacian"),
    (cv2.absdiff(canny_cv, canny_sk), "Khac biet OpenCV vs skimage"),
]
for a, (im, t) in zip(ax.ravel(), items):
    a.imshow(im, cmap=None if im.ndim == 3 else "gray")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()

luu_anh(ket_qua(f"c2b1_{name}_canny_opencv.png"), canny_cv)
luu_anh(ket_qua(f"c2b1_{name}_canny_skimage.png"), canny_sk)
hien_thi_matplotlib(fig, f"c2b1_{name}_so_sanh.png")
