"""
CHUONG 2 - Cac cau hoi mo rong (minh hoa bang code):
  1. Danh gia chat luong canh Canny: Precision / Recall / F1 so voi "canh chuan".
     (Neu khong co ground truth, lay canh Canny cua anh SACH lam chuan, roi danh
      gia tren anh bi nhieu.)  Co them chi so Pratt FOM.
  2. Cai thien hieu suat: nguong tu dong (median / Otsu), loc song phuong,
     non-local-means khu nhieu truoc Canny.
  3. Canny tren anh mau: (a) chuyen xam, (b) Canny tung kenh roi OR,
     (c) lay gradient lon nhat 3 kenh (Di Zenzo gan dung).
  4. Canny cho video: xem canny_video_webcam.py

Chay:  python canny_danh_gia_va_anh_mau.py
"""
import cv2
import numpy as np
from scipy import ndimage

from common import lay_anh, luoi_anh, hien_thi_cv

img, name = lay_anh("Chon anh de danh gia Canny", resize_max=600)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
np.random.seed(0)


def prf(pred, gt, tol=2):
    """Precision/Recall/F1 co dung sai tol pixel (canh lech 1-2 pixel van tinh dung)."""
    pred_b, gt_b = pred > 0, gt > 0
    dt_gt = ndimage.distance_transform_edt(~gt_b)
    dt_pr = ndimage.distance_transform_edt(~pred_b)
    tp_p = np.count_nonzero(pred_b & (dt_gt <= tol))
    tp_r = np.count_nonzero(gt_b & (dt_pr <= tol))
    p = tp_p / max(1, np.count_nonzero(pred_b))
    r = tp_r / max(1, np.count_nonzero(gt_b))
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1


def pratt_fom(pred, gt, alpha=1 / 9):
    """Pratt Figure of Merit (1 = hoan hao)."""
    pred_b, gt_b = pred > 0, gt > 0
    n_i, n_d = np.count_nonzero(gt_b), np.count_nonzero(pred_b)
    if max(n_i, n_d) == 0:
        return 1.0
    d = ndimage.distance_transform_edt(~gt_b)[pred_b]
    return float(np.sum(1.0 / (1.0 + alpha * d ** 2)) / max(n_i, n_d))


def auto_canny(g, s=0.33):
    m = np.median(g)
    return cv2.Canny(g, int(max(0, (1 - s) * m)), int(min(255, (1 + s) * m)))


def otsu_canny(g):
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(g, 0.5 * t, t)


# ----------------- 1 + 2. Danh gia & cai thien -----------------
gt = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), 50, 150)        # "canh chuan"
noisy = np.clip(gray.astype(np.float32) + np.random.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)

methods = {
    "Canny co dinh 50/150 (khong loc)": cv2.Canny(noisy, 50, 150),
    "Gauss 5x5 + 50/150": cv2.Canny(cv2.GaussianBlur(noisy, (5, 5), 1.4), 50, 150),
    "Gauss + nguong auto (median)": auto_canny(cv2.GaussianBlur(noisy, (5, 5), 1.4)),
    "Gauss + nguong Otsu": otsu_canny(cv2.GaussianBlur(noisy, (5, 5), 1.4)),
    "Bilateral + 50/150": cv2.Canny(cv2.bilateralFilter(noisy, 9, 75, 75), 50, 150),
    "NL-Means + 50/150": cv2.Canny(cv2.fastNlMeansDenoising(noisy, None, 15, 7, 21), 50, 150),
}
print("1-2. Danh gia tren anh nhieu (chuan = Canny cua anh sach, sai so 2 pixel)")
print(f"{'Phuong phap':<36}{'Precision':>10}{'Recall':>9}{'F1':>8}{'Pratt':>8}")
for k, e in methods.items():
    p, r, f1 = prf(e, gt)
    print(f"{k:<36}{p:>10.3f}{r:>9.3f}{f1:>8.3f}{pratt_fom(e, gt):>8.3f}")

res1 = luoi_anh([gt, noisy] + list(methods.values()),
                ["Canh chuan", "Anh nhieu"] + list(methods.keys()), cols=4, cell=(330, 260))
hien_thi_cv("Danh gia & cai thien Canny", res1, f"c2mr_{name}_danh_gia.png")

# ----------------- 3. Canny tren anh mau -----------------
b, g, r = cv2.split(cv2.GaussianBlur(img, (5, 5), 1.4))
e_gray = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), 50, 150)
e_or = cv2.bitwise_or(cv2.bitwise_or(cv2.Canny(b, 50, 150), cv2.Canny(g, 50, 150)),
                      cv2.Canny(r, 50, 150))
# Gradient lon nhat cua 3 kenh -> NMS don gian bang cach dua vao Canny tren kenh co gradient manh nhat
mags = []
for ch in (b, g, r):
    gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1)
    mags.append(cv2.magnitude(gx, gy))
mag_max = np.max(mags, axis=0)
idx = np.argmax(mags, axis=0)
stack = np.stack([b, g, r], axis=-1)
best = np.take_along_axis(stack, idx[..., None], axis=2)[..., 0]
e_max = cv2.Canny(best, 50, 150)

print("\n3. Canny tren anh mau - so pixel canh:")
for k, e in [("Chuyen xam", e_gray), ("Tung kenh BGR + OR", e_or), ("Kenh gradient lon nhat", e_max)]:
    print(f"   {k:<26}{np.count_nonzero(e):>8} pixel")
res3 = luoi_anh([img, e_gray, e_or, e_max],
                ["Anh mau", "Chuyen xam roi Canny", "Canny 3 kenh + OR", "Kenh gradient lon nhat"],
                cols=4, cell=(330, 260))
hien_thi_cv("Canny tren anh mau", res3, f"c2mr_{name}_anh_mau.png")
