"""
CHUONG 2 - Bai 2: Thay doi tham so Canny va quan sat ket qua.
  - sigma cua bo loc Gaussian
  - nguong thap / nguong cao
  - So sanh voi gia tri mac dinh (sigma=1.0, 100/200 cho OpenCV)

Chay:  python canny_khao_sat_tham_so.py
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

from common import lay_anh, hien_thi_matplotlib

img, name = lay_anh("Chon anh de khao sat tham so Canny", resize_max=700)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(gray, sigma, low, high):
    """Canny voi sigma dieu chinh duoc (OpenCV khong co tham so sigma nen lam mo truoc)."""
    k = int(6 * sigma + 1) | 1          # kich thuoc kernel le, ~ 6*sigma
    blur = cv2.GaussianBlur(gray, (k, k), sigma) if sigma > 0 else gray
    return cv2.Canny(blur, low, high)


def dem(e):
    return 100.0 * np.count_nonzero(e) / e.size


# ----- A. Anh huong cua sigma (co dinh 50/150) -----
sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
fig1, ax = plt.subplots(1, len(sigmas) + 1, figsize=(4 * (len(sigmas) + 1), 4.4))
ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Anh xam"); ax[0].axis("off")
print("A. Anh huong cua sigma (nguong 50/150)")
for a, s in zip(ax[1:], sigmas):
    e = canny(gray, s, 50, 150)
    a.imshow(e, cmap="gray"); a.axis("off")
    a.set_title(f"sigma={s}\n{dem(e):.1f}% pixel canh")
    print(f"   sigma={s:<4} -> {dem(e):5.2f}% pixel la canh")
plt.tight_layout()
hien_thi_matplotlib(fig1, f"c2b2_{name}_sigma.png")

# ----- B. Anh huong cua nguong thap / cao (sigma co dinh 1.4) -----
cap = [(10, 30), (30, 90), (50, 150), (100, 200), (150, 250), (30, 250)]
fig2, ax = plt.subplots(2, 3, figsize=(14, 9))
print("B. Anh huong cua nguong thap/cao (sigma=1.4)")
for a, (lo, hi) in zip(ax.ravel(), cap):
    e = canny(gray, 1.4, lo, hi)
    a.imshow(e, cmap="gray"); a.axis("off")
    a.set_title(f"low={lo}, high={hi}\n{dem(e):.1f}% pixel canh")
    print(f"   low={lo:<4} high={hi:<4} -> {dem(e):5.2f}%")
plt.tight_layout()
hien_thi_matplotlib(fig2, f"c2b2_{name}_nguong.png")

# ----- C. So sanh gia tri mac dinh va gia tri tu chinh (OpenCV + skimage) -----
mac_dinh_cv = cv2.Canny(gray, 100, 200)
mac_dinh_sk = feature.canny(gray)                          # sigma=1, nguong tu dong theo dtype
tu_chinh_cv = canny(gray, 1.4, 40, 120)
tu_chinh_sk = feature.canny(gray, sigma=2.0, low_threshold=30, high_threshold=90)

fig3, ax = plt.subplots(1, 4, figsize=(18, 5))
for a, im, t in [(ax[0], mac_dinh_cv, "OpenCV mac dinh (100/200)"),
                 (ax[1], tu_chinh_cv, "OpenCV tu chinh (sigma1.4, 40/120)"),
                 (ax[2], mac_dinh_sk, "skimage mac dinh (sigma=1)"),
                 (ax[3], tu_chinh_sk, "skimage tu chinh (sigma2, 30/90)")]:
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout()
hien_thi_matplotlib(fig3, f"c2b2_{name}_mac_dinh_vs_tu_chinh.png")

print("\nKet luan:")
print(" - sigma nho: giu nhieu chi tiet nhung nhieu canh gia; sigma lon: it nhieu, mat canh mong.")
print(" - nguong thap nho: nhieu canh yeu/nhieu; nguong cao lon: chi con canh rat manh.")
print(" - Quy tac thuong dung: high ~ 2-3 lan low.")
