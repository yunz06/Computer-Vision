"""
CHUONG 2 - Bai 3: Ap dung Canny cho cac loai anh khac nhau:
  - anh nhieu, anh do tuong phan thap, anh nhieu chi tiet.

Cach dung: hop thoai se hoi 3 anh (nhieu / tuong phan thap / nhieu chi tiet).
Neu bam Cancel o buoc nao, chuong trinh tu tao anh do tu mot anh goc
(them nhieu, giam tuong phan) de ban van co ket qua.

Chay:  python canny_cac_loai_anh.py
"""
import cv2
import numpy as np

from common import (chon_anh, doc_anh, lay_anh, anh_mau, luoi_anh,
                    hien_thi_cv, luu_anh, ket_qua)

import sys
sys.argv = sys.argv[:1]     # tranh dung tham so dong lenh cho tung lan chon


def them_nhieu(g, sigma=25):
    n = g.astype(np.float32) + np.random.normal(0, sigma, g.shape)
    return np.clip(n, 0, 255).astype(np.uint8)


def giam_tuong_phan(g, alpha=0.3):
    return np.clip(128 + alpha * (g.astype(np.float32) - 128), 0, 255).astype(np.uint8)


def doc_hoac_none(tieu_de):
    p = chon_anh(tieu_de)
    if not p:
        return None
    im = doc_anh(p)
    return im


np.random.seed(0)
goc, name = lay_anh("Chon ANH GOC (dung de tu tao cac anh thieu)", resize_max=600)
gray0 = cv2.cvtColor(goc, cv2.COLOR_BGR2GRAY)

loai = {}
for key, td, tao in [
    ("Nhieu", "Chon anh CO NHIEU (Cancel = tu tao)", lambda: them_nhieu(gray0)),
    ("Tuong phan thap", "Chon anh TUONG PHAN THAP (Cancel = tu tao)", lambda: giam_tuong_phan(gray0)),
    ("Nhieu chi tiet", "Chon anh NHIEU CHI TIET (Cancel = dung anh goc)", lambda: gray0),
]:
    im = doc_hoac_none(td)
    loai[key] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im is not None else tao()
    loai[key] = cv2.resize(loai[key], (500, 380))


def canny_tu_dong(g, sigma=0.33, blur_k=5):
    """Nguong tu dong theo trung vi: low=(1-s)*median, high=(1+s)*median."""
    b = cv2.GaussianBlur(g, (blur_k, blur_k), 0)
    m = np.median(b)
    lo, hi = int(max(0, (1 - sigma) * m)), int(min(255, (1 + sigma) * m))
    return cv2.Canny(b, lo, hi), lo, hi


def ty_le(e):
    return 100.0 * np.count_nonzero(e) / e.size


hang = []
print(f"{'Loai anh':<18}{'Khong loc (%)':>15}{'Co loc+auto (%)':>18}   nguong auto")
for key, g in loai.items():
    e_raw = cv2.Canny(g, 100, 200)                       # khong loc nhieu, nguong co dinh
    e_auto, lo, hi = canny_tu_dong(g)
    print(f"{key:<18}{ty_le(e_raw):>15.2f}{ty_le(e_auto):>18.2f}   {lo}/{hi}")
    hang.append((g, f"{key}: anh xam"))
    hang.append((e_raw, f"{key}: Canny 100/200"))
    hang.append((e_auto, f"{key}: Gauss + nguong auto {lo}/{hi}"))

res = luoi_anh([h[0] for h in hang], [h[1] for h in hang], cols=3, cell=(420, 320))
luu_anh(ket_qua("c2b3_cac_loai_anh.png"), res)
hien_thi_cv("Canny tren cac loai anh", res, "c2b3_cac_loai_anh.png")

print("\nNhan xet:")
print(" - Anh nhieu: Canny khong loc cho rat nhieu canh gia -> nen Gaussian blur (sigma lon hon).")
print(" - Tuong phan thap: nguong co dinh bo sot canh -> dung nguong tu dong theo median.")
print(" - Nhieu chi tiet: canh day dac -> tang sigma / nguong cao de giu canh quan trong.")
