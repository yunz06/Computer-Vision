"""
CHUONG 4 - Phan doan anh (Image Segmentation).
Chay lan luot nhieu phuong phap tren cung 1 anh va so sanh:
  1. Nguong toan cuc (Otsu)              4. Watershed (phan doan doi tuong dinh nhau)
  2. Nguong thich nghi (adaptive)        5. GrabCut (tach tien canh - hop thoai ve khung neu co man hinh)
  3. K-means theo mau (K = 2..4)         6. Phan doan theo canh Canny + contour
Ket qua: 1 anh luoi tong hop + tung phuong phap luu rieng trong results/.

Chay:  python phan_doan_anh_cac_phuong_phap.py            (hop thoai chon anh)
       python phan_doan_anh_cac_phuong_phap.py anh.jpg
Phim tat khi GrabCut mo cua so chon vung: keo chuot ve khung bao doi tuong, ENTER/SPACE de xac nhan,
bam 'c' de dung khung mac dinh (o giua anh).
"""
import os
import cv2
import numpy as np

from common import lay_anh, luoi_anh, hien_thi_cv, luu_anh, ket_qua

img, name = lay_anh("Chon anh de phan doan", resize_max=700)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ket = {}


def to_mau_nhan(labels, n=None):
    """To mau ngau nhien (co dinh seed) cho anh nhan."""
    n = n or int(labels.max()) + 1
    rng = np.random.default_rng(3)
    pal = rng.integers(40, 255, (n + 1, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)
    return pal[labels.astype(np.int32).clip(0, n)]


# ---------- 1. Otsu ----------
t_otsu, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ket[f"1. Otsu (T={t_otsu:.0f})"] = otsu

# ---------- 2. Adaptive ----------
adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5)
ket["2. Adaptive threshold"] = adapt

# ---------- 3. K-means ----------
Z = img.reshape(-1, 3).astype(np.float32)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
for K in (2, 3, 4):
    cv2.setRNGSeed(0)
    _, lab, cen = cv2.kmeans(Z, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    seg = cen.astype(np.uint8)[lab.flatten()].reshape(img.shape)
    ket[f"3. K-means K={K}"] = seg

# ---------- 4. Watershed ----------
_, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
if np.count_nonzero(bw) > 0.6 * bw.size:          # nen sang hay toi -> dam bao doi tuong la vung trang
    bw = cv2.bitwise_not(bw)
k3 = np.ones((3, 3), np.uint8)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=2)
sure_bg = cv2.dilate(bw, k3, iterations=3)
dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)
n_lab, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
ws = cv2.watershed(img.copy(), markers)
ws_img = img.copy()
ws_img[ws == -1] = (0, 0, 255)                      # duong bien do
ws_color = to_mau_nhan(np.where(ws > 1, ws - 1, 0))
ket[f"4. Watershed ({n_lab - 1} vung)"] = cv2.addWeighted(ws_img, 0.6, ws_color, 0.4, 0)

# ---------- 5. GrabCut ----------
rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))     # mac dinh: khung giua anh
if os.environ.get("LAB_NO_SHOW") != "1":
    try:
        print("GrabCut: keo khung quanh doi tuong, ENTER de xac nhan, 'c' de dung khung mac dinh.")
        r = cv2.selectROI("Chon vung chua doi tuong (GrabCut)", img, showCrosshair=False)
        cv2.destroyAllWindows()
        if r[2] > 10 and r[3] > 10:
            rect = tuple(int(v) for v in r)
    except cv2.error:
        pass
mask = np.zeros((h, w), np.uint8)
bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
m2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
grab = img * m2[:, :, None]
cv2.rectangle(grab, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
ket["5. GrabCut (tien canh)"] = grab

# ---------- 6. Canny + contour ----------
edges = cv2.Canny(blur, 40, 120)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = [c for c in cnts if cv2.contourArea(c) > 0.002 * gray.size]
cimg = np.zeros_like(img)
rng = np.random.default_rng(5)
for c in cnts:
    cv2.drawContours(cimg, [c], -1, tuple(int(v) for v in rng.integers(60, 255, 3)), cv2.FILLED)
ket[f"6. Canny + contour ({len(cnts)} vung)"] = cv2.addWeighted(img, 0.5, cimg, 0.5, 0)

# ---------- Hien thi + luu ----------
imgs = [img] + list(ket.values())
titles = ["Anh goc"] + list(ket.keys())
res = luoi_anh(imgs, titles, cols=4, cell=(360, 280))
hien_thi_cv("Phan doan anh - so sanh phuong phap", res, f"c4_phan_doan_{name}.png")

for i, (t, im) in enumerate(ket.items(), 1):
    safe = "".join(ch if ch.isalnum() else "_" for ch in t)[:40]
    luu_anh(ket_qua(f"c4_pd_{name}_{safe}.png"), im)

print("\nNhan xet nhanh:")
print(" - Otsu: nhanh, tot khi anh co 2 nhom muc xam ro; kem khi anh sang khong deu.")
print(" - Adaptive: nguong theo vung nen xu ly duoc sang khong deu, nhung nhieu nhieu.")
print(" - K-means: phan theo mau, khong dung thong tin khong gian.")
print(" - Watershed: tach duoc cac vat dinh nhau nhung de qua phan manh neu anh nhieu.")
print(" - GrabCut: tach tien canh chinh xac nhat nhung can khung ban dau.")
