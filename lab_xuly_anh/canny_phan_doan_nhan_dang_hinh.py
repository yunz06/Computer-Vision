"""
CHUONG 2 - Bai 4: Ket hop Canny voi ky thuat khac.
  (a) Canny + phan doan: dong canh (morphology) -> findContours -> to mau tung vung.
  (b) Canny + nhan dang hinh dang: approxPolyDP (tam giac/vuong/chu nhat/da giac),
      HoughCircles (hinh tron) -> tim doi tuong trong anh.

Chay:  python canny_phan_doan_nhan_dang_hinh.py
"""
import cv2
import numpy as np

from common import lay_anh, luoi_anh, hien_thi_cv

img, name = lay_anh("Chon anh co cac vat the don gian (hinh hoc)", resize_max=700)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blur, 40, 120)

# ---------- (a) Phan doan vung bang Canny ----------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)   # noi cac net dut
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 0.002 * gray.size
contours = [c for c in contours if cv2.contourArea(c) > min_area]

seg = np.zeros_like(img)
rng = np.random.default_rng(1)
for c in contours:
    col = tuple(int(v) for v in rng.integers(60, 255, 3))
    cv2.drawContours(seg, [c], -1, col, thickness=cv2.FILLED)
overlay = cv2.addWeighted(img, 0.5, seg, 0.5, 0)
cv2.drawContours(overlay, contours, -1, (0, 0, 0), 2)
print(f"(a) Tim duoc {len(contours)} vung (dien tich > {int(min_area)} px)")

# ---------- (b) Nhan dang hinh dang ----------
shape_img = img.copy()
dem_hinh = {}
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    n = len(approx)
    area = cv2.contourArea(c)
    circularity = 4 * np.pi * area / (peri * peri + 1e-9)
    if n == 3:
        label = "Tam giac"
    elif n == 4:
        x, y, w, h = cv2.boundingRect(approx)
        label = "Vuong" if 0.9 <= w / float(h) <= 1.1 else "Chu nhat"
    elif circularity > 0.8:
        label = "Tron"
    elif n > 4:
        label = f"Da giac/{n}"
    else:
        label = "Khac"
    dem_hinh[label] = dem_hinh.get(label, 0) + 1
    cv2.drawContours(shape_img, [approx], -1, (0, 0, 255), 2)
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        cv2.putText(shape_img, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(shape_img, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
print("(b) Thong ke hinh dang:", dem_hinh)

# Hough circles (dung ket hop voi Canny: param1 = nguong cao cua Canny noi bo)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                           param1=120, param2=35, minRadius=10, maxRadius=200)
hough_img = img.copy()
if circles is not None:
    for x, y, r in np.round(circles[0]).astype(int):
        cv2.circle(hough_img, (x, y), r, (0, 255, 0), 3)
        cv2.circle(hough_img, (x, y), 3, (0, 0, 255), -1)
    print(f"    HoughCircles phat hien {len(circles[0])} duong tron")
else:
    print("    HoughCircles khong thay duong tron nao")

res = luoi_anh(
    [img, edges, closed, overlay, shape_img, hough_img],
    ["Anh goc", "Canny", "Canny sau khi dong (morphology)",
     "Phan doan theo contour", "Nhan dang hinh dang", "HoughCircles"],
    cols=3, cell=(420, 330))
hien_thi_cv("Canny ket hop phan doan / nhan dang", res, f"c2b4_{name}_ket_hop.png")
