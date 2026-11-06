"""
CHUONG 2 - Cau hoi mo rong 4: Ap dung Canny cho video.

Chon file video bang hop thoai (Cancel = dung webcam; neu khong co webcam -> tao video mau).
Co 2 thanh truot (trackbar) chinh nguong low/high theo thoi gian thuc.
Phim tat: q/ESC thoat | s luu khung hien tai | space tam dung.
Ket qua video luu tai results/canny_video.mp4

Chay:  python canny_video_webcam.py            python canny_video_webcam.py video.mp4
"""
import os
import cv2
import numpy as np

from common import chon_video, ket_qua, luu_anh

LAB_NO_SHOW = os.environ.get("LAB_NO_SHOW") == "1"


def video_mau(n=90, size=(480, 360)):
    """Sinh video mau: hinh tron + hinh vuong chuyen dong (khi khong co video/webcam)."""
    w, h = size
    for i in range(n):
        f = np.full((h, w, 3), 220, np.uint8)
        cv2.circle(f, (60 + i * 4, 120), 40, (0, 0, 200), -1)
        cv2.rectangle(f, (300 - i * 2, 200), (380 - i * 2, 280), (200, 80, 0), -1)
        cv2.putText(f, "Video", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
        yield f


def nguon_khung():
    path = chon_video("Chon video (Cancel = webcam)")
    if path:
        cap = cv2.VideoCapture(path)
        print("Doc video:", path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        return gen(), fps
    cap = cv2.VideoCapture(0)
    if cap.isOpened() and not LAB_NO_SHOW:
        print("Dung webcam.")
        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
        return gen(), 25
    print("Khong co video/webcam -> dung video mau.")
    return video_mau(), 25


def nhap_nhay_nguong(low, high):
    return min(low, high - 1), max(high, low + 1)


frames, fps = nguon_khung()
cua_so = "Canny video (goc | canny)"
if not LAB_NO_SHOW:
    cv2.namedWindow(cua_so, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("low", cua_so, 50, 255, lambda v: None)
    cv2.createTrackbar("high", cua_so, 150, 255, lambda v: None)

writer = None
os.makedirs("results", exist_ok=True)
n = 0
for frame in frames:
    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
    if not LAB_NO_SHOW:
        try:
            lo, hi = cv2.getTrackbarPos("low", cua_so), cv2.getTrackbarPos("high", cua_so)
        except cv2.error:
            lo, hi = 50, 150
    else:
        lo, hi = 50, 150
    lo, hi = nhap_nhay_nguong(lo, hi)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), lo, hi)
    out = np.hstack([frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
    cv2.putText(out, f"low={lo} high={hi}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    if writer is None:
        writer = cv2.VideoWriter(ket_qua("canny_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (out.shape[1], out.shape[0]))
    writer.write(out)
    n += 1

    if not LAB_NO_SHOW:
        cv2.imshow(cua_so, out)
        k = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("s"):
            luu_anh(ket_qua(f"canny_khung_{n}.png"), out)
            print("Da luu khung", n)
        if k == ord(" "):
            cv2.waitKey(0)

if writer:
    writer.release()
cv2.destroyAllWindows()
print(f"Xong {n} khung. Video ket qua: {ket_qua('canny_video.mp4')}")
