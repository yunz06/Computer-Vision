"""
CHUONG 4 - Nhan dien khuon mat thoi gian thuc tu webcam: OpenCV + MTCNN + FaceNet.

Cac buoc chinh:
  1. Truy cap webcam bang OpenCV (cv2.VideoCapture)
  2. MTCNN (facenet-pytorch) phat hien khuon mat
  3. FaceNet (InceptionResnetV1, pretrained vggface2) trich dac trung 512 chieu,
     so sanh khoang cach L2 voi cac khuon mat da dang ky -> ten nguoi / "Unknown"

CAI DAT (1 lan):
    pip install torch torchvision facenet-pytorch opencv-python
(Lan chay dau facenet-pytorch tu tai trong so pretrained ~100MB - can Internet.)
Neu CHUA cai duoc -> chuong trinh van chay o che do Haar Cascade (chi phat hien, khong nhan dien).

CACH DUNG:
  - Khi chay: hop thoai hoi THU MUC khuon mat da biet. Cau truc:
        thu_muc/An/1.jpg, 2.jpg      (moi nguoi 1 thu muc con, ten thu muc = ten nguoi)
    hoac thu_muc/An.jpg, Binh.jpg    (ten file = ten nguoi)
    Cancel = khong nap, ban dang ky bang phim.
  - Phim tat trong cua so webcam:
        e : dang ky khuon mat dang thay (hop thoai nhap ten)
        + / - : tang / giam nguong khoang cach (mac dinh 0.9)
        s : luu khung hinh      q hoac ESC : thoat
  - Co the chay tren video / anh thay webcam:  python nhan_dien_khuon_mat_mtcnn_facenet.py video.mp4
"""
import os
import sys
import time
import numpy as np
import cv2

from common import chon_thu_muc, doc_anh, ket_qua, luu_anh, IMG_EXT

# ----------- Nap model (neu co) -----------
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    CO_FACENET = True
except Exception as e:                         # chua cai torch / facenet-pytorch
    CO_FACENET = False
    print("[!] Chua co torch/facenet-pytorch -> che do Haar Cascade (chi phat hien).")
    print("    Cai dat: pip install torch torchvision facenet-pytorch")

DEVICE = "cuda" if CO_FACENET and torch.cuda.is_available() else "cpu"
if CO_FACENET:
    mtcnn = MTCNN(keep_all=True, device=DEVICE, post_process=True, min_face_size=60)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
else:
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

NGUONG = 0.9                                   # khoang cach L2 toi da de coi la cung nguoi
gallery = {}                                   # ten -> list vector 512 chieu


def phat_hien(frame_bgr):
    """Tra ve (list box [x1,y1,x2,y2], list tensor khuon mat da cat) - tensor None neu Haar."""
    if CO_FACENET:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        if boxes is None:
            return [], []
        keep = [i for i, p in enumerate(probs) if p is not None and p > 0.9]
        boxes = [boxes[i] for i in keep]
        faces = mtcnn.extract(rgb, np.array(boxes), None) if boxes else []
        return [b.astype(int) for b in boxes], list(faces) if len(boxes) else []
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    r = haar.detectMultiScale(g, 1.2, 5, minSize=(60, 60))
    return [np.array([x, y, x + w, y + h]) for x, y, w, h in r], [None] * len(r)


def embedding(face_tensor):
    with torch.no_grad():
        e = facenet(face_tensor.unsqueeze(0).to(DEVICE))
    return e[0].cpu().numpy()


def dang_ky(ten, face_tensor):
    gallery.setdefault(ten, []).append(embedding(face_tensor))
    print(f"Da dang ky '{ten}' (tong {len(gallery[ten])} mau).")


def nhan_dien(emb):
    """Tra ve (ten, khoang_cach_nho_nhat)."""
    best, bd = "Unknown", 1e9
    for ten, lst in gallery.items():
        d = min(np.linalg.norm(emb - e) for e in lst)
        if d < bd:
            best, bd = ten, d
    return (best if bd < NGUONG else "Unknown"), bd


def nap_thu_muc(folder):
    """Nap khuon mat tu thu muc (cau truc nhu mo ta o dau file)."""
    if not CO_FACENET:
        return
    muc = []
    for item in sorted(os.listdir(folder)):
        p = os.path.join(folder, item)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(IMG_EXT):
                    muc.append((item, os.path.join(p, f)))
        elif item.lower().endswith(IMG_EXT):
            muc.append((os.path.splitext(item)[0], p))
    for ten, path in muc:
        im = doc_anh(path)
        if im is None:
            continue
        boxes, faces = phat_hien(im)
        if faces:
            # lay khuon mat lon nhat
            i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            dang_ky(ten, faces[i])
        else:
            print(f"  (khong thay mat trong {os.path.basename(path)})")
    print(f"Da nap {len(gallery)} nguoi.")


def hoi_ten():
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        t = simpledialog.askstring("Dang ky", "Nhap ten nguoi:", parent=root)
        root.destroy()
        return t
    except Exception:
        return input("Nhap ten nguoi: ").strip() or None


def nguon():
    """Webcam mac dinh; hoac video/anh truyen qua dong lenh."""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return cv2.VideoCapture(sys.argv[1]), os.path.basename(sys.argv[1])
    return cv2.VideoCapture(0), "webcam"


def main():
    global NGUONG
    if CO_FACENET:
        folder, _ = chon_thu_muc("Chon thu muc khuon mat da biet (Cancel = bo qua)")
        if folder:
            nap_thu_muc(folder)

    cap, ten_nguon = nguon()
    if not cap.isOpened():
        print(f"Khong mo duoc {ten_nguon}. Kiem tra webcam hoac truyen duong dan video.")
        return
    print("Phim: e = dang ky | +/- = nguong | s = luu | q = thoat")

    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, faces = phat_hien(frame)
        for b, f in zip(boxes, faces):
            if CO_FACENET:
                ten, d = nhan_dien(embedding(f)) if gallery else ("Chua co du lieu", 0)
                nhan = f"{ten} ({d:.2f})" if gallery else ten
                mau = (0, 200, 0) if ten not in ("Unknown", "Chua co du lieu") else (0, 0, 255)
            else:
                nhan, mau = "Face", (255, 200, 0)
            x1, y1, x2, y2 = b
            cv2.rectangle(frame, (x1, y1), (x2, y2), mau, 2)
            cv2.putText(frame, nhan, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now
        info = f"{'MTCNN+FaceNet' if CO_FACENET else 'Haar'} | FPS {fps:.1f} | nguong {NGUONG:.2f} | {len(boxes)} mat"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Face recognition - q de thoat", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("s"):
            luu_anh(ket_qua(f"face_{int(time.time())}.png"), frame)
            print("Da luu khung hinh.")
        elif k in (ord("+"), ord("=")):
            NGUONG += 0.05
        elif k in (ord("-"), ord("_")):
            NGUONG = max(0.1, NGUONG - 0.05)
        elif k == ord("e"):
            if not CO_FACENET:
                print("Can facenet-pytorch de dang ky.")
            elif not faces:
                print("Khong thay khuon mat de dang ky.")
            else:
                i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
                ten = hoi_ten()
                if ten:
                    dang_ky(ten, faces[i])
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
