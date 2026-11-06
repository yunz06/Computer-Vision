"""
common.py - Ham dung chung cho cac bai lab (giong phong cach cac file cu:
hop thoai chon anh bang tkinter, doc/ghi duong dan tieng Viet bang imdecode/imencode).

Neu may khong co tkinter hoac khong co man hinh (server), chuong trinh se:
  - doc duong dan tu dong lenh (sys.argv), hoac
  - dung anh mau tu sinh ra,
  - luu anh ra thu muc results thay vi hien thi cua so.
"""
import os
import sys
import numpy as np
import cv2

RESULTS_DIR = "results"
IMG_TYPES = [("Anh", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("Tat ca", "*.*")]
VIDEO_TYPES = [("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Tat ca", "*.*")]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _tk_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def chon_anh(title="Chon anh", argv_index=1):
    """Hop thoai chon 1 anh. Uu tien tham so dong lenh. Tra ve '' neu huy / khong co tkinter."""
    if len(sys.argv) > argv_index and os.path.isfile(sys.argv[argv_index]):
        return sys.argv[argv_index]
    try:
        from tkinter import filedialog
        root = _tk_root()
        path = filedialog.askopenfilename(title=title, filetypes=IMG_TYPES)
        root.destroy()
        return path
    except Exception:
        return ""


def chon_nhieu_anh(title="Chon nhieu anh"):
    """Hop thoai chon nhieu anh. Neu co tham so dong lenh thi dung cac tham so la file."""
    files = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if files:
        return files
    try:
        from tkinter import filedialog
        root = _tk_root()
        paths = filedialog.askopenfilenames(title=title, filetypes=IMG_TYPES)
        root.destroy()
        return list(paths)
    except Exception:
        return []


def chon_thu_muc(title="Chon thu muc chua anh"):
    """Hop thoai chon thu muc. Tra ve danh sach file anh ben trong."""
    folder = ""
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
    else:
        try:
            from tkinter import filedialog
            root = _tk_root()
            folder = filedialog.askdirectory(title=title)
            root.destroy()
        except Exception:
            folder = ""
    if not folder:
        return "", []
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXT)
    )
    return folder, files


def chon_video(title="Chon video"):
    """Hop thoai chon file video. Tra ve '' neu huy."""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    try:
        from tkinter import filedialog
        root = _tk_root()
        path = filedialog.askopenfilename(title=title, filetypes=VIDEO_TYPES)
        root.destroy()
        return path
    except Exception:
        return ""


def doc_anh(path, flags=cv2.IMREAD_COLOR):
    """Doc anh ho tro duong dan co dau tieng Viet."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def luu_anh(path, img):
    """Luu anh (duong dan Unicode). Tu tao thu muc cha."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def ket_qua(name):
    """Duong dan file trong thu muc results (tu tao thu muc)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def anh_mau(size=400):
    """Anh mau tu sinh: hinh hoc + chu + vai chi tiet de thu Canny / phan doan."""
    img = np.full((size, size, 3), 230, np.uint8)
    cv2.rectangle(img, (40, 40), (170, 170), (0, 0, 200), -1)
    cv2.circle(img, (290, 110), 65, (0, 150, 0), -1)
    pts = np.array([[80, 330], [200, 220], [320, 330]], np.int32)
    cv2.fillPoly(img, [pts], (200, 100, 0))
    cv2.ellipse(img, (330, 300), (50, 28), 20, 0, 360, (0, 180, 220), -1)
    cv2.putText(img, "Lab", (150, 390), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
    rng = np.random.default_rng(0)
    for _ in range(40):
        x, y = rng.integers(0, size, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)), (90, 90, 90), -1)
    return img


def lay_anh(title="Chon anh", resize_max=None):
    """Chon anh bang hop thoai; neu khong chon duoc thi dung anh mau.
    Tra ve (img_bgr, ten_hien_thi)."""
    path = chon_anh(title)
    if path:
        img = doc_anh(path)
        if img is None:
            sys.exit(f"Khong doc duoc anh: {path}")
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        print("Khong chon anh -> dung anh mau.")
        img = anh_mau()
        name = "anh_mau"
    if resize_max:
        h, w = img.shape[:2]
        s = min(1.0, resize_max / max(h, w))
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, name


def hien_thi_matplotlib(fig, save_name=None, show=True):
    """Luu figure matplotlib vao results va hien thi (neu co man hinh)."""
    import matplotlib.pyplot as plt
    if save_name:
        fig.savefig(ket_qua(save_name), dpi=130, bbox_inches="tight")
        print("Da luu:", ket_qua(save_name))
    if show and os.environ.get("LAB_NO_SHOW") != "1":
        plt.show()
    plt.close(fig)


def hien_thi_cv(title, img, save_name=None, max_w=1300, max_h=750):
    """Luu anh vao results va hien cua so OpenCV (tu thu nho cho vua man hinh)."""
    if save_name:
        luu_anh(ket_qua(save_name), img)
        print("Da luu:", ket_qua(save_name))
    if os.environ.get("LAB_NO_SHOW") == "1":
        return
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    try:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("(Khong co man hinh de hien cua so - chi luu file.)")


def gan_nhan(img, text, h=28):
    """Ghi tieu de len anh (giong ham label trong loc_trung_binh.py). Tra ve anh BGR."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def luoi_anh(imgs, titles, cols=3, cell=(380, 300)):
    """Ghep nhieu anh (xam/mau) thanh luoi co tieu de, de hien thi/luu bang OpenCV."""
    cells = []
    for im, t in zip(imgs, titles):
        im = cv2.resize(im, cell, interpolation=cv2.INTER_AREA)
        cells.append(gan_nhan(im, t))
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)
