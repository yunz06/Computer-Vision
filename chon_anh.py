"""
Module dùng chung: chọn ảnh bằng hộp thoại và lưu ảnh an toàn.
Các file khác chỉ cần:  from chon_anh import chon_anh, luu_anh
"""
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog


def chon_anh(title="Chọn ảnh", mode=cv2.IMREAD_COLOR):
    """Mở hộp thoại cho người dùng chọn ảnh, trả về (ảnh, đường dẫn).

    - mode = cv2.IMREAD_COLOR      -> ảnh màu BGR
    - mode = cv2.IMREAD_GRAYSCALE  -> ảnh xám
    """
    # Tạo cửa sổ tkinter ẩn, chỉ dùng để hiện hộp thoại chọn file
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # đưa hộp thoại lên trên cùng

    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                   ("Tất cả", "*.*")],
    )
    root.destroy()

    # Người dùng bấm Cancel -> dừng chương trình
    if not path:
        raise SystemExit("Bạn chưa chọn ảnh!")

    # cv2.imread không đọc được đường dẫn có dấu tiếng Việt,
    # nên đọc bằng np.fromfile rồi giải mã bằng cv2.imdecode
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {path}")

    return img, path


def luu_anh(ten_file, img, thu_muc="results"):
    """Lưu ảnh vào thư mục (tự tạo nếu chưa có). Hỗ trợ đường dẫn Unicode."""
    os.makedirs(thu_muc, exist_ok=True)
    duong_dan = os.path.join(thu_muc, ten_file)
    ext = os.path.splitext(ten_file)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(duong_dan)
        print(f"Đã lưu: {duong_dan}")
    return duong_dan
