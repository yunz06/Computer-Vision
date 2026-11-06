"""
CHUONG 4 - Bai nang cao 2: Ung dung TIM KIEM ANH dua tren ham bam wavelet.

Cach dung:
  1. Hop thoai 1: chon THU MUC "co so du lieu" anh -> chuong trinh tinh ma bam cho tat ca anh.
  2. Hop thoai 2: chon ANH TRUY VAN -> hien thi top-K anh gan nhat (khoang cach Hamming nho nhat).
  3. Lap lai hop thoai 2 cho den khi bam Cancel.
Ma bam duoc luu cache o results/c4_index.npz de lan sau chay nhanh hon.

Chay:  python tim_kiem_anh_wavelet_hash.py
"""
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import chon_thu_muc, chon_anh, doc_anh, ket_qua, hien_thi_matplotlib
import hash_lib as H

TOP_K = 5

folder, files = chon_thu_muc("Chon thu muc CO SO DU LIEU anh (Cancel = du lieu tong hop)")
db = []
for f in files:
    im = doc_anh(f)
    if im is not None:
        db.append((os.path.basename(f), im))
if not db:
    print("Khong co thu muc -> dung 30 anh tong hop.")
    db = [(f"tong_hop_{i:02d}", H.anh_tong_hop(i)) for i in range(30)]

print(f"Tinh ma bam cho {len(db)} anh ...")
hashes = np.array([H.wavelet_hash(im) for _, im in db])
os.makedirs("results", exist_ok=True)
np.savez(ket_qua("c4_index.npz"), ten=np.array([n for n, _ in db]), hashes=hashes)


def tim(query_img, k=TOP_K):
    q = H.wavelet_hash(query_img)
    d = np.count_nonzero(hashes != q, axis=1)
    order = np.argsort(d)[:k]
    return [(int(i), int(d[i])) for i in order]


def ve_ket_qua(query_img, ten_query, ket_qua_tim, luu=None):
    k = len(ket_qua_tim)
    fig, ax = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.6))
    ax[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)); ax[0].set_title("TRUY VAN\n" + ten_query[:20])
    ax[0].axis("off")
    for a, (i, d) in zip(ax[1:], ket_qua_tim):
        a.imshow(cv2.cvtColor(db[i][1], cv2.COLOR_BGR2RGB))
        a.set_title(f"{db[i][0][:18]}\nHamming={d}")
        a.axis("off")
    plt.tight_layout()
    hien_thi_matplotlib(fig, luu)


lan = 0
queries = []
if len(sys.argv) > 2:                    # co the truyen san: thu_muc anh_truy_van
    queries = [p for p in sys.argv[2:] if os.path.isfile(p)]

while True:
    if queries:
        p = queries.pop(0)
    else:
        sys.argv = sys.argv[:1]
        p = chon_anh("Chon ANH TRUY VAN (Cancel = thoat)")
    if p:
        q = doc_anh(p)
        ten_q = os.path.basename(p)
    else:
        if lan > 0:
            break
        # khong chon duoc -> lay chinh 1 anh trong CSDL va bien doi de demo
        rng = np.random.default_rng(7)
        q, loai = H.bien_doi_ngau_nhien(db[0][1], rng)
        ten_q = f"demo({db[0][0]}+{loai})"
    if q is None:
        print("Khong doc duoc anh truy van.")
        continue
    lan += 1
    kq = tim(q)
    print(f"\nTruy van: {ten_q}")
    for r, (i, d) in enumerate(kq, 1):
        print(f"  #{r}: {db[i][0]:<30} Hamming = {d}")
    ve_ket_qua(q, ten_q, kq, f"c4_tim_kiem_{lan}.png")
    if not p:
        break
print("Ket thuc.")
