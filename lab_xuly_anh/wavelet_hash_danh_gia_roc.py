"""
CHUONG 4 - Bai tap: Ham bam wavelet (wavelet hash) de xac dinh anh tuong thich.

Cac buoc:
  1. Chuan bi du lieu : chon THU MUC anh -> tu tao cap tuong tu (cung anh + nhieu/xoay/sang...)
                        va cap khong tuong tu (2 anh khac nhau).
                        (hoac dat file pairs.csv trong thu muc: path1,path2,label  label=1 tuong tu, 0 khong)
  2-3. Trich xuat he so wavelet -> luong tu hoa -> ma bam 64 bit
  4. So sanh: khoang cach Hamming
  5. Danh gia: Accuracy, Sensitivity (do nhay), Specificity (do dac hieu), duong cong ROC + AUC
  Nang cao 1: so sanh cac phuong phap bam (wavelet-LL, wavelet-detail, aHash, dHash, pHash)

Chay:  python wavelet_hash_danh_gia_roc.py             (hop thoai chon thu muc anh)
       python wavelet_hash_danh_gia_roc.py thu_muc_anh
Ket qua trong thu muc results/: c4_cap_anh.csv, c4_roc.png, c4_so_sanh_phuong_phap.png, c4_bao_cao.txt
"""
import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import chon_thu_muc, doc_anh, hien_thi_matplotlib, ket_qua, luu_anh
import hash_lib as H

SEED = 42
rng = np.random.default_rng(SEED)

# ----------------------- 1. Chuan bi du lieu -----------------------
folder, files = chon_thu_muc("Chon thu muc chua anh (Cancel = dung du lieu tong hop)")
anh = []                       # list (ten, anh BGR)
for f in files:
    im = doc_anh(f)
    if im is not None:
        anh.append((os.path.basename(f), cv2.resize(im, (256, 256))))

if len(anh) < 2:
    print("Khong co (du) anh -> sinh 40 anh tong hop de demo.")
    anh = [(f"tong_hop_{i:02d}", H.anh_tong_hop(i)) for i in range(40)]

print(f"So anh goc: {len(anh)}")

cap = []                        # (anh1, anh2, nhan, mo_ta)
csv_pairs = os.path.join(folder, "pairs.csv") if folder else ""
if csv_pairs and os.path.isfile(csv_pairs):
    print("Dung cap anh tu pairs.csv")
    with open(csv_pairs, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 3 or row[2].strip() not in ("0", "1"):
                continue
            a, b = doc_anh(os.path.join(folder, row[0])), doc_anh(os.path.join(folder, row[1]))
            if a is not None and b is not None:
                cap.append((cv2.resize(a, (256, 256)), cv2.resize(b, (256, 256)),
                            int(row[2]), "csv"))
else:
    # cap tuong tu: moi anh x 2 bien doi khac nhau
    for ten, im in anh:
        for _ in range(2):
            bd, loai = H.bien_doi_ngau_nhien(im, rng)
            cap.append((im, bd, 1, loai))
    # cap khong tuong tu: so luong xap xi bang cap tuong tu
    n_pos = len(cap)
    n = len(anh)
    seen = set()
    while len(seen) < min(n_pos, n * (n - 1) // 2):
        i, j = rng.choice(n, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        cap.append((anh[i][1], anh[j][1], 0, "khac_anh"))

y = np.array([c[2] for c in cap])
print(f"Tong so cap: {len(cap)}  (tuong tu={int(y.sum())}, khong tuong tu={int((1 - y).sum())})")


# ----------------------- Ham danh gia -----------------------
def roc_tu_khoang_cach(d, y):
    """d: khoang cach Hamming (nho = giong). Tra ve fpr, tpr, nguong, auc."""
    ths = np.arange(0, d.max() + 2)
    tpr, fpr = [], []
    for t in ths:
        pred = d <= t
        tpr.append(np.sum(pred & (y == 1)) / max(1, np.sum(y == 1)))
        fpr.append(np.sum(pred & (y == 0)) / max(1, np.sum(y == 0)))
    tpr, fpr = np.array(tpr), np.array(fpr)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, ths, auc


def chi_so(d, y, t):
    pred = d <= t
    tp, tn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
    fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    acc = (tp + tn) / len(y)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return acc, sens, spec, (tp, tn, fp, fn)


def tinh_khoang_cach(ham_bam):
    return np.array([H.hamming(ham_bam(a), ham_bam(b)) for a, b, _, _ in cap])


# ----------------------- 2-5. Wavelet hash chinh -----------------------
print(f"\nPyWavelets: {'co' if H.CO_PYWT else 'chua cai (dung Haar viet bang numpy)'}")
d_main = tinh_khoang_cach(H.wavelet_hash)
fpr, tpr, ths, auc = roc_tu_khoang_cach(d_main, y)
best = int(np.argmax(tpr - fpr))                          # nguong Youden
t_best = int(ths[best])
acc, sens, spec, (tp, tn, fp, fn) = chi_so(d_main, y, t_best)

dong = []
dong.append("BAO CAO WAVELET HASH (hash 8x8 = 64 bit)")
dong.append(f"So cap: {len(cap)} | tuong tu: {int(y.sum())} | khong tuong tu: {int((1 - y).sum())}")
dong.append(f"Khoang cach Hamming TB - cap tuong tu: {d_main[y == 1].mean():.2f}, "
            f"cap khac nhau: {d_main[y == 0].mean():.2f}")
dong.append(f"Nguong toi uu (Youden) T = {t_best}  (tuong tu neu Hamming <= T)")
dong.append(f"Accuracy (do chinh xac)   = {acc:.3f}")
dong.append(f"Sensitivity (do nhay)     = {sens:.3f}   (TP={tp}, FN={fn})")
dong.append(f"Specificity (do dac hieu) = {spec:.3f}   (TN={tn}, FP={fp})")
dong.append(f"AUC cua ROC               = {auc:.3f}")
for t in (5, 10, 15):
    a, s, sp, _ = chi_so(d_main, y, t)
    dong.append(f"  T={t:<3}: acc={a:.3f} sens={s:.3f} spec={sp:.3f}")
print("\n" + "\n".join(dong))

# luu bao cao + csv
os.makedirs("results", exist_ok=True)
with open(ket_qua("c4_bao_cao.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(dong))
with open(ket_qua("c4_cap_anh.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["stt", "nhan(1=tuong tu)", "mo_ta", "hamming"])
    for i, ((_, _, lb, mt), dd) in enumerate(zip(cap, d_main)):
        w.writerow([i, lb, mt, int(dd)])

# hinh: histogram khoang cach + ROC
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
bins = np.arange(0, 66, 2)
ax[0].hist(d_main[y == 1], bins, alpha=0.7, label="Cap tuong tu", color="tab:green")
ax[0].hist(d_main[y == 0], bins, alpha=0.7, label="Cap khong tuong tu", color="tab:red")
ax[0].axvline(t_best, color="k", ls="--", label=f"Nguong T={t_best}")
ax[0].set_xlabel("Khoang cach Hamming"); ax[0].set_ylabel("So cap"); ax[0].legend()
ax[0].set_title("Phan bo khoang cach Hamming")
ax[1].plot(fpr, tpr, "b-o", ms=3, label=f"Wavelet hash (AUC={auc:.3f})")
ax[1].plot([0, 1], [0, 1], "k--", label="Ngau nhien")
ax[1].scatter([fpr[best]], [tpr[best]], c="r", zorder=5, label=f"T={t_best}")
ax[1].set_xlabel("FPR = 1 - Specificity"); ax[1].set_ylabel("TPR = Sensitivity")
ax[1].set_title("Duong cong ROC"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
hien_thi_matplotlib(fig, "c4_roc.png")

# vi du vai cap anh
ids = list(np.where(y == 1)[0][:3]) + list(np.where(y == 0)[0][:3])
fig, ax = plt.subplots(2, 6, figsize=(18, 6))
for k, i in enumerate(ids):
    a, b, lb, mt = cap[i]
    ax[0, k].imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)); ax[0, k].axis("off")
    ax[1, k].imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)); ax[1, k].axis("off")
    ax[0, k].set_title(f"{'TUONG TU' if lb else 'KHAC'} ({mt})\nHamming={int(d_main[i])}")
plt.tight_layout()
hien_thi_matplotlib(fig, "c4_vi_du_cap_anh.png", show=False)

# ----------------------- Nang cao 1: so sanh phuong phap -----------------------
print("\nNANG CAO 1 - So sanh cac phuong phap bam")
print(f"{'Phuong phap':<26}{'AUC':>7}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'T*':>5}")
fig, ax = plt.subplots(figsize=(7, 6))
tom_tat = []
for ten, fn_hash in H.PHUONG_PHAP.items():
    d = tinh_khoang_cach(fn_hash)
    fp_, tp_, th_, au = roc_tu_khoang_cach(d, y)
    b = int(np.argmax(tp_ - fp_))
    a_, s_, sp_, _ = chi_so(d, y, int(th_[b]))
    print(f"{ten:<26}{au:>7.3f}{a_:>8.3f}{s_:>8.3f}{sp_:>8.3f}{int(th_[b]):>5}")
    tom_tat.append((ten, au, a_, s_, sp_, int(th_[b])))
    ax.plot(fp_, tp_, label=f"{ten} (AUC={au:.3f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC - so sanh cac phuong phap bam")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
hien_thi_matplotlib(fig, "c4_so_sanh_phuong_phap.png")
with open(ket_qua("c4_bao_cao.txt"), "a", encoding="utf-8") as fh:
    fh.write("\n\nSO SANH PHUONG PHAP\n")
    for r in tom_tat:
        fh.write(f"{r[0]:<26} AUC={r[1]:.3f} acc={r[2]:.3f} sens={r[3]:.3f} spec={r[4]:.3f} T*={r[5]}\n")
print("\nXong. Xem thu muc results/")
