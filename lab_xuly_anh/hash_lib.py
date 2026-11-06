"""
hash_lib.py - Cac ham bam anh (wavelet hash + vai phuong phap so sanh) dung cho Chuong 4.

Neu may co thu vien PyWavelets (pip install PyWavelets) thi dung pywt.wavedec2 that.
Neu chua cai, tu dong dung ban Haar viet bang numpy (ket qua tuong duong cho wavelet 'haar').
"""
import numpy as np
import cv2

try:
    import pywt
    CO_PYWT = True
except ImportError:                      # chay duoc ca khi chua cai PyWavelets
    pywt = None
    CO_PYWT = False


# ---------------- Wavelet Haar du phong (khi khong co pywt) ----------------
def _haar_dwt2(a):
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]
    lo_r = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2)
    hi_r = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
    ll = (lo_r[0::2] + lo_r[1::2]) / np.sqrt(2)
    lh = (lo_r[0::2] - lo_r[1::2]) / np.sqrt(2)
    hl = (hi_r[0::2] + hi_r[1::2]) / np.sqrt(2)
    hh = (hi_r[0::2] - hi_r[1::2]) / np.sqrt(2)
    return ll, (lh, hl, hh)


def wavedec2(a, wavelet="haar", level=1):
    """Tra ve [LL_n, (details_n), ..., (details_1)] giong pywt.wavedec2."""
    if CO_PYWT:
        return pywt.wavedec2(a, wavelet, level=level)
    if wavelet != "haar":
        raise RuntimeError(f"Wavelet '{wavelet}' can PyWavelets: pip install PyWavelets")
    out, cur = [], a
    for _ in range(level):
        cur, det = _haar_dwt2(cur)
        out.append(det)
    return [cur] + out[::-1]


# ---------------- Tien xu ly ----------------
def _gray01(img, size):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return g.astype(np.float64) / 255.0


# ---------------- Cac ham bam (tra ve mang bool 1-D) ----------------
def wavelet_hash(img, hash_size=8, wavelet="haar", image_scale=None):
    """
    Bam wavelet (phong cach imagehash.whash):
      1. Chuyen xam, resize ve image_scale x image_scale (luy thua cua 2, >= hash_size)
      2. Bien doi wavelet nhieu cap cho den khi he so xap xi LL con hash_size x hash_size
      3. Luong tu hoa: bit = 1 neu he so LL > trung vi cua cac he so (64 bit voi hash 8x8)
    """
    if image_scale is None:
        image_scale = max(64, 2 ** int(np.ceil(np.log2(hash_size * 4))))
    level = int(np.log2(image_scale // hash_size))
    a = _gray01(img, image_scale)
    ll = wavedec2(a, wavelet, level=level)[0]
    return (ll > np.median(ll)).flatten()


def _tai_tao(coeffs, wavelet):
    if CO_PYWT:
        return pywt.waverec2(coeffs, wavelet)
    cur = coeffs[0]
    for (lh, hl, hh) in coeffs[1:]:
        h, w = cur.shape
        lo_r = np.zeros((h * 2, w))
        lo_r[0::2], lo_r[1::2] = (cur + lh) / np.sqrt(2), (cur - lh) / np.sqrt(2)
        hi_r = np.zeros((h * 2, w))
        hi_r[0::2], hi_r[1::2] = (hl + hh) / np.sqrt(2), (hl - hh) / np.sqrt(2)
        out = np.zeros((h * 2, w * 2))
        out[:, 0::2], out[:, 1::2] = (lo_r + hi_r) / np.sqrt(2), (lo_r - hi_r) / np.sqrt(2)
        cur = out
    return cur


def wavelet_hash_chi_tiet(img, hash_size=8, wavelet="haar", level=3):
    """Bien the: bam tu nang luong cac he so chi tiet (LH, HL, HH) theo luoi."""
    a = _gray01(img, hash_size * 2 ** level)
    c = wavedec2(a, wavelet, level=level)
    lh, hl, hh = c[1]
    nl = np.abs(lh) + np.abs(hl) + np.abs(hh)
    nl = cv2.resize(nl, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return (nl > np.median(nl)).flatten()


def average_hash(img, hash_size=8):
    g = _gray01(img, hash_size)
    return (g > g.mean()).flatten()


def difference_hash(img, hash_size=8):
    g = cv2.resize(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                   (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA).astype(np.float64)
    return (g[:, 1:] > g[:, :-1]).flatten()


def perceptual_hash(img, hash_size=8, highfreq=4):
    g = _gray01(img, hash_size * highfreq).astype(np.float32)
    d = cv2.dct(g)[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


PHUONG_PHAP = {
    "wavelet-haar (LL)": wavelet_hash,
    "wavelet-detail": wavelet_hash_chi_tiet,
    "average (aHash)": average_hash,
    "difference (dHash)": difference_hash,
    "perceptual (pHash/DCT)": perceptual_hash,
}
if CO_PYWT:
    PHUONG_PHAP["wavelet-db4 (LL)"] = lambda im, hash_size=8: wavelet_hash(im, hash_size, wavelet="db4")


def hamming(h1, h2):
    return int(np.count_nonzero(h1 != h2))


# ---------------- Bien doi tao cap anh "tuong tu" ----------------
def bien_doi_ngau_nhien(img, rng):
    """Tra ve (anh_bien_doi, ten) - cung mot doi tuong nhung nhieu / sang / xoay / nen JPEG / cat..."""
    kind = rng.choice(["nhieu", "sang", "tuong_phan", "xoay", "jpeg", "co_gian", "lam_mo", "cat"])
    h, w = img.shape[:2]
    if kind == "nhieu":
        out = np.clip(img.astype(np.float32) + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
    elif kind == "sang":
        out = cv2.convertScaleAbs(img, alpha=1.0, beta=int(rng.integers(-40, 40)))
    elif kind == "tuong_phan":
        out = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.7, 1.3)), beta=0)
    elif kind == "xoay":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-6, 6)), 1.0)
        out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 50))])
        out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    elif kind == "co_gian":
        s = float(rng.uniform(0.6, 0.9))
        out = cv2.resize(cv2.resize(img, None, fx=s, fy=s), (w, h))
    elif kind == "lam_mo":
        out = cv2.GaussianBlur(img, (5, 5), float(rng.uniform(1, 2)))
    else:
        dx, dy = int(w * 0.06), int(h * 0.06)
        out = cv2.resize(img[dy:h - dy, dx:w - dx], (w, h))
    return out, kind


def anh_tong_hop(seed, size=256):
    """Sinh anh nhan tao ngau nhien (de chay duoc khi khong co du lieu)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), int(rng.integers(150, 255)), np.uint8)
    for _ in range(int(rng.integers(4, 9))):
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = (int(v) for v in rng.integers(10, size - 10, 2))
        t = rng.integers(0, 3)
        if t == 0:
            cv2.circle(img, (x, y), int(rng.integers(15, 60)), col, -1)
        elif t == 1:
            cv2.rectangle(img, (x, y), (min(size - 1, x + int(rng.integers(20, 90))),
                                        min(size - 1, y + int(rng.integers(20, 90)))), col, -1)
        else:
            cv2.line(img, (x, y), (int(rng.integers(0, size)), int(rng.integers(0, size))), col, 6)
    return img
