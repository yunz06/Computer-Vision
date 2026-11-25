from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_fill_holes

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def image_key(path):
    stem = Path(path).stem
    if stem.lower().endswith('_segmentation'):
        stem = stem[:-13]
    return stem.lower()


def find_pairs(image_dir, mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    imgs = {
        image_key(p): p
        for p in image_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.stem.lower().endswith('_segmentation')
    }
    masks = {
        image_key(p): p
        for p in mask_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.lower().endswith('_segmentation')
    }
    pairs = [(k, imgs[k], masks[k]) for k in sorted(imgs.keys() & masks.keys())]
    return pairs, sorted(imgs.keys() - masks.keys()), sorted(masks.keys() - imgs.keys())


def read_image_size(path):
    with Image.open(path) as im:
        return im.size  # (width, height)


def read_rgb(path):
    with Image.open(path) as im:
        im.load()
        return np.asarray(im.convert('RGB'))


def read_mask_raw(path):
    with Image.open(path) as im:
        im.load()
        return np.asarray(im.convert('L'))


def read_mask(path, threshold=127):
    arr = read_mask_raw(path)
    return (arr > threshold).astype(np.uint8)


def letterbox_meta(w, h, size):
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    left = (size - nw) // 2
    top = (size - nh) // 2
    return {
        'original_width': w,
        'original_height': h,
        'scale': scale,
        'left': left,
        'top': top,
        'new_width': nw,
        'new_height': nh,
        'size': size,
    }


def letterbox(image, size, mask=None, image_fill=(0, 0, 0)):
    h, w = image.shape[:2]
    meta = letterbox_meta(w, h, size)
    nw, nh, left, top = meta['new_width'], meta['new_height'], meta['left'], meta['top']
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), image_fill, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    if mask is None:
        return canvas, meta
    m = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    mc = np.zeros((size, size), dtype=np.uint8)
    mc[top:top + nh, left:left + nw] = m
    return canvas, mc, meta


def load_or_cache_letterbox(key, ip, mp, size, mask_threshold=127, cache_dir=None):
    if cache_dir is not None:
        base = Path(cache_dir) / str(size)
        img_cache = base / 'images' / f'{key}.png'
        msk_cache = base / 'masks' / f'{key}.png' if mp is not None else None
        if img_cache.exists() and (msk_cache is None or msk_cache.exists()):
            w, h = read_image_size(ip)
            meta = letterbox_meta(w, h, size)
            lb_img = read_rgb(img_cache)
            if msk_cache is None:
                return lb_img, meta
            lb_mask = (read_mask_raw(msk_cache) > 0).astype(np.uint8)
            return lb_img, lb_mask, meta

    image = read_rgb(ip)
    if mp is None:
        lb_img, meta = letterbox(image, size)
        if cache_dir is not None:
            img_cache = Path(cache_dir) / str(size) / 'images' / f'{key}.png'
            img_cache.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(lb_img).save(img_cache)
        return lb_img, meta

    mask = read_mask(mp, threshold=mask_threshold)
    lb_img, lb_mask, meta = letterbox(image, size, mask)
    if cache_dir is not None:
        base = Path(cache_dir) / str(size)
        img_cache = base / 'images' / f'{key}.png'
        msk_cache = base / 'masks' / f'{key}.png'
        img_cache.parent.mkdir(parents=True, exist_ok=True)
        msk_cache.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(lb_img).save(img_cache)
        Image.fromarray((lb_mask * 255).astype(np.uint8)).save(msk_cache)
    return lb_img, lb_mask, meta


def unletterbox_prob_map(prob_map, meta):
    l, t = meta['left'], meta['top']
    nw, nh = meta['new_width'], meta['new_height']
    cropped = np.asarray(prob_map, dtype=np.float32)[t:t + nh, l:l + nw]
    restored = cv2.resize(
        cropped,
        (meta['original_width'], meta['original_height']),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(restored, 0.0, 1.0)


def unletterbox_mask(mask, meta):
    l, t = meta['left'], meta['top']
    nw, nh = meta['new_width'], meta['new_height']
    cropped = np.asarray(mask, dtype=np.uint8)[t:t + nh, l:l + nw]
    return cv2.resize(
        cropped,
        (meta['original_width'], meta['original_height']),
        interpolation=cv2.INTER_NEAREST,
    )


def unletterbox_boxes(boxes, meta):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scale, left, top = meta['scale'], meta['left'], meta['top']
    ow, oh = meta['original_width'], meta['original_height']
    out = boxes.copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / scale, 0.0, float(ow))
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top) / scale, 0.0, float(oh))
    valid = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out[valid]


def postprocess_mask(binary_mask, largest_component=False, hole_filling=False):
    m = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if not m.any():
        return m
    if largest_component:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if count > 1:
            comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (labels == comp).astype(np.uint8)
    if hole_filling:
        m = binary_fill_holes(m > 0).astype(np.uint8)
    return m


def mask_to_boxes(mask, min_area_ratio=0.001, fallback_largest=True):
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros((0, 4), dtype=np.float32)
    min_area = max(1, int(np.ceil(float(min_area_ratio) * h * w))) if min_area_ratio > 0 else 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    boxes = []
    for idx in order:
        comp = 1 + int(idx)
        x, y, bw, bh, area = stats[comp]
        if area >= min_area and bw > 0 and bh > 0:
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
    if not boxes and fallback_largest:
        comp = 1 + int(order[0])
        x, y, bw, bh, _ = stats[comp]
        if bw > 0 and bh > 0:
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def mask_to_box(mask, min_area_ratio=0.0):
    boxes = mask_to_boxes(mask, min_area_ratio=min_area_ratio, fallback_largest=(min_area_ratio <= 0))
    if len(boxes) == 0:
        return None
    return boxes[0]


def _apply_brightness_contrast(image, rng):
    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.9, 1.1))
        beta = float(rng.uniform(-12.0, 12.0))
        out = np.clip(image.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)
        return out
    return image


def augment_pair(image, mask, rng, allow_affine=True):
    if rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    if allow_affine and rng.random() < 0.5:
        h, w = image.shape[:2]
        angle = float(rng.uniform(-15.0, 15.0))
        scale = float(rng.uniform(0.92, 1.08))
        tx = float(rng.uniform(-0.05, 0.05) * w)
        ty = float(rng.uniform(-0.05, 0.05) * h)
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask.astype(np.uint8), mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    image = _apply_brightness_contrast(image, rng)
    return np.ascontiguousarray(image), np.ascontiguousarray((mask > 0).astype(np.uint8))


class SegmentationDataset(Dataset):
    def __init__(self, records, size=256, augment=False, mask_threshold=127, seed=42, cache_dir=None):
        self.records = list(records)
        self.size = size
        self.augment = augment
        self.mask_threshold = mask_threshold
        self.rng = np.random.default_rng(seed)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        key, ip, mp = self.records[idx]
        image, mask, _ = load_or_cache_letterbox(
            key, ip, mp, self.size, mask_threshold=self.mask_threshold, cache_dir=self.cache_dir
        )
        if self.augment:
            image, mask = augment_pair(image, mask, self.rng, allow_affine=True)
        x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        return x, y


class DetectionDataset(Dataset):
    def __init__(
        self,
        records,
        size=512,
        augment=False,
        mask_threshold=127,
        min_area_ratio=0.001,
        multi_component=True,
        seed=42,
        cache_dir=None,
    ):
        self.records = []
        self.size = size
        self.augment = augment
        self.mask_threshold = mask_threshold
        self.min_area_ratio = min_area_ratio
        self.multi_component = multi_component
        self.rng = np.random.default_rng(seed)
        self.cache_dir = cache_dir
        for r in records:
            try:
                m = read_mask(r[2], threshold=self.mask_threshold)
                boxes = mask_to_boxes(m, min_area_ratio=self.min_area_ratio, fallback_largest=True)
                if len(boxes) > 0:
                    self.records.append(r)
            except Exception:
                pass

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        key, ip, mp = self.records[idx]
        image, mask, _ = load_or_cache_letterbox(
            key, ip, mp, self.size, mask_threshold=self.mask_threshold, cache_dir=self.cache_dir
        )
        if self.augment:
            # Flip/rot90 trên ảnh + mask trước khi trích box đảm bảo đồng bộ tuyệt đối.
            image, mask = augment_pair(image, mask, self.rng, allow_affine=False)
        boxes = mask_to_boxes(mask, min_area_ratio=self.min_area_ratio, fallback_largest=True)
        if not self.multi_component and len(boxes) > 1:
            boxes = boxes[:1]
        if len(boxes) == 0:
            raise RuntimeError(
                f'Mask của mẫu {key} không còn vùng hợp lệ sau tiền xử lý; '
                'không thể tạo bounding box ground-truth.'
            )
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        target = {
            'boxes': torch.from_numpy(boxes.copy()),
            'labels': torch.ones((len(boxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.from_numpy(areas.astype(np.float32)),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return x, target


def detection_collate(batch):
    return tuple(zip(*batch))
