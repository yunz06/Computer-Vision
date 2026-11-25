from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
from .data import find_pairs, read_mask_raw, read_rgb, IMAGE_EXTS
from .utils import save_json


def _audit_one(item, mask_threshold):
    key, ip, mp = item
    try:
        with Image.open(ip) as im:
            im_size = im.size
        image_shape = read_rgb(ip).shape[:2]
    except Exception as e:
        return {'status': 'bad_image', 'key': key, 'error': str(e)}

    try:
        raw_m = read_mask_raw(mp)
        uniq = np.unique(raw_m)
        is_binary_0_255 = set(int(v) for v in uniq).issubset({0, 255})
        m = (raw_m > mask_threshold).astype(np.uint8)
        n, _, _, _ = cv2.connectedComponentsWithStats(m, 8)
        return {
            'status': 'ok',
            'key': key,
            'im_size': str(im_size),
            'size_mismatch': m.shape != image_shape,
            'image_shape': list(image_shape),
            'mask_shape': list(m.shape),
            'empty': not bool(m.any()),
            'components': max(0, int(n) - 1),
            'is_binary_0_255': is_binary_0_255,
            'uniq_sample': [int(v) for v in uniq[:10]],
            'min': int(uniq.min()),
            'max': int(uniq.max()),
            'unique_count': int(len(uniq)),
        }
    except Exception as e:
        return {'status': 'bad_mask', 'key': key, 'error': str(e)}


def run_audit(image_dir, mask_dir, report_dir, mask_threshold=127, workers=8):
    pairs, missing_masks, orphan_masks = find_pairs(image_dir, mask_dir)
    bad_images = []
    bad_masks = []
    size_mismatch = []
    empty = []
    non_binary_masks = []
    sizes = Counter()
    components = Counter()
    pixel_values_seen = set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, res in enumerate(pool.map(lambda it: _audit_one(it, mask_threshold), pairs), 1):
            if idx % 500 == 0 or idx == len(pairs):
                print(f'[audit] {idx}/{len(pairs)}', flush=True)
            if res['status'] == 'bad_image':
                bad_images.append({'key': res['key'], 'error': res['error']})
                continue
            if res['status'] == 'bad_mask':
                bad_masks.append({'key': res['key'], 'error': res['error']})
                continue
            sizes[res['im_size']] += 1
            if len(pixel_values_seen) < 32:
                pixel_values_seen.update(res['uniq_sample'])
            if not res['is_binary_0_255']:
                non_binary_masks.append({
                    'key': res['key'],
                    'unique_count': res['unique_count'],
                    'min': res['min'],
                    'max': res['max'],
                })
            if res['size_mismatch']:
                size_mismatch.append({
                    'key': res['key'],
                    'image_shape': res['image_shape'],
                    'mask_shape': res['mask_shape'],
                })
            if res['empty']:
                empty.append(res['key'])
            components[str(res['components'])] += 1

    bad_img_keys = {x['key'] for x in bad_images}
    bad_mask_keys = {x['key'] for x in bad_masks}
    mismatch_keys = {x['key'] for x in size_mismatch}
    empty_keys = set(empty)

    result = {
        'image_count': sum(1 for p in Path(image_dir).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.stem.lower().endswith('_segmentation')),
        'mask_count': sum(1 for p in Path(mask_dir).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.lower().endswith('_segmentation')),
        'paired_count': len(pairs),
        'missing_masks': missing_masks,
        'orphan_masks': orphan_masks,
        'bad_images': bad_images,
        'bad_masks': bad_masks,
        'mask_size_mismatch': size_mismatch,
        'empty_masks': empty,
        'non_binary_masks_count': len(non_binary_masks),
        'non_binary_masks_sample': non_binary_masks[:10],
        'mask_pixel_values_sample': sorted(pixel_values_seen),
        'mask_binarization_threshold': mask_threshold,
        'image_sizes': dict(sizes),
        'connected_component_count': dict(components),
        'segmentation_usable_count': sum(
            1 for key, _, _ in pairs
            if key not in bad_img_keys and key not in bad_mask_keys and key not in mismatch_keys
        ),
        'detection_usable_count': sum(
            1 for key, _, _ in pairs
            if key not in empty_keys and key not in bad_img_keys and key not in bad_mask_keys and key not in mismatch_keys
        ),
    }

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    save_json(result, Path(report_dir) / 'audit.json')
    lines = [
        f"Tổng ảnh: {result['image_count']}",
        f"Tổng mask: {result['mask_count']}",
        f"Cặp ảnh-mask: {len(pairs)}",
        f"Thiếu mask: {len(missing_masks)}",
        f"Mask không có ảnh: {len(orphan_masks)}",
        f"Ảnh lỗi: {len(bad_images)}",
        f"Mask lỗi: {len(bad_masks)}",
        f"Sai kích thước ảnh-mask: {len(size_mismatch)}",
        f"Mask rỗng: {len(empty)}",
        f"Mask có giá trị trung gian (ngoài 0/255): {len(non_binary_masks)}",
        f"Ngưỡng nhị phân hóa mask: > {mask_threshold}",
        f"Mẫu segmentation dùng được: {result['segmentation_usable_count']}",
        f"Mẫu detection dùng được: {result['detection_usable_count']}",
    ]
    (Path(report_dir) / 'audit.txt').write_text('\n'.join(lines), encoding='utf-8')
    return result
