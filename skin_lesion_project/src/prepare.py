import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .data import find_pairs, read_mask, mask_to_boxes
from .utils import save_json


def _load_lesion_metadata(metadata_path):
    mapping = {}
    if not metadata_path:
        return mapping
    p = Path(metadata_path)
    if not p.exists():
        return mapping
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = (row.get('image_id') or row.get('image') or '').strip().lower()
            lesion_id = (row.get('lesion_id') or '').strip()
            if img_id and lesion_id:
                mapping[img_id] = lesion_id
    return mapping


def _compute_dhash_fast(image_path):
    with Image.open(image_path) as im:
        w, h = im.size
        im.draft('L', (64, 64))
        im.load()
        gray = np.asarray(im.convert('L'))
    if float(gray.std()) < 1e-3:
        return (w, h), None
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return (w, h), bits



def _group_by_phash(records, max_distance=2):
    """
    Gom nhóm ảnh gần trùng bằng dHash và complete-link.
    Ảnh mới chỉ được thêm vào nhóm nếu gần tất cả ảnh trong nhóm.
    Tránh hiện tượng chaining của Union-Find.
    """
    hashes = {
        idx: record.get('_dhash')
        for idx, record in enumerate(records)
    }

    valid_indices = sorted(
        (idx for idx, value in hashes.items() if value is not None),
        key=lambda idx: records[idx]['key']
    )

    clusters = []

    for idx in valid_indices:
        current_hash = hashes[idx]
        assigned = False

        for cluster in clusters:
            if all(
                (current_hash ^ hashes[other]).bit_count() <= max_distance
                for other in cluster
            ):
                cluster.append(idx)
                assigned = True
                break

        if not assigned:
            clusters.append([idx])

    groups = {}

    for cluster in clusters:
        group_id = f"dhash_{min(cluster):05d}"
        for idx in cluster:
            groups[idx] = group_id

    # Ảnh không tính được hash được giữ thành nhóm riêng.
    for idx, value in hashes.items():
        if value is None:
            groups[idx] = f"dhash_single_{idx:05d}"

    return groups



def _group_split(records, train_ratio, val_ratio, seed):
    n = len(records)
    if n == 0:
        return [], [], []
    groups = [r['group_id'] for r in records]
    unique_groups = sorted(set(groups))
    # Mỗi ảnh là một nhóm riêng thì chia ngẫu nhiên tương đương group split.
    if len(unique_groups) == n:
        rng = random.Random(seed)
        copied = list(records)
        rng.shuffle(copied)
        ntr = int(n * train_ratio)
        nval = int(n * val_ratio)
        return copied[:ntr], copied[ntr:ntr + nval], copied[ntr + nval:]

    # Không được fallback sang chia từng ảnh khi có nhóm trùng:
    # làm vậy có thể đưa cùng một lesion vào nhiều split.
    if len(unique_groups) < 3:
        raise ValueError(
            f'Chỉ có {len(unique_groups)} nhóm dữ liệu; không thể tạo train/val/test '
            'mà vẫn giữ nguyên group. Hãy kiểm tra metadata hoặc dữ liệu đầu vào.'
        )

    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    g_total = len(unique_groups)
    g_tr = max(1, min(g_total - 2, int(round(g_total * train_ratio))))
    if val_ratio > 0:
        g_va = max(1, min(g_total - g_tr - 1, int(round(g_total * val_ratio))))
    else:
        g_va = 0
    tr_set = set(unique_groups[:g_tr])
    va_set = set(unique_groups[g_tr:g_tr + g_va])
    train_recs = [r for r in records if r['group_id'] in tr_set]
    val_recs = [r for r in records if r['group_id'] in va_set]
    test_recs = [r for r in records if r['group_id'] not in tr_set and r['group_id'] not in va_set]
    return train_recs, val_recs, test_recs


def prepare_splits(
    image_dir,
    mask_dir,
    split_dir,
    sample_size=0,
    seed=42,
    train=0.7,
    val=0.15,
    mask_threshold=127,
    min_area_ratio=0.001,
    metadata_path=None,
    use_phash=False,
    phash_max_distance=2,
):
    if train <= 0 or val < 0 or train + val >= 1.0:
        raise ValueError('Tỷ lệ split không hợp lệ: train phải > 0, val >= 0 và train + val < 1.')

    if metadata_path is None:
        raw_root = Path(image_dir).parent
        candidates = list(raw_root.rglob('*metadata*.csv')) if raw_root.exists() else []
        metadata_path = candidates[0] if candidates else None
    lesion_map = _load_lesion_metadata(metadata_path)

    pairs, _, _ = find_pairs(image_dir, mask_dir)
    valid = []
    manifest_rows = []

    def _inspect_pair(item):
        k, ip, mp = item
        try:
            if use_phash and k not in lesion_map:
                (w, h), dhash = _compute_dhash_fast(ip)
            else:
                with Image.open(ip) as im:
                    w, h = im.size
                dhash = None
            mask = read_mask(mp, threshold=mask_threshold)
            if mask.shape != (h, w):
                return {'ok': False, 'row': {
                    'image_id': k, 'image_path': str(ip.resolve()), 'mask_path': str(mp.resolve()),
                    'width': w, 'height': h, 'connected_components': 0, 'boxes': '[]',
                    'lesion_area_ratio': 0.0, 'group_id': k, 'valid': False, 'reason': 'size_mismatch',
                }}
            n_comp, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            comp_count = max(0, int(n_comp) - 1)
            boxes = mask_to_boxes(mask, min_area_ratio=min_area_ratio, fallback_largest=False)
            area_ratio = float(mask.mean())
            group_id = lesion_map.get(k, k)
            rec = {
                'key': k,
                'image': str(ip.resolve()),
                'mask': str(mp.resolve()),
                'width': w,
                'height': h,
                'connected_components': comp_count,
                'boxes': boxes.tolist(),
                'lesion_area_ratio': round(area_ratio, 6),
                'group_id': group_id,
                '_dhash': dhash,
            }
            row = {
                'image_id': k, 'image_path': rec['image'], 'mask_path': rec['mask'],
                'width': w, 'height': h, 'connected_components': comp_count,
                'boxes': json.dumps(rec['boxes']), 'lesion_area_ratio': rec['lesion_area_ratio'],
                'group_id': group_id, 'valid': True, 'reason': 'empty_mask' if comp_count == 0 else 'ok',
            }
            return {'ok': True, 'rec': rec, 'row': row}
        except Exception as exc:
            return {'ok': False, 'row': {
                'image_id': k, 'image_path': str(ip.resolve()), 'mask_path': str(mp.resolve()),
                'width': 0, 'height': 0, 'connected_components': 0, 'boxes': '[]',
                'lesion_area_ratio': 0.0, 'group_id': k, 'valid': False, 'reason': str(exc),
            }}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for idx, res in enumerate(pool.map(_inspect_pair, pairs), 1):
            if idx % 500 == 0 or idx == len(pairs):
                print(f'[prepare] {idx}/{len(pairs)}', flush=True)
            manifest_rows.append(res['row'])
            if res['ok']:
                valid.append(res['rec'])

    if use_phash and valid:
        unmapped_indices = [i for i, r in enumerate(valid) if r['key'] not in lesion_map]
        if unmapped_indices:
            sub_records = [valid[i] for i in unmapped_indices]
            phash_groups = _group_by_phash(sub_records, max_distance=phash_max_distance)
            updated_groups = {}
            for local_i, global_i in enumerate(unmapped_indices):
                valid[global_i]['group_id'] = phash_groups[local_i]
                updated_groups[valid[global_i]['key']] = phash_groups[local_i]
            for row in manifest_rows:
                if row['image_id'] in updated_groups:
                    row['group_id'] = updated_groups[row['image_id']]

    rng = random.Random(seed)
    rng.shuffle(valid)
    if sample_size and sample_size > 0:
        valid = valid[:min(sample_size, len(valid))]

    for r in valid:
        r.pop('_dhash', None)

    train_recs, val_recs, test_recs = _group_split(valid, train, val, seed)

    metadata_count = sum(
        r['key'] in lesion_map for r in valid
    )

    if metadata_count == len(valid) and valid:
        policy = 'group-based split using lesion_id'
    elif metadata_count > 0:
        policy = (
            f'mixed grouping using lesion_id and complete-link dHash '
            f'(max_distance={phash_max_distance})'
        )
    elif use_phash:
        policy = (
            f'complete-link dHash grouping '
            f'(max_distance={phash_max_distance}); lesion_id unavailable'
        )
    else:
        policy = 'image-level random split; lesion-level grouping not inferred'

    def _clean(items):
        return [{k: v for k, v in x.items() if k in ('key', 'image', 'mask', 'group_id', 'width', 'height')} for x in items]

    split = {
        'train': _clean(train_recs),
        'val': _clean(val_recs),
        'test': _clean(test_recs),
        'seed': seed,
        'sample_size': len(valid),
        'policy': policy,
    }

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    save_json(split, split_dir / 'split.json')

    for name, items in (('train', train_recs), ('val', val_recs), ('test', test_recs)):
        (split_dir / f'{name}.txt').write_text('\n'.join(x['key'] for x in items), encoding='utf-8')

    manifest_path = split_dir / 'manifest.csv'
    fieldnames = [
        'image_id', 'image_path', 'mask_path', 'width', 'height',
        'connected_components', 'boxes', 'lesion_area_ratio', 'group_id', 'valid', 'reason',
    ]
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return split


def load_records(split_path, part):
    d = json.loads(Path(split_path).read_text(encoding='utf-8'))
    return [(x['key'], Path(x['image']), Path(x['mask'])) for x in d[part]]
