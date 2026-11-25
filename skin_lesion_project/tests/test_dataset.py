import csv
import json
import numpy as np
from PIL import Image
from src.data import find_pairs, SegmentationDataset, DetectionDataset
from src.audit import run_audit
from src.eda import run_eda
from src.prepare import prepare_splits, load_records, _group_split
from src.baselines import run as run_baselines


def _make_synthetic_dataset(root, count=12, with_metadata=False):
    im = root / 'images'
    ma = root / 'masks'
    im.mkdir(parents=True, exist_ok=True)
    ma.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    meta_rows = []
    for i in range(count):
        arr = rng.integers(140, 220, (48, 64, 3), dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=np.uint8)
        if i > 0:  # sample 0 is an empty mask
            r0, c0 = 10 + (i % 5), 14 + (i % 6)
            arr[r0:r0 + 20, c0:c0 + 24] = 40
            mask[r0:r0 + 20, c0:c0 + 24] = 255
        img_id = f'ISIC_{i:07d}'
        Image.fromarray(arr).save(im / f'{img_id}.jpg')
        Image.fromarray(mask).save(ma / f'{img_id}_segmentation.png')
        meta_rows.append({'image_id': img_id, 'lesion_id': f'HAM_{i // 2:04d}'})
    if with_metadata:
        meta_path = root / 'HAM10000_metadata.csv'
        with meta_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_id', 'lesion_id'])
            writer.writeheader()
            writer.writerows(meta_rows)
        return im, ma, meta_path
    return im, ma, None


def test_audit_eda_and_group_split(tmp_path):
    im, ma, meta_path = _make_synthetic_dataset(tmp_path, count=12, with_metadata=True)
    # Add a corrupt image to verify bad_images is tracked separately from bad_masks
    (im / 'ISIC_9999999.jpg').write_bytes(b'not-an-image')
    Image.fromarray(np.zeros((48, 64), dtype=np.uint8)).save(ma / 'ISIC_9999999_segmentation.png')

    audit = run_audit(im, ma, tmp_path / 'reports', mask_threshold=127)
    assert audit['paired_count'] == 13
    assert len(audit['bad_images']) == 1
    assert len(audit['bad_masks']) == 0
    assert 'isic_0000000' in audit['empty_masks']

    eda = run_eda(im, ma, tmp_path / 'reports', tmp_path / 'figures', mask_threshold=127, min_area_ratio=0.001)
    assert eda['sample_count'] == 12
    assert eda['empty_mask_count'] == 1
    assert (tmp_path / 'reports' / 'eda_report.md').exists()
    assert (tmp_path / 'figures' / 'eda_summary.png').exists()

    split_dir = tmp_path / 'splits'
    s = prepare_splits(im, ma, split_dir, sample_size=0, seed=7, metadata_path=meta_path)
    assert len(s['train']) + len(s['val']) + len(s['test']) == 12
    assert (split_dir / 'manifest.csv').exists()
    assert (split_dir / 'train.txt').exists()
    assert (split_dir / 'val.txt').exists()
    assert (split_dir / 'test.txt').exists()

    # Check zero group_id overlap across train/val/test
    g_tr = {x['group_id'] for x in s['train']}
    g_va = {x['group_id'] for x in s['val']}
    g_te = {x['group_id'] for x in s['test']}
    assert not (g_tr & g_va)
    assert not (g_tr & g_te)
    assert not (g_va & g_te)


def test_pair_and_split_sample_size(tmp_path):
    im, ma, _ = _make_synthetic_dataset(tmp_path, count=10)
    pairs, mi, orphan = find_pairs(im, ma)
    assert len(pairs) == 10 and not mi and not orphan
    s = prepare_splits(im, ma, tmp_path / 'splits', sample_size=8, seed=7)
    assert len(s['train']) + len(s['val']) + len(s['test']) == 8


def test_datasets_and_baselines(tmp_path):
    im, ma, _ = _make_synthetic_dataset(tmp_path, count=6)
    pairs, _, _ = find_pairs(im, ma)
    seg_ds = SegmentationDataset(pairs, size=64, augment=True)
    x, y = seg_ds[1]
    assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64)

    det_ds = DetectionDataset(pairs, size=64, augment=True)
    assert len(det_ds) == 5  # empty mask sample 0 skipped
    dx, target = det_ds[0]
    assert dx.shape == (3, 64, 64)
    assert target['boxes'].shape[1] == 4
    assert (target['boxes'][:, 2] > target['boxes'][:, 0]).all()

    res = run_baselines(im, ma, tmp_path / 'results', size=64, records=pairs[1:])
    assert len(res) == 5
    summary = json.loads((tmp_path / 'results' / 'baseline_summary.json').read_text(encoding='utf-8'))
    assert 'otsu_morphology' in summary and 'kmeans_lab' in summary and 'watershed' in summary and 'center_box' in summary



def test_group_split_refuses_to_break_small_repeated_groups():
    records = [
        {'key': 'a', 'group_id': 'lesion_a'},
        {'key': 'b', 'group_id': 'lesion_a'},
        {'key': 'c', 'group_id': 'lesion_b'},
        {'key': 'd', 'group_id': 'lesion_b'},
    ]
    try:
        _group_split(records, 0.7, 0.15, 42)
    except ValueError as exc:
        assert 'không thể tạo train/val/test' in str(exc)
    else:
        raise AssertionError('Không được chia rời các ảnh thuộc cùng group.')


def test_detection_uses_one_box_for_largest_mask_component(tmp_path):
    im = tmp_path / 'images'
    ma = tmp_path / 'masks'
    im.mkdir()
    ma.mkdir()
    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 255       # vùng chính
    mask[70:80, 70:80] = 255       # vùng rời nhỏ hơn
    Image.fromarray(image).save(im / 'ISIC_0000001.jpg')
    Image.fromarray(mask).save(ma / 'ISIC_0000001_segmentation.png')
    pairs, _, _ = find_pairs(im, ma)

    ds = DetectionDataset(pairs, size=100, augment=False, multi_component=False)
    _, target = ds[0]
    assert target['boxes'].shape == (1, 4)
    assert torch_allclose(target['boxes'][0], [10, 10, 50, 50])


def torch_allclose(actual, expected):
    import torch
    return torch.allclose(actual, torch.tensor(expected, dtype=actual.dtype))
