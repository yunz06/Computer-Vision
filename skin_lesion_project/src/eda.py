from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from .data import find_pairs, read_image_size, read_mask, mask_to_boxes
from .utils import save_json


def _summary_stats(values):
    if not values:
        return {'count': 0, 'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0, 'std': 0.0, 'p05': 0.0, 'p95': 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'count': int(arr.size),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std()),
        'p05': float(np.percentile(arr, 5)),
        'p95': float(np.percentile(arr, 95)),
    }


def _save_eda_plot(widths, heights, area_ratios, comp_counts, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not widths:
            return
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

        axes[0].scatter(widths, heights, alpha=0.35, s=12)
        axes[0].set_title('Image Dimensions (W vs H)')
        axes[0].set_xlabel('Width (px)')
        axes[0].set_ylabel('Height (px)')

        axes[1].hist(area_ratios, bins=30, color='teal', edgecolor='white')
        axes[1].set_title('Lesion Area Ratio')
        axes[1].set_xlabel('Mask pixels / Image pixels')
        axes[1].set_ylabel('Count')

        comp_counter = Counter(comp_counts)
        xs = sorted(comp_counter.keys())
        ys = [comp_counter[x] for x in xs]
        axes[2].bar([str(x) for x in xs], ys, color='coral')
        axes[2].set_title('Connected Components per Mask')
        axes[2].set_xlabel('Components count')
        axes[2].set_ylabel('Images')

        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f'Không vẽ được biểu đồ EDA: {exc}')


def _eda_one(item, mask_threshold, min_area_ratio):
    key, ip, mp = item
    try:
        w, h = read_image_size(ip)
        m = read_mask(mp, threshold=mask_threshold)
    except Exception as e:
        return {'ok': False, 'key': key, 'reason': str(e)}
    if m.shape != (h, w):
        return {'ok': False, 'key': key, 'reason': f'size_mismatch {m.shape} vs {(h, w)}'}
    n_comp, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = mask_to_boxes(m, min_area_ratio=min_area_ratio, fallback_largest=False)
    return {
        'ok': True,
        'w': w,
        'h': h,
        'area_ratio': float(m.mean()),
        'comp': max(0, int(n_comp) - 1),
        'boxes': int(len(boxes)),
    }


def run_eda(image_dir, mask_dir, report_dir, figure_dir, mask_threshold=127, min_area_ratio=0.001, limit=0, workers=8):
    pairs, _, _ = find_pairs(image_dir, mask_dir)
    if limit and limit > 0:
        pairs = pairs[:limit]

    widths, heights, aspect_ratios = [], [], []
    area_ratios, comp_counts, filtered_box_counts = [], [], []
    invalid_samples = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, res in enumerate(pool.map(lambda it: _eda_one(it, mask_threshold, min_area_ratio), pairs), 1):
            if idx % 500 == 0 or idx == len(pairs):
                print(f'[eda] {idx}/{len(pairs)}', flush=True)
            if not res['ok']:
                invalid_samples.append({'key': res['key'], 'reason': res['reason']})
                continue
            widths.append(res['w'])
            heights.append(res['h'])
            aspect_ratios.append(res['w'] / max(1, res['h']))
            area_ratios.append(res['area_ratio'])
            comp_counts.append(res['comp'])
            filtered_box_counts.append(res['boxes'])

    summary = {
        'sample_count': len(widths),
        'invalid_count': len(invalid_samples),
        'invalid_samples': invalid_samples[:50],
        'mask_threshold': mask_threshold,
        'min_box_area_ratio': min_area_ratio,
        'width_stats': _summary_stats(widths),
        'height_stats': _summary_stats(heights),
        'aspect_ratio_stats': _summary_stats(aspect_ratios),
        'lesion_area_ratio_stats': _summary_stats(area_ratios),
        'raw_connected_components_dist': {str(k): int(v) for k, v in sorted(Counter(comp_counts).items())},
        'filtered_boxes_dist': {str(k): int(v) for k, v in sorted(Counter(filtered_box_counts).items())},
        'multi_component_rate': float(np.mean([c > 1 for c in comp_counts])) if comp_counts else 0.0,
        'empty_mask_count': int(sum(1 for c in comp_counts if c == 0)),
    }

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, report_dir / 'eda_summary.json')

    md_lines = [
        '# Báo cáo EDA – ISIC 2018 Task 1',
        '',
        f"- Số mẫu hợp lệ đã phân tích: **{summary['sample_count']}** (lỗi: {summary['invalid_count']})",
        f"- Ngưỡng nhị phân hóa mask: `> {mask_threshold}`",
        f"- Ngưỡng diện tích lọc box: `{min_area_ratio}` ({min_area_ratio * 100:.2f}% diện tích ảnh)",
        f"- Chiều rộng ảnh: mean `{summary['width_stats']['mean']:.1f}`, median `{summary['width_stats']['median']:.1f}`, min `{summary['width_stats']['min']:.0f}`, max `{summary['width_stats']['max']:.0f}`",
        f"- Chiều cao ảnh: mean `{summary['height_stats']['mean']:.1f}`, median `{summary['height_stats']['median']:.1f}`, min `{summary['height_stats']['min']:.0f}`, max `{summary['height_stats']['max']:.0f}`",
        f"- Tỷ lệ diện tích tổn thương: mean `{summary['lesion_area_ratio_stats']['mean']:.4f}`, median `{summary['lesion_area_ratio_stats']['median']:.4f}`, p05 `{summary['lesion_area_ratio_stats']['p05']:.4f}`, p95 `{summary['lesion_area_ratio_stats']['p95']:.4f}`",
        f"- Tỷ lệ ảnh có >1 connected component: `{summary['multi_component_rate'] * 100:.2f}%`",
        f"- Phân bố connected components (thô): `{summary['raw_connected_components_dist']}`",
        f"- Phân bố box sau lọc diện tích: `{summary['filtered_boxes_dist']}`",
    ]
    (report_dir / 'eda_report.md').write_text('\n'.join(md_lines), encoding='utf-8')
    _save_eda_plot(widths, heights, area_ratios, comp_counts, Path(figure_dir) / 'eda_summary.png')
    return summary
