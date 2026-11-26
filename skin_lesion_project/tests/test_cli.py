import sys

import src.cli as cli
import src.prepare as prepare


def test_prepare_cli_forwards_metadata_and_phash(tmp_path, monkeypatch):
    cfg = {
        'seed': 42,
        'paths': {
            'images': str(tmp_path / 'images'),
            'masks': str(tmp_path / 'masks'),
            'splits': str(tmp_path / 'splits'),
            'reports': str(tmp_path / 'reports'),
            'processed': str(tmp_path / 'processed'),
            'checkpoints': str(tmp_path / 'checkpoints'),
            'results': str(tmp_path / 'results'),
            'figures': str(tmp_path / 'figures'),
            'demo': str(tmp_path / 'demo'),
        },
        'data': {
            'mask_threshold': 127,
            'min_box_area_ratio': 0.001,
            'use_phash_grouping': True,
            'phash_max_distance': 2,
        },
        'split': {'train': 0.7, 'val': 0.15},
    }
    captured = {}
    monkeypatch.setattr(cli, 'load_config', lambda: cfg)
    monkeypatch.setattr(cli, 'ensure_dirs', lambda _cfg: None)
    monkeypatch.setattr(cli, 'seed_everything', lambda _seed: None)

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return {'train': [], 'val': [], 'test': []}

    monkeypatch.setattr(prepare, 'prepare_splits', fake_prepare)
    metadata = tmp_path / 'metadata.csv'
    monkeypatch.setattr(sys, 'argv', ['src.cli', 'prepare', '--metadata', str(metadata), '--no-phash'])
    cli.main()

    assert captured['metadata_path'] == str(metadata)
    assert captured['use_phash'] is False
