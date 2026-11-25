from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    path = Path(path) if path else ROOT / 'configs' / 'default.yaml'
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg['paths'].items():
        cfg['paths'][key] = str((ROOT / value).resolve())
    return cfg

def ensure_dirs(cfg):
    for key in ('processed','splits','reports','checkpoints','results','figures','demo'):
        Path(cfg['paths'][key]).mkdir(parents=True, exist_ok=True)
