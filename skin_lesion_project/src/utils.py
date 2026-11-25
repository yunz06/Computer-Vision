import json, random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_json(data, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_loss_plot(history, path, title):
    try:
        import matplotlib.pyplot as plt
        if not history: return
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        epochs = [row['epoch'] for row in history]
        values = [row['train_loss'] for row in history]
        plt.figure(figsize=(7, 4)); plt.plot(epochs, values, label='Train loss')
        if 'val_dice' in history[0]:
            plt.twinx(); plt.plot(epochs, [row['val_dice'] for row in history], linestyle='--', label='Validation Dice')
        plt.title(title); plt.xlabel('Epoch'); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    except Exception as exc:
        print(f'Không lưu được biểu đồ: {exc}')
