import json
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_loss_plot(history, path, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not history:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        epochs = [row['epoch'] for row in history]
        values = [row['train_loss'] for row in history]
        fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
        l1 = ax1.plot(epochs, values, color='tab:blue', label='Train loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train loss', color='tab:blue')

        lines = list(l1)
        val_key = None
        val_label = None
        for candidate, label in (
            ('val_dice', 'Val Dice'),
            ('val_ap5095', 'Val AP@0.5:0.95'),
            ('val_ap50', 'Val AP@0.5'),
        ):
            if candidate in history[0]:
                val_key = candidate
                val_label = label
                break
        if val_key is not None:
            ax2 = ax1.twinx()
            l2 = ax2.plot(
                epochs,
                [row[val_key] for row in history],
                color='tab:orange',
                linestyle='--',
                label=val_label,
            )
            ax2.set_ylabel(val_label, color='tab:orange')
            lines.extend(l2)

        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')
        ax1.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f'Không lưu được biểu đồ: {exc}')
