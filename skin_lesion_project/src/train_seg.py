from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import SegmentationDataset
from .prepare import load_records
from .models import build_segmentation_model
from .losses import segmentation_loss
from .metrics import binary_metrics
from .utils import device, save_json, save_loss_plot


def run(
    cfg,
    epochs=None,
    limit=0,
    model_name=None,
    loss_kind=None,
    image_size=None,
    batch_size=None,
    learning_rate=None,
    augment=None,
    pretrained=None,
    run_name=None,
):
    split_path = Path(cfg['paths']['splits']) / 'split.json'
    tr = load_records(split_path, 'train')
    va = load_records(split_path, 'val')
    if limit:
        tr = tr[:limit]
        va = va[:max(1, limit // 4)] if va else []
    if not tr:
        raise RuntimeError('Train split rỗng. Hãy chạy lệnh prepare trước.')

    seg_cfg = cfg['segmentation']
    size = image_size or cfg['image_size']
    bs = batch_size or seg_cfg['batch_size']
    lr = learning_rate or seg_cfg['learning_rate']
    loss_name = loss_kind or seg_cfg['loss']
    arch = model_name or seg_cfg.get('model', 'unet')
    use_pretrained = seg_cfg.get('pretrained', True) if pretrained is None else pretrained
    use_augment = seg_cfg.get('augment', True) if augment is None else augment
    threshold = seg_cfg.get('threshold', 0.5)
    mask_thresh = cfg.get('data', {}).get('mask_threshold', 127)
    tol_px = cfg.get('evaluation', {}).get('boundary_tolerance_px', 2)
    seed = cfg.get('seed', 42)
    dev = device()
    use_amp = dev.type == 'cuda' and bool(seg_cfg.get('amp', True))

    cache_dir = cfg['paths'].get('processed')
    train_ds = SegmentationDataset(tr, size, augment=use_augment, mask_threshold=mask_thresh, seed=seed, cache_dir=cache_dir)
    val_ds = SegmentationDataset(va, size, augment=False, mask_threshold=mask_thresh, seed=seed, cache_dir=cache_dir) if va else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=seg_cfg.get('num_workers', 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0) if val_ds is not None else []

    model = build_segmentation_model(arch, pretrained=use_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best = -1.0
    history = []
    n_epochs = epochs or seg_cfg['epochs']
    ckpt_dir = Path(cfg['paths']['checkpoints'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = run_name or 'unet'

    for ep in range(1, n_epochs + 1):
        model.train()
        losses = []
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
                loss = segmentation_loss(logits, y, loss_name)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        max_vram_mb = round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 1) if dev.type == 'cuda' else 0.0
        model.eval()
        vals = []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(dev))
                pred = torch.sigmoid(logits).cpu().numpy()[0, 0] >= threshold
                vals.append(binary_metrics(pred, y.numpy()[0, 0], boundary_tolerance_px=tol_px))

        dice = sum(v['dice'] for v in vals) / len(vals) if vals else 0.0
        iou = sum(v['iou'] for v in vals) / len(vals) if vals else 0.0
        row = {
            'epoch': ep,
            'train_loss': sum(losses) / max(1, len(losses)),
            'val_dice': dice,
            'val_iou': iou,
            'max_vram_mb': max_vram_mb,
        }
        history.append(row)
        print(row)

        state = {
            'model': model.state_dict(),
            'model_name': arch,
            'loss': loss_name,
            'epoch': ep,
            'val_dice': dice,
            'val_iou': iou,
            'image_size': size,
            'seed': seed,
            'pretrained': use_pretrained,
        }
        torch.save(state, ckpt_dir / f'{tag}_last.pt')
        if run_name is not None:
            torch.save(state, ckpt_dir / 'unet_last.pt')
        if dice > best:
            best = dice
            torch.save(state, ckpt_dir / f'{tag}_best.pt')
            if run_name is not None:
                torch.save(state, ckpt_dir / 'unet_best.pt')

    save_json(history, Path(cfg['paths']['results']) / f'{tag}_seg_training.json')
    if tag == 'unet':
        save_json(history, Path(cfg['paths']['results']) / 'seg_training.json')
    save_loss_plot(history, Path(cfg['paths']['figures']) / f'{tag}_seg_training.png', f'{arch} ({loss_name}, {size}px)')
    return history
