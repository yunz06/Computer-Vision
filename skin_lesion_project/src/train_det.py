from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import DetectionDataset, detection_collate
from .prepare import load_records
from .models import build_detector
from .metrics import detection_ap, detection_ap5095
from .utils import device, save_json, save_loss_plot


def run(
    cfg,
    epochs=None,
    limit=0,
    pretrained=None,
    model_name=None,
    batch_size=None,
    learning_rate=None,
    trainable_backbone_layers=None,
    augment=None,
    run_name=None,
):
    split_path = Path(cfg['paths']['splits']) / 'split.json'
    train_records = load_records(split_path, 'train')
    val_records = load_records(split_path, 'val')

    if limit:
        train_records = train_records[:limit]
        val_records = val_records[:max(1, limit // 4)] if val_records else []

    det_cfg = cfg['detection']
    data_cfg = cfg.get('data', {})
    det_size = cfg['detection_size']
    mask_thresh = data_cfg.get('mask_threshold', 127)
    min_area = data_cfg.get('min_box_area_ratio', 0.001)
    multi_comp = data_cfg.get('multi_component_boxes', True)
    use_augment = det_cfg.get('augment', True) if augment is None else augment
    seed = cfg.get('seed', 42)

    cache_dir = cfg['paths'].get('processed')
    train_ds = DetectionDataset(
        train_records,
        size=det_size,
        augment=use_augment,
        mask_threshold=mask_thresh,
        min_area_ratio=min_area,
        multi_component=multi_comp,
        seed=seed,
        cache_dir=cache_dir,
    )
    val_ds = DetectionDataset(
        val_records,
        size=det_size,
        augment=False,
        mask_threshold=mask_thresh,
        min_area_ratio=min_area,
        multi_component=multi_comp,
        seed=seed,
        cache_dir=cache_dir,
    )

    if len(train_ds) == 0:
        raise RuntimeError('Không có mask dương hợp lệ cho Detection.')

    dev = device()
    use_pretrained = det_cfg['pretrained'] if pretrained is None else pretrained
    arch = model_name or det_cfg['model']
    tb_layers = (
        det_cfg.get('trainable_backbone_layers', None)
        if trainable_backbone_layers is None
        else trainable_backbone_layers
    )
    use_amp = dev.type == 'cuda' and bool(det_cfg.get('amp', True))

    model = build_detector(
        pretrained=use_pretrained,
        model_name=arch,
        image_size=det_size,
        trainable_backbone_layers=tb_layers,
        box_nms_thresh=det_cfg.get('nms_iou_threshold', 0.5),
        box_detections_per_img=det_cfg.get('max_detections', 5),
    ).to(dev)

    bs = batch_size or det_cfg.get('batch_size', 2)
    lr = learning_rate or det_cfg['learning_rate']

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=det_cfg.get('num_workers', 0),
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=det_cfg.get('momentum', 0.9),
        weight_decay=det_cfg.get('weight_decay', 0.0005),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = []
    best_ap5095 = -1.0
    total_epochs = epochs or det_cfg['epochs']

    checkpoint_dir = Path(cfg['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(cfg['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path(cfg['paths']['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    tag = run_name or 'fasterrcnn'

    print(f'Thiết bị: {dev}')
    print(f'Mô hình: {arch}')
    print(f'Kích thước ảnh: {det_size}')
    print(f'Tập train: {len(train_ds)} mẫu | {len(train_loader)} batches/epoch')
    print(f'Tập validation: {len(val_ds)} mẫu')
    print(f'Batch size: {bs} | AMP: {use_amp}')
    print(f'Số epoch: {total_epochs}')

    for epoch in range(1, total_epochs + 1):
        model.train()
        losses = []

        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)

        train_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f'Epoch {epoch}/{total_epochs} [Train]',
            dynamic_ncols=True,
        )

        for images, targets in train_bar:
            images = [x.to(dev) for x in images]
            targets = [
                {k: v.to(dev) for k, v in target.items()}
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                train_bar.set_postfix(status='Loss không hợp lệ')
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            current_loss = float(loss.detach().cpu())
            losses.append(current_loss)

            train_bar.set_postfix(
                loss=f'{current_loss:.4f}',
                avg_loss=f'{sum(losses) / len(losses):.4f}',
            )

        train_bar.close()

        max_vram_mb = (
            round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 1)
            if dev.type == 'cuda'
            else 0.0
        )

        model.eval()
        predictions = []
        ground_truth = []

        val_bar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f'Epoch {epoch}/{total_epochs} [Val]',
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for images, targets in val_bar:
                output = model([images[0].to(dev)])[0]

                predictions.append({
                    key: value.detach().cpu()
                    for key, value in output.items()
                })
                ground_truth.append(targets[0]['boxes'].cpu())

        val_bar.close()

        val_ap50 = (
            detection_ap(predictions, ground_truth, 0.5)
            if ground_truth
            else 0.0
        )
        val_ap5095 = (
            detection_ap5095(predictions, ground_truth)
            if ground_truth
            else 0.0
        )

        row = {
            'epoch': epoch,
            'train_loss': sum(losses) / max(1, len(losses)),
            'val_ap50': val_ap50,
            'val_ap5095': val_ap5095,
            'max_vram_mb': max_vram_mb,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_ap50={val_ap50:.4f} | "
            f"val_ap5095={val_ap5095:.4f} | "
            f"VRAM={max_vram_mb:.1f} MB",
            flush=True,
        )

        state = {
            'model': model.state_dict(),
            'epoch': epoch,
            'model_name': arch,
            'pretrained': use_pretrained,
            'image_size': det_size,
            'trainable_backbone_layers': tb_layers,
            'val_ap50': val_ap50,
            'val_ap5095': val_ap5095,
            'seed': seed,
        }

        torch.save(state, checkpoint_dir / f'{tag}_last.pt')

        if run_name is not None:
            torch.save(state, checkpoint_dir / 'fasterrcnn_last.pt')

        if val_ap5095 > best_ap5095:
            best_ap5095 = val_ap5095
            torch.save(state, checkpoint_dir / f'{tag}_best.pt')

            if run_name is not None:
                torch.save(state, checkpoint_dir / 'fasterrcnn_best.pt')

    save_json(
        history,
        results_dir / f'{tag}_det_training.json',
    )

    if tag == 'fasterrcnn':
        save_json(history, results_dir / 'det_training.json')

    save_loss_plot(
        history,
        figures_dir / f'{tag}_det_training.png',
        f'{arch} ({det_size}px)',
    )

    print(f'Hoàn tất huấn luyện. Checkpoint tốt nhất: {checkpoint_dir / f"{tag}_best.pt"}')
    return history
