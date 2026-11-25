from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import DetectionDataset, detection_collate
from .prepare import load_records
from .models import build_detector
from .metrics import detection_ap
from .utils import device, save_json, save_loss_plot


def run(cfg, epochs=None, limit=0, pretrained=None, model_name=None):
    split_path = Path(cfg['paths']['splits']) / 'split.json'
    train_records = load_records(split_path, 'train')
    val_records = load_records(split_path, 'val')
    if limit:
        train_records = train_records[:limit]
        val_records = val_records[:max(1, limit // 4)]

    train_ds = DetectionDataset(train_records, cfg['detection_size'])
    val_ds = DetectionDataset(val_records, cfg['detection_size'])
    if len(train_ds) == 0:
        raise RuntimeError('Không có mask dương hợp lệ cho Detection.')

    dev = device()
    use_pretrained = cfg['detection']['pretrained'] if pretrained is None else pretrained
    model_name = model_name or cfg['detection']['model']
    model = build_detector(use_pretrained, model_name, cfg['detection_size']).to(dev)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=detection_collate)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['detection']['learning_rate'], momentum=0.9, weight_decay=0.0005
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda' and cfg['detection'].get('amp', True)))
    history = []
    best_ap = -1.0
    total_epochs = epochs or cfg['detection']['epochs']
    checkpoint_dir = Path(cfg['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = [x.to(dev) for x in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=(dev.type == 'cuda')):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        predictions, ground_truth = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                output = model([images[0].to(dev)])[0]
                predictions.append({k: v.detach().cpu() for k, v in output.items()})
                ground_truth.append(targets[0]['boxes'].cpu())
        val_ap50 = detection_ap(predictions, ground_truth, 0.5) if ground_truth else 0.0
        row = {'epoch': epoch, 'train_loss': sum(losses) / max(1, len(losses)), 'val_ap50': val_ap50}
        history.append(row)
        print(row)

        state = {'model': model.state_dict(), 'epoch': epoch, 'model_name': model_name, 'pretrained': use_pretrained, 'val_ap50': val_ap50}
        torch.save(state, checkpoint_dir / 'fasterrcnn_last.pt')
        if val_ap50 > best_ap:
            best_ap = val_ap50
            torch.save(state, checkpoint_dir / 'fasterrcnn_best.pt')

    save_json(history, Path(cfg['paths']['results']) / 'det_training.json')
    save_loss_plot(history, Path(cfg['paths']['figures']) / 'det_training.png', 'Faster R-CNN training')
    return history
