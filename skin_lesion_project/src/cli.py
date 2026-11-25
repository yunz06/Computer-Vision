import argparse
from pathlib import Path
from .config import load_config, ensure_dirs
from .utils import seed_everything


def main():
    p = argparse.ArgumentParser(description='ISIC 2018 Detection + Segmentation')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('audit')

    ed = sub.add_parser('eda')
    ed.add_argument('--limit', type=int, default=0)

    pr = sub.add_parser('prepare')
    pr.add_argument('--sample-size', type=int, default=0)
    pr.add_argument('--seed', type=int, default=42)
    pr.add_argument('--metadata', type=str, default=None)
    pr.add_argument('--use-phash', action='store_true', default=None)
    pr.add_argument('--no-phash', action='store_true')

    bs = sub.add_parser('baselines')
    bs.add_argument('--limit', type=int, default=0)
    bs.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='test')

    qs = sub.add_parser('train-seg')
    qs.add_argument('--epochs', type=int, default=0)
    qs.add_argument('--limit', type=int, default=0)
    qs.add_argument('--model', choices=['unet', 'unet_resnet34', 'attention_unet'], default=None)
    qs.add_argument('--loss', choices=['bce', 'dice', 'bce_dice'], default=None)
    qs.add_argument('--size', type=int, default=0)
    qs.add_argument('--batch-size', type=int, default=0)
    qs.add_argument('--lr', type=float, default=0.0)
    qs.add_argument('--weights', choices=['pretrained', 'none'], default='pretrained')
    qs.add_argument('--no-augment', action='store_true')
    qs.add_argument('--run-name', type=str, default=None)
    qs.add_argument('--seed', type=int, default=None)

    qd = sub.add_parser('train-det')
    qd.add_argument('--epochs', type=int, default=0)
    qd.add_argument('--limit', type=int, default=0)
    qd.add_argument('--weights', choices=['pretrained', 'none'], default='pretrained')
    qd.add_argument('--model', choices=['fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_fpn'], default=None)
    qd.add_argument('--batch-size', type=int, default=0)
    qd.add_argument('--lr', type=float, default=0.0)
    qd.add_argument('--trainable-backbone-layers', type=int, default=None)
    qd.add_argument('--no-augment', action='store_true')
    qd.add_argument('--run-name', type=str, default=None)
    qd.add_argument('--seed', type=int, default=None)

    sv = sub.add_parser('select-val')
    sv.add_argument('--limit', type=int, default=0)

    ev = sub.add_parser('evaluate')
    ev.add_argument('--task', choices=['seg', 'det', 'unet_box'], required=True)
    ev.add_argument('--split', choices=['val', 'test'], default='test')
    ev.add_argument('--checkpoint', type=str, default=None)
    ev.add_argument('--threshold', type=float, default=None)
    ev.add_argument('--score-threshold', type=float, default=None)
    ev.add_argument('--largest-component', action='store_true', default=None)
    ev.add_argument('--hole-filling', action='store_true', default=None)

    sub.add_parser('demo')

    args = p.parse_args()
    cfg = load_config()
    if getattr(args, 'seed', None) is not None:
        cfg['seed'] = args.seed
    ensure_dirs(cfg)
    seed_everything(cfg['seed'])

    data_cfg = cfg.get('data', {})
    mask_thresh = data_cfg.get('mask_threshold', 127)
    min_area = data_cfg.get('min_box_area_ratio', 0.001)

    if args.cmd == 'audit':
        from .audit import run_audit
        r = run_audit(cfg['paths']['images'], cfg['paths']['masks'], cfg['paths']['reports'], mask_threshold=mask_thresh)
        print(r)
    elif args.cmd == 'eda':
        from .eda import run_eda
        r = run_eda(
            cfg['paths']['images'], cfg['paths']['masks'], cfg['paths']['reports'], cfg['paths']['figures'],
            mask_threshold=mask_thresh, min_area_ratio=min_area, limit=args.limit,
        )
        print(r)
    elif args.cmd == 'prepare':
        from .prepare import prepare_splits
        use_phash = data_cfg.get('use_phash_grouping', True)
        if args.no_phash:
            use_phash = False
        elif args.use_phash is True:
            use_phash = True
        r = prepare_splits(
            cfg['paths']['images'], cfg['paths']['masks'], cfg['paths']['splits'],
            sample_size=args.sample_size, seed=args.seed,
            train=cfg['split']['train'], val=cfg['split']['val'],
            mask_threshold=mask_thresh, min_area_ratio=min_area,
            metadata_path=args.metadata, use_phash=use_phash,
            phash_max_distance=data_cfg.get('phash_max_distance', 4),
        )
        print({k: len(v) for k, v in r.items() if isinstance(v, list)})
    elif args.cmd == 'baselines':
        from .baselines import run
        from .prepare import load_records
        split_path = Path(cfg['paths']['splits']) / 'split.json'
        records = None
        if split_path.exists():
            if args.split == 'all':
                records = []
                for part in ('train', 'val', 'test'):
                    records.extend(load_records(split_path, part))
            else:
                records = load_records(split_path, args.split)
        result = run(
            cfg['paths']['images'], cfg['paths']['masks'], cfg['paths']['results'],
            args.limit, cfg['image_size'], records,
            mask_threshold=mask_thresh, min_area_ratio=min_area,
            boundary_tolerance_px=cfg.get('evaluation', {}).get('boundary_tolerance_px', 2),
            cache_dir=cfg['paths'].get('processed'),
        )
        print('Số mẫu:', len(result))
    elif args.cmd == 'train-seg':
        from .train_seg import run
        run(
            cfg,
            epochs=args.epochs or None,
            limit=args.limit,
            model_name=args.model,
            loss_kind=args.loss,
            image_size=args.size or None,
            batch_size=args.batch_size or None,
            learning_rate=args.lr or None,
            augment=False if args.no_augment else None,
            pretrained=(args.weights != 'none'),
            run_name=args.run_name,
        )
    elif args.cmd == 'train-det':
        from .train_det import run
        run(
            cfg,
            epochs=args.epochs or None,
            limit=args.limit,
            pretrained=(args.weights != 'none'),
            model_name=args.model,
            batch_size=args.batch_size or None,
            learning_rate=args.lr or None,
            trainable_backbone_layers=args.trainable_backbone_layers,
            augment=False if args.no_augment else None,
            run_name=args.run_name,
        )
    elif args.cmd == 'select-val':
        from .evaluate import select_on_validation
        print(select_on_validation(cfg, limit=args.limit))
    elif args.cmd == 'evaluate':
        from .evaluate import eval_seg, eval_det, eval_unet_box
        if args.task == 'seg':
            print(eval_seg(
                cfg, split=args.split, checkpoint=args.checkpoint,
                threshold=args.threshold, largest_component=args.largest_component,
                hole_filling=args.hole_filling,
            ))
        elif args.task == 'det':
            print(eval_det(cfg, split=args.split, checkpoint=args.checkpoint, score_threshold=args.score_threshold))
        else:
            print(eval_unet_box(cfg, split=args.split, checkpoint=args.checkpoint))
    elif args.cmd == 'demo':
        from .demo import launch
        launch(cfg)


if __name__ == '__main__':
    main()
