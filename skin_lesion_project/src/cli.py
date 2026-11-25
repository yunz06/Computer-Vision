import argparse
from pathlib import Path
from .config import load_config,ensure_dirs
from .utils import seed_everything

def main():
    p=argparse.ArgumentParser(description='ISIC 2018 Detection + Segmentation'); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('audit')
    q=sub.add_parser('prepare'); q.add_argument('--sample-size',type=int,default=0); q.add_argument('--seed',type=int,default=42)
    b=sub.add_parser('baselines'); b.add_argument('--limit',type=int,default=0)
    for name in ('train-seg','train-det'):
        q=sub.add_parser(name); q.add_argument('--epochs',type=int,default=0); q.add_argument('--limit',type=int,default=0)
        if name=='train-det':
            q.add_argument('--weights',choices=['pretrained','none'],default='pretrained')
            q.add_argument('--model',choices=['fasterrcnn_resnet50_fpn_v2','fasterrcnn_mobilenet_v3_large_fpn'],default=None)
    e=sub.add_parser('evaluate'); e.add_argument('--task',choices=['seg','det'],required=True)
    sub.add_parser('demo')
    args=p.parse_args(); cfg=load_config(); ensure_dirs(cfg); seed_everything(cfg['seed'])
    if args.cmd=='audit':
        from .audit import run_audit
        r=run_audit(cfg['paths']['images'],cfg['paths']['masks'],cfg['paths']['reports']); print(r)
    elif args.cmd=='prepare':
        from .prepare import prepare_splits
        r=prepare_splits(cfg['paths']['images'],cfg['paths']['masks'],cfg['paths']['splits'],args.sample_size,args.seed,cfg['split']['train'],cfg['split']['val']); print({k:len(v) for k,v in r.items() if isinstance(v,list)})
    elif args.cmd=='baselines':
        from .baselines import run
        from .prepare import load_records
        split_path=Path(cfg['paths']['splits'])/'split.json'
        records=None
        if split_path.exists():
            records=[]
            for part in ('train','val','test'):
                records.extend(load_records(split_path,part))
        result=run(cfg['paths']['images'],cfg['paths']['masks'],cfg['paths']['results'],args.limit,cfg['image_size'],records)
        print('Số mẫu:',len(result))
    elif args.cmd=='train-seg':
        from .train_seg import run
        run(cfg,args.epochs or None,args.limit)
    elif args.cmd=='train-det':
        from .train_det import run
        run(cfg,args.epochs or None,args.limit,args.weights!='none',args.model)
    elif args.cmd=='evaluate':
        from .evaluate import eval_seg,eval_det
        print(eval_seg(cfg) if args.task=='seg' else eval_det(cfg))
    elif args.cmd=='demo':
        from .demo import launch
        launch(cfg)
if __name__=='__main__': main()
