import numpy as np

def binary_metrics(pred,target,eps=1e-7):
    p=np.asarray(pred).astype(bool); t=np.asarray(target).astype(bool)
    inter=np.logical_and(p,t).sum(); union=np.logical_or(p,t).sum(); ps=p.sum(); ts=t.sum()
    dice=(2*inter+eps)/(ps+ts+eps); iou=(inter+eps)/(union+eps)
    return {'dice':float(dice),'iou':float(iou),'thresholded_jaccard_065':float(iou if iou>=.65 else 0.0)}

def box_iou(a,b,eps=1e-7):
    ax1,ay1,ax2,ay2=map(float,a); bx1,by1,bx2,by2=map(float,b)
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1)); inter=iw*ih
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); ab=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/(aa+ab-inter+eps)

def detection_ap(predictions,targets,iou_threshold=0.5):
    # AP 101 điểm nội suy, một lớp đối tượng.
    preds=[]; total_gt=0
    for image_idx,(p,t) in enumerate(zip(predictions,targets)):
        gt=t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
        total_gt+=len(gt)
        if len(p):
            boxes=p['boxes'].detach().cpu().numpy(); scores=p['scores'].detach().cpu().numpy()
            for box,score in zip(boxes,scores): preds.append((float(score),image_idx,box))
    if total_gt==0: return 0.0
    preds.sort(key=lambda z:z[0],reverse=True); matched=set(); tp=[]; fp=[]
    for score,i,box in preds:
        gt=targets[i].detach().cpu().numpy() if hasattr(targets[i],'detach') else np.asarray(targets[i])
        best=-1; best_iou=0.0
        for j,g in enumerate(gt):
            value=box_iou(box,g)
            if value>best_iou: best_iou=value; best=j
        if best>=0 and best_iou>=iou_threshold and (i,best) not in matched:
            tp.append(1); fp.append(0); matched.add((i,best))
        else:
            tp.append(0); fp.append(1)
    if not tp: return 0.0
    tp=np.cumsum(tp); fp=np.cumsum(fp)
    recall=tp/total_gt; precision=tp/np.maximum(tp+fp,1)
    precision_envelope=np.maximum.accumulate(precision[::-1])[::-1]
    samples=np.linspace(0.0,1.0,101)
    values=[float(precision_envelope[recall>=r].max()) if np.any(recall>=r) else 0.0 for r in samples]
    return float(np.mean(values))

def detection_precision_recall(predictions,targets,score_threshold=0.5,iou_threshold=0.5):
    tp=fp=0; total_gt=0
    for pred,target in zip(predictions,targets):
        boxes=pred['boxes'].detach().cpu().numpy(); scores=pred['scores'].detach().cpu().numpy()
        gt=target.detach().cpu().numpy() if hasattr(target,'detach') else np.asarray(target)
        total_gt+=len(gt); order=np.argsort(-scores); used=set()
        for idx in order:
            if scores[idx] < score_threshold: continue
            best=-1; best_iou=0.0
            for j,g in enumerate(gt):
                v=box_iou(boxes[idx],g)
                if v>best_iou: best_iou=v; best=j
            if best>=0 and best_iou>=iou_threshold and best not in used: tp+=1; used.add(best)
            else: fp+=1
    fn=max(0,total_gt-tp)
    return {'precision_at_score_0_5_iou_0_5':float(tp/(tp+fp)) if tp+fp else 0.0,'recall_at_score_0_5_iou_0_5':float(tp/(tp+fn)) if tp+fn else 0.0}


def segmentation_summary(rows):
    if not rows: return {'count':0}
    keys=('dice','iou','thresholded_jaccard_065')
    out={f'mean_{k}':float(np.mean([r[k] for r in rows])) for k in keys}
    out['failure_rate_dice_below_0_65']=float(np.mean([r['dice']<0.65 for r in rows]))
    out['count']=len(rows)
    return out
