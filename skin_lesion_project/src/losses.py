import torch
import torch.nn.functional as F

def dice_loss_logits(logits,targets,eps=1e-6):
    probs=torch.sigmoid(logits); dims=(1,2,3)
    inter=(probs*targets).sum(dims); den=probs.sum(dims)+targets.sum(dims)
    return (1-(2*inter+eps)/(den+eps)).mean()

def segmentation_loss(logits,targets,kind='bce_dice'):
    bce=F.binary_cross_entropy_with_logits(logits,targets)
    if kind=='bce': return bce
    dice=dice_loss_logits(logits,targets)
    if kind=='dice': return dice
    if kind=='bce_dice': return bce+dice
    raise ValueError(f'Loss không hỗ trợ: {kind}')
