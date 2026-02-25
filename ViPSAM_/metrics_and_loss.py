import torch, torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice

def iou_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou

def soft_dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

def hd95_score(logits: torch.Tensor,
                       target: torch.Tensor,
                       spacing_b2: torch.Tensor,
                       connectivity: int = 1) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5)
    gt = (target > 0.5)

    B = pred.shape[0]
    struct = generate_binary_structure(2, connectivity)

    hd = np.empty(B, dtype=np.float32)

    spacing_np = spacing_b2.detach().cpu().numpy()  

    for b in range(B):
        P = pred[b, 0].detach().cpu().numpy()
        T = gt[b, 0].detach().cpu().numpy()
        sp = (float(spacing_np[b, 0]), float(spacing_np[b, 1]))

        p_any, t_any = P.any(), T.any()

        if (not p_any) and (not t_any):
            hd[b] = 0.0
            continue
        if (not p_any) or (not t_any):
            hd[b] = np.nan
            continue

        P_surf = P ^ binary_erosion(P, structure=struct, border_value=0)
        T_surf = T ^ binary_erosion(T, structure=struct, border_value=0)

        dt_T = distance_transform_edt(~T_surf, sampling=sp)
        dt_P = distance_transform_edt(~P_surf, sampling=sp)

        d1 = dt_T[P_surf]
        d2 = dt_P[T_surf]

        hd1 = np.percentile(d1, 95) if d1.size else 0.0
        hd2 = np.percentile(d2, 95) if d2.size else 0.0
        hd[b] = max(hd1, hd2)

    return torch.from_numpy(hd).to(device=logits.device, dtype=torch.float32)

def seg_loss(logits, target, use_bce=True, eps=1e-6):
    dice_loss = soft_dice_loss(logits, target, eps=eps)
    if not use_bce:
        return dice_loss
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice_loss + bce
