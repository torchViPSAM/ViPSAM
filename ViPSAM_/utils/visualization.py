import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_numpy_img(x):
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = x.transpose(1, 2, 0)  
    return x

def _to_numpy_mask(x):
    if x.shape[1] == 1:
        x = torch.sigmoid(x)
    x = x.detach().cpu().squeeze(0).clamp(0, 1).numpy()  
    return x

def save_epoch_visual_overlay(ct, gt_liver, gt_lesion, pred_liver, pred_lesion, out_png, image_size=1024):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    ct_np = _to_numpy_img(ct)
    h, w = ct_np.shape[:2]

    gt_l = gt_liver.detach().cpu().squeeze().numpy().reshape(h, w)
    gt_le = gt_lesion.detach().cpu().squeeze().numpy().reshape(h, w)
    pr_l = np.squeeze(_to_numpy_mask(pred_liver)).reshape(h, w)
    pr_le = np.squeeze(_to_numpy_mask(pred_lesion)).reshape(h, w)

    liver_color = np.array([1, 0, 0, 0.6])
    lesion_color = np.array([0, 1, 0, 0.6])

    gt_overlay = np.zeros((h, w, 4), dtype=np.float32)
    gt_overlay[gt_l > 0.5] = liver_color
    gt_overlay[gt_le > 0.5] = lesion_color

    pred_overlay = np.zeros((h, w, 4), dtype=np.float32)
    pred_overlay[pr_l > 0.5] = liver_color
    pred_overlay[pr_le > 0.5] = lesion_color

    plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(ct_np, cmap='gray')
    ax.set_title("CT")
    ax.axis("off")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(ct_np, cmap='gray')
    ax.imshow(gt_overlay)
    ax.set_title("CT + GT (Liver + Lesion)")
    ax.axis("off")

    ax = plt.subplot(1, 3, 3)
    ax.imshow(ct_np, cmap='gray')
    ax.imshow(pred_overlay)
    ax.set_title("CT + Pred (Liver + Lesion)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

def save_epoch_visual(ct, mr, gt, pred_logits, out_png, image_size=1024, class_name="liver"):    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    ct_np = _to_numpy_img(ct)
    mr_np = _to_numpy_img(mr)
    gt_np = gt.detach().cpu().squeeze(0).numpy()
    pr_np = _to_numpy_mask(pred_logits)

    if class_name == "liver":
        mask_color = [1, 0, 0, 0.6] 
        title_suffix = " (Liver)"
    elif class_name == "lesion":
        mask_color = [0, 1, 0, 0.6]  
        title_suffix = " (Lesion)"
    else:
        mask_color = [0, 0, 1, 0.6]  
        title_suffix = ""

    plt.figure(figsize=(16, 4))
    
    ax = plt.subplot(1, 4, 1)
    ax.imshow(ct_np, cmap='gray')
    ax.set_title("CT")
    ax.axis("off")
    
    ax = plt.subplot(1, 4, 2)
    ax.imshow(mr_np, cmap='gray')
    ax.set_title("MR")
    ax.axis("off")
    
    ax = plt.subplot(1, 4, 3)
    ax.imshow(ct_np, cmap='gray')
    gt_colored = np.zeros((*gt_np.shape, 4))
    gt_mask = gt_np > 0.5
    gt_colored[gt_mask] = mask_color
    ax.imshow(gt_colored)
    ax.set_title(f"GT{title_suffix}")  
    ax.axis("off")
    
    ax = plt.subplot(1, 4, 4)
    ax.imshow(ct_np, cmap='gray')
    pred_colored = np.zeros((*pr_np.shape, 4))
    pred_mask = pr_np > 0.5
    pred_colored[pred_mask] = mask_color
    ax.imshow(pred_colored)
    ax.set_title(f"Pred{title_suffix}")  
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

def update_overlay_and_save(overlay_cache, cache_key, ct, gt_vis, pred_vis, class_name, vis_dir, image_size):
    if cache_key not in overlay_cache:
        overlay_cache[cache_key] = {"ct": None, "gt_liver": None, "pred_liver": None, "gt_lesion": None, "pred_lesion": None}
    overlay_cache[cache_key]["ct"] = ct.detach()
    overlay_cache[cache_key]["gt_" + class_name] = gt_vis.detach()
    overlay_cache[cache_key]["pred_" + class_name] = pred_vis.detach()

    entry = overlay_cache[cache_key]
    if entry["gt_liver"] is not None and entry["gt_lesion"] is not None:
        pk, si = cache_key
        case_id_vis = f"{pk}_slice{si:03d}" if si >= 0 else pk
        out_png = os.path.join(vis_dir, f"case_{case_id_vis}_overlay.png")
        save_epoch_visual_overlay(
            entry["ct"], entry["gt_liver"], entry["gt_lesion"],
            entry["pred_liver"], entry["pred_lesion"],
            out_png=out_png, image_size=image_size,
        )
        del overlay_cache[cache_key]
        return True
    return False