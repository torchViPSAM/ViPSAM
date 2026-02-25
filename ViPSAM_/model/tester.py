import os
import numpy as np
import torch
import torch.nn.functional as F

from .metrics_loss import seg_loss, dice_coeff, iou_score, hd95_score
from utils import generate_box_prompt, generate_click_prompt, save_epoch_visual, update_overlay_and_save
from data import patient_key_from_case_id

MAX_VIS_OVERLAYS = 5

def validation_loop(model, val_loader, cfg, device, vis_dir=None, epoch=None, save_vis=True):
    model.eval()
    va_loss, va_dice, va_iou = 0.0, 0.0, 0.0
    all_dices = []
    all_ious = []
    all_hd95s = []

    liver_dices = []
    liver_ious = []
    liver_hd95s = []
    lesion_dices = []
    lesion_ious = []
    lesion_hd95s = []

    per_sample_metrics = []

    image_size = cfg.img_size
    prompt_type = cfg.prompt_type

    num_overlays_saved = 0
    saved_cases = {}

    with torch.no_grad():
        first_batch = True
        sample_idx = 0
        overlay_cache = {}

        for batch_idx, batch in enumerate(val_loader):
            ct, mr = batch["ct"].to(device), batch["mr"].to(device)
            gt = batch["mask"].float().to(device)
            spacing = batch["spacing"].to(device)
            s = batch["slice_idx"]
            class_names = batch.get("class_name", [])
            case_ids = batch.get("case_id", [])
            batch_size = ct.shape[0]

            slice_indices = s.cpu().tolist() if s.dim() > 0 else [int(s.item())]

            patient_keys = [patient_key_from_case_id(cid) for cid in case_ids]

            if prompt_type == "box":
                boxes = generate_box_prompt(gt, bbox_shift=0, image_size=image_size, fixed_shift=5)
                boxes = boxes.to(device)
                points = None
                point_labels = None
            else:
                points, point_labels = generate_click_prompt(gt, pt_label=1, image_size=image_size)
                points = points.to(device)
                point_labels = point_labels.to(device)
                boxes = None

            logits, _ = model(ct, mr, boxes=boxes, points=points, point_labels=point_labels, masks=None)
            gt_256 = F.interpolate(gt, size=(256, 256), mode="nearest")

            loss = seg_loss(logits, gt_256, use_bce=cfg.use_bce).item()
            va_loss += loss

            dice_sample = dice_coeff(logits, gt_256)
            iou_sample = iou_score(logits, gt_256)
            hd95_sample = hd95_score(logits, gt_256, spacing)

            all_dices.extend(dice_sample.cpu().numpy().tolist())
            all_ious.extend(iou_sample.cpu().numpy().tolist())
            all_hd95s.extend(hd95_sample.cpu().numpy().tolist())

            dice_list = dice_sample.cpu().numpy().tolist()
            iou_list = iou_sample.cpu().numpy().tolist()
            hd95_list = hd95_sample.cpu().numpy().tolist()

            for i, class_name in enumerate(class_names):
                if i < len(dice_list):
                    pk = patient_keys[i] if i < len(patient_keys) else ""
                    per_sample_metrics.append((pk, class_name, dice_list[i], iou_list[i], hd95_list[i]))
                    if class_name == "liver":
                        liver_dices.append(dice_list[i])
                        liver_ious.append(iou_list[i])
                        liver_hd95s.append(hd95_list[i])
                    elif class_name == "lesion":
                        lesion_dices.append(dice_list[i])
                        lesion_ious.append(iou_list[i])
                        lesion_hd95s.append(hd95_list[i])

            should_save = False
            if save_vis and vis_dir is not None and num_overlays_saved < MAX_VIS_OVERLAYS:
                if epoch is not None:
                    should_save = first_batch
                else:
                    should_save = True

            if should_save:
                logits_vis = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=False)
                gt_vis = F.interpolate(gt, size=(image_size, image_size), mode="nearest")

                for b in range(batch_size):
                    if epoch is not None and (batch_idx > 0 or b > 0):
                        break
                    if num_overlays_saved >= MAX_VIS_OVERLAYS:
                        break

                    if epoch is None:
                        if b >= len(patient_keys) or b >= len(class_names) or b >= len(slice_indices):
                            continue
                        patient_key = patient_keys[b]
                        class_name = class_names[b]
                        case_id = case_ids[b]
                        slice_idx = slice_indices[b]
                        key = (patient_key, class_name)
                        if key not in saved_cases:
                            saved_cases[key] = []
                        si = int(slice_idx) if slice_idx is not None else -1
                        if (case_id, si if si >= 0 else None) in saved_cases[key]:
                            continue
                        saved_cases[key].append((case_id, si if si >= 0 else None))

                        cache_key = (patient_key, si)
                        did_save = update_overlay_and_save(
                            overlay_cache, cache_key,
                            ct[b], gt_vis[b, 0], logits_vis[b, 0].unsqueeze(0),
                            class_name, vis_dir, image_size,
                        )
                        if did_save:
                            num_overlays_saved += 1
                        continue
                    else:
                        class_name_vis = class_names[b] if b < len(class_names) else "unknown"
                        case_id_vis = case_ids[b] if b < len(case_ids) else f"case_{sample_idx + b}"

                    if epoch is not None:
                        out_png = os.path.join(vis_dir, f"epoch_{epoch:03d}_{class_name_vis}.png")
                        save_epoch_visual(
                            ct[b], mr[b], gt_vis[b, 0], logits_vis[b, 0].unsqueeze(0),
                            out_png=out_png,
                            image_size=image_size,
                            class_name=class_name_vis,
                        )
                        break

            sample_idx += batch_size
            first_batch = False

    num_batches = max(1, len(val_loader))
    va_loss /= num_batches

    va_dice = np.mean(all_dices) if len(all_dices) > 0 else 0.0
    va_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    va_hd95 = np.nanmean(all_hd95s) if len(all_hd95s) > 0 else 0.0
    dice_std = np.std(all_dices) if len(all_dices) > 1 else 0.0
    iou_std = np.std(all_ious) if len(all_ious) > 1 else 0.0
    hd95_std = np.nanstd(all_hd95s) if len(all_hd95s) > 1 else 0.0

    va_dice_liver = np.mean(liver_dices) if len(liver_dices) > 0 else 0.0
    va_iou_liver = np.mean(liver_ious) if len(liver_ious) > 0 else 0.0
    va_hd95_liver = np.nanmean(liver_hd95s) if len(liver_hd95s) > 0 else 0.0
    dice_std_liver = np.std(liver_dices) if len(liver_dices) > 1 else 0.0
    iou_std_liver = np.std(liver_ious) if len(liver_ious) > 1 else 0.0
    hd95_std_liver = np.nanstd(liver_hd95s) if len(liver_hd95s) > 1 else 0.0

    va_dice_lesion = np.mean(lesion_dices) if len(lesion_dices) > 0 else 0.0
    va_iou_lesion = np.mean(lesion_ious) if len(lesion_ious) > 0 else 0.0
    va_hd95_lesion = np.nanmean(lesion_hd95s) if len(lesion_hd95s) > 0 else 0.0
    dice_std_lesion = np.std(lesion_dices) if len(lesion_dices) > 1 else 0.0
    iou_std_lesion = np.std(lesion_ious) if len(lesion_ious) > 1 else 0.0
    hd95_std_lesion = np.nanstd(lesion_hd95s) if len(lesion_hd95s) > 1 else 0.0

    return {
        "loss": va_loss,
        "dice": va_dice,
        "iou": va_iou,
        "hd95": va_hd95,
        "dice_std": dice_std,
        "iou_std": iou_std,
        "hd95_std": hd95_std,
        "dice_liver": va_dice_liver,
        "iou_liver": va_iou_liver,
        "hd95_liver": va_hd95_liver,
        "dice_std_liver": dice_std_liver,
        "iou_std_liver": iou_std_liver,
        "hd95_std_liver": hd95_std_liver,
        "dice_lesion": va_dice_lesion,
        "iou_lesion": va_iou_lesion,
        "hd95_lesion": va_hd95_lesion,
        "dice_std_lesion": dice_std_lesion,
        "iou_std_lesion": iou_std_lesion,
        "hd95_std_lesion": hd95_std_lesion,
        "num_samples_liver": len(liver_dices),
        "num_samples_lesion": len(lesion_dices),
        "per_sample_metrics": per_sample_metrics,
    }
