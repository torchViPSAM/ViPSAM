import random
import numpy as np
import torch
import torch.nn.functional as F
import os
import csv
import matplotlib.pyplot as plt

from collections import Counter
from torch.utils.data import DataLoader
from metrics_and_loss import dice_coeff
from dataset import CTMRDataset

def generate_box_prompt(mask, bbox_shift=20, image_size=None, fixed_shift=None):
    if image_size is None:
        if len(mask.shape) == 4:
            _,_,H,W = mask.shape
        else:
            _,H,W = mask.shape
        image_size = H
        
    mask = mask[:, 0, :, :]  
    boxes = []
    for b in range(mask.shape[0]):
        coords = torch.nonzero(mask[b])
        if coords.size(0) > 0:
            y_min = coords[:, 0].min().item()
            y_max = coords[:, 0].max().item()
            x_min = coords[:, 1].min().item()
            x_max = coords[:, 1].max().item()
            
            H, W = mask[b].shape
            
            if fixed_shift is not None:
                shift_x = fixed_shift
                shift_y = fixed_shift
            elif bbox_shift > 0:
                shift_x = random.randint(0, bbox_shift)
                shift_y = random.randint(0, bbox_shift)
            else:
                shift_x = 0
                shift_y = 0
            
            x_min = max(0, x_min - shift_x)
            x_max = min(W, x_max + shift_x)
            y_min = max(0, y_min - shift_y)
            y_max = min(H, y_max + shift_y)
            
            boxes.append([x_min, y_min, x_max, y_max])
        else:
            H, W = mask[b].shape
            boxes.append([0, 0, W, H])
    return torch.tensor(boxes)  # (B, 4)

def generate_click_prompt(mask, pt_label=1,image_size=None):

    if image_size is None:
        if len(mask.shape) == 4:
            _,_,H,W = mask.shape
        else:
            _,H,W = mask.shape
        image_size = H

    if len(mask.shape) == 4:
        mask = mask[:, 0, :, :]  
    elif len(mask.shape) == 3:
        mask = mask  

    
    B, H, W = mask.shape
    points_list = []
    labels_list = []
    
    for b in range(B):
        mask_2d = mask[b].detach().cpu().numpy()  
        mask_binary = (mask_2d > 0.5).astype(np.uint8)
        
        y_indices, x_indices = np.where(mask_binary > 0)
        
        if len(x_indices) > 0 and len(y_indices) > 0:
            idx = random.randint(0, len(x_indices) - 1)
            x_point = float(x_indices[idx])
            y_point = float(y_indices[idx])
        else:
            x_point = float(W // 2)
            y_point = float(H // 2)
        
        points_list.append([x_point, y_point])
        labels_list.append(pt_label)
    
    points = torch.tensor(points_list, dtype=torch.float32, device=mask.device) 
    points = points.unsqueeze(1)  
    
    point_labels = torch.tensor(labels_list, dtype=torch.long, device=mask.device)  
    point_labels = point_labels.unsqueeze(1)  
    
    return points, point_labels

def get_test_loader(test_ct_mr_root, test_mask_root, image_size=1024, batch_size=1, num_workers=4):
    test_cases = scan_cases(test_ct_mr_root, test_mask_root)
    test_dataset = CTMRDataset(test_cases, image_size=image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader

def patient_key_from_case_id(case_id: str) -> str:
    #   base_liver
    #   base_lesion_mask_GTV...resampled
    if "_liver" in case_id:
        return case_id.split("_liver")[0]
    if "_lesion_" in case_id:
        return case_id.split("_lesion_")[0]
    return case_id

def round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(x, ndigits) for x in obj]
    return obj

def split_cases_by_patient(all_cases, val_split: float, seed: int):
    keys = [patient_key_from_case_id(c["case_id"]) for c in all_cases]
    uniq_keys = sorted(list(set(keys)))

    rng = np.random.RandomState(seed)
    rng.shuffle(uniq_keys)

    split = int(np.floor(val_split * len(uniq_keys)))
    val_keys = set(uniq_keys[:split])
    train_keys = set(uniq_keys[split:])

    train_cases = [c for c in all_cases if patient_key_from_case_id(c["case_id"]) in train_keys]
    val_cases   = [c for c in all_cases if patient_key_from_case_id(c["case_id"]) in val_keys]

    return train_cases, val_cases, len(uniq_keys), len(train_keys), len(val_keys)


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

def parse_slice_indices_from_batch(batch, batch_size, sample_idx, dataset):
    if "slice_idx" not in batch:
        return [
            (dataset[sample_idx + i].get("slice_idx") if sample_idx + i < len(dataset) else None)
            for i in range(batch_size)
        ]
    raw = batch["slice_idx"]
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
        if raw.ndim == 0:
            return [int(raw.item())] * batch_size
        return [int(x) if x is not None else None for x in raw.flatten().tolist()]
    if isinstance(raw, list):
        out = []
        for s in raw:
            if isinstance(s, torch.Tensor):
                out.append(int(s.item()) if s.numel() == 1 else None)
            elif s is None:
                out.append(None)
            else:
                out.append(int(s))
        return out
    return [int(raw)] if raw is not None else [None]
    

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

def scan_cases(image_root, mask_root):
    cases = []
    skipped_ct = 0
    skipped_mr = 0
    skipped_mask_folder = 0
    skipped_mask_file = 0
    
    
    print(f"> Image root: {image_root}")
    print(f"> Mask root: {mask_root}")
    
    date_folders = [f for f in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, f))]
    print(f"> Found {len(date_folders)} date folders")
    
    for date_folder in sorted(date_folders):
        date_path = os.path.join(image_root, date_folder)
        if not os.path.isdir(date_path):
            continue
        
        patient_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f))]
        print(f"> {date_folder}: {len(patient_folders)} patient folders found")
        
        for idx, patient_folder in enumerate(sorted(patient_folders)):
            patient_path = os.path.join(date_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue
            
            debug_this = (idx < 3)
            
            try:
                files_in_folder = os.listdir(patient_path)
                if debug_this:
                    print(f"    [{idx+1}] {patient_folder}: {len(files_in_folder)} files")
            except PermissionError:
                print(f"    Permission error: {patient_path}")
                continue
            
            ct_files = [f for f in files_in_folder 
                       if f.endswith('.nii.gz') and 'CT' in f.upper() and 'resampled' in f]
            mr_files = [f for f in files_in_folder 
                       if f.endswith('.nii.gz') and ('MR' in f.upper() or 'MRI' in f.upper()) and 'resampled' in f]
            
            if len(ct_files) == 0:
                skipped_ct += 1
                continue
            if len(mr_files) == 0:
                skipped_mr += 1
                continue
            
            ct_path = os.path.join(patient_path, ct_files[0])
            mr_path = os.path.join(patient_path, mr_files[0])
            
            # patient ID extraction
            parts = patient_folder.split('_')
            if len(parts) < 2:
                continue
            
            name_parts = []
            patient_id = None
            
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) >= 6:
                    patient_id = part
                    name_parts = parts[:i]
                    break
            
            if patient_id is None:
                if len(parts) >= 2:
                    name_parts = [parts[0]]
                    patient_id = parts[1] if parts[1].isdigit() else None
            
            if patient_id is None:
                continue
            
            name_str = '_'.join(name_parts) if name_parts else parts[0]
            mask_patient_folder = f"{name_str}_{patient_id}_mask"
            mask_date_path = os.path.join(mask_root, date_folder)
            
            if not os.path.isdir(mask_date_path):
                continue
            
            mask_patient_path = os.path.join(mask_date_path, mask_patient_folder)
            
            if not os.path.isdir(mask_patient_path):
                skipped_mask_folder += 1
                continue
            
            case_id_base = f"{date_folder}_{patient_folder}"
            has_any_mask = False
            
            liver_mask_file_options = ["mask_LIVER_resampled.nii.gz", "mask_Liver_resampled.nii.gz"]
            liver_mask_path = None
            
            for mask_file in liver_mask_file_options:
                candidate_path = os.path.join(mask_patient_path, mask_file)
                if os.path.isfile(candidate_path):
                    liver_mask_path = candidate_path
                    break
            
            if liver_mask_path and os.path.isfile(liver_mask_path):
                cases.append({
                    "case_id": f"{case_id_base}_liver",
                    "ct": ct_path,
                    "mr": mr_path,
                    "label": liver_mask_path,
                    "class_name": "liver"
                })
                has_any_mask = True
                if len(cases) <= 3:
                    print(f"    Liver case added: {case_id_base}_liver")
            
            try:
                mask_files = os.listdir(mask_patient_path)
                
                gtv_files = []
                for f in mask_files:
                    if (f.startswith('mask_GTV') and 
                        f.endswith('_resampled.nii.gz')):
                        name_before_resampled = f[:-len('_resampled.nii.gz')]
                        if 'resampled' not in name_before_resampled.lower():
                            gtv_files.append(f)
                
                for gtv_file in sorted(gtv_files):
                    lesion_mask_path = os.path.join(mask_patient_path, gtv_file)
                    
                    if os.path.isfile(lesion_mask_path):
                        cases.append({
                            "case_id": f"{case_id_base}_lesion_{gtv_file.replace('.nii.gz', '')}",
                            "ct": ct_path,
                            "mr": mr_path,
                            "label": lesion_mask_path,
                            "class_name": "lesion"
                        })
                        has_any_mask = True
                        if len(cases) <= 5:
                            print(f"    Lesion case added: {case_id_base}_lesion ({gtv_file})")
            except Exception as e:
                if debug_this:
                    print(f"    Error: {e}")
            
            if not has_any_mask:
                skipped_mask_file += 1
    
    print(f"\n> Scan statistics:")
    print(f"   > CT file not found: {skipped_ct}")
    print(f"   > MR file not found: {skipped_mr}")
    print(f"   > Mask folder not found: {skipped_mask_folder}")
    print(f"   > Mask file not found: {skipped_mask_file}")
    print(f"   > Valid cases: {len(cases)}")
    
    liver_count = sum(1 for c in cases if c.get("class_name") == "liver")
    lesion_count = sum(1 for c in cases if c.get("class_name") == "lesion")
    print(f"   > Liver cases: {liver_count}")
    print(f"   > Lesion cases: {lesion_count}")
    
    print(f"Total {len(cases)} cases scanned")
    return cases