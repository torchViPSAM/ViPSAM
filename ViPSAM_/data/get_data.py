import os
import numpy as np
from torch.utils.data import DataLoader

from .dataset import CTMRDataset


def get_test_loader(test_ct_mr_root, test_mask_root, image_size=1024, batch_size=1, num_workers=4):
    test_cases = scan_cases(test_ct_mr_root, test_mask_root)
    test_dataset = CTMRDataset(test_cases, image_size=image_size)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def patient_key_from_case_id(case_id: str) -> str:
    if "_liver" in case_id:
        return case_id.split("_liver")[0]
    if "_lesion_" in case_id:
        return case_id.split("_lesion_")[0]
    return case_id

def split_cases_by_patient(all_cases, val_split: float, seed: int):
    keys = [patient_key_from_case_id(c["case_id"]) for c in all_cases]
    uniq_keys = sorted(set(keys))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq_keys)
    split = int(np.floor(val_split * len(uniq_keys)))
    val_keys = set(uniq_keys[:split])
    train_keys = set(uniq_keys[split:])
    train_cases = [c for c in all_cases if patient_key_from_case_id(c["case_id"]) in train_keys]
    val_cases = [c for c in all_cases if patient_key_from_case_id(c["case_id"]) in val_keys]
    return train_cases, val_cases, len(uniq_keys), len(train_keys), len(val_keys)

def scan_cases(image_root, mask_root):
    cases = []
    skipped_ct = skipped_mr = skipped_mask_folder = skipped_mask_file = 0
    print(f"> Image root: {image_root}\n> Mask root: {mask_root}")
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
            debug_this = idx < 3
            try:
                files_in_folder = os.listdir(patient_path)
                if debug_this:
                    print(f"    [{idx+1}] {patient_folder}: {len(files_in_folder)} files")
            except PermissionError:
                print(f"    Permission error: {patient_path}")
                continue
            ct_files = [f for f in files_in_folder if f.endswith(".nii.gz") and "CT" in f.upper() and "resampled" in f]
            mr_files = [f for f in files_in_folder if f.endswith(".nii.gz") and ("MR" in f.upper() or "MRI" in f.upper()) and "resampled" in f]
            if len(ct_files) == 0:
                skipped_ct += 1
                continue
            if len(mr_files) == 0:
                skipped_mr += 1
                continue
            ct_path = os.path.join(patient_path, ct_files[0])
            mr_path = os.path.join(patient_path, mr_files[0])
            parts = patient_folder.split("_")
            if len(parts) < 2:
                continue
            name_parts = []
            patient_id = None
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) >= 6:
                    patient_id = part
                    name_parts = parts[:i]
                    break
            if patient_id is None and len(parts) >= 2:
                name_parts = [parts[0]]
                patient_id = parts[1] if parts[1].isdigit() else None
            if patient_id is None:
                continue
            name_str = "_".join(name_parts) if name_parts else parts[0]
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
            liver_mask_options = ["mask_LIVER_resampled.nii.gz", "mask_Liver_resampled.nii.gz"]
            liver_mask_path = None
            for m in liver_mask_options:
                cand = os.path.join(mask_patient_path, m)
                if os.path.isfile(cand):
                    liver_mask_path = cand
                    break
            if liver_mask_path:
                cases.append({"case_id": f"{case_id_base}_liver", "ct": ct_path, "mr": mr_path, "label": liver_mask_path, "class_name": "liver"})
                has_any_mask = True
                if len(cases) <= 3:
                    print(f"    Liver case added: {case_id_base}_liver")
            try:
                mask_files = os.listdir(mask_patient_path)
                gtv_files = [f for f in mask_files if f.startswith("mask_GTV") and f.endswith("_resampled.nii.gz") and "resampled" not in f[:-len("_resampled.nii.gz")].lower()]
                for gtv_file in sorted(gtv_files):
                    lesion_mask_path = os.path.join(mask_patient_path, gtv_file)
                    if os.path.isfile(lesion_mask_path):
                        cases.append({"case_id": f"{case_id_base}_lesion_{gtv_file.replace('.nii.gz', '')}", "ct": ct_path, "mr": mr_path, "label": lesion_mask_path, "class_name": "lesion"})
                        has_any_mask = True
                        if len(cases) <= 5:
                            print(f"    Lesion case added: {case_id_base}_lesion ({gtv_file})")
            except Exception as e:
                if debug_this:
                    print(f"    Error: {e}")
            if not has_any_mask:
                skipped_mask_file += 1
    print(f"\n> Scan statistics:\n   > CT file not found: {skipped_ct}\n   > MR file not found: {skipped_mr}\n   > Mask folder not found: {skipped_mask_folder}\n   > Mask file not found: {skipped_mask_file}\n   > Valid cases: {len(cases)}")
    liver_count = sum(1 for c in cases if c.get("class_name") == "liver")
    lesion_count = sum(1 for c in cases if c.get("class_name") == "lesion")
    print(f"   > Liver cases: {liver_count}\n   > Lesion cases: {lesion_count}\nTotal {len(cases)} cases scanned")
    return cases
