import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .prepare_data import load_volume,resize2d,norm_ct,norm_mr,to_3c,nonempty_slices,crop_bbox_from_ct,apply_crop,apply_augmentation

class CTMRDataset(Dataset):
    def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, augment_lesion_only=False):
        self.data_list = data_list
        self.image_size = int(image_size)
        self.crop_thr = float(crop_thr)
        self.crop_margin = int(crop_margin)
        self.augment_lesion_only = augment_lesion_only
        self.case_bbox = {}
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for case_idx, case in enumerate(self.data_list):
            mask_vol = load_volume(case["label"])
            valid_z = nonempty_slices(mask_vol)
            if valid_z.size == 0:
                continue
            ct_vol = load_volume(case["ct"])
            bbox = crop_bbox_from_ct(ct_vol, valid_z, self.crop_thr, self.crop_margin)
            self.case_bbox[case_idx] = bbox
            for z in valid_z:
                samples.append((case_idx, int(z)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, z = self.samples[idx]
        case = self.data_list[case_idx]
        bbox = self.case_bbox[case_idx]

        ct_nii = nib.load(case["ct"])
        sx, sy = ct_nii.header.get_zooms()[:2]

        ct_vol = load_volume(case["ct"])
        mr_vol = load_volume(case["mr"])
        mk_vol = load_volume(case["label"])

        ct2d = ct_vol[:, :, z]
        mr2d = mr_vol[:, :, z]
        mk2d = (mk_vol[:, :, z] > 0).astype(np.float32)

        ct2d = apply_crop(ct2d, bbox)
        mr2d = apply_crop(mr2d, bbox)
        mk2d = apply_crop(mk2d, bbox)

        crop_h, crop_w = ct2d.shape
        metric_size = 256
        spacing = (float(sx) * (crop_w / metric_size), float(sy) * (crop_h / metric_size))

        ct2d = resize2d(ct2d, self.image_size, is_mask=False)
        mr2d = resize2d(mr2d, self.image_size, is_mask=False)
        mk2d = resize2d(mk2d, self.image_size, is_mask=True)

        if self.augment_lesion_only and case.get("class_name") == "lesion":
            ct2d, mr2d, mk2d = apply_augmentation(ct2d, mr2d, mk2d)

        ct3 = to_3c(norm_ct(ct2d))
        mr3 = to_3c(norm_mr(mr2d))
        mask = (mk2d > 0.5).astype(np.float32)

        return {
            "case_id": case["case_id"],
            "class_name": case.get("class_name", "unknown"),
            "slice_idx": torch.tensor(z, dtype=torch.long),
            "ct": torch.from_numpy(ct3),
            "mr": torch.from_numpy(mr3),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }