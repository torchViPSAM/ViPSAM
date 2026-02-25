import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage import transform

class CTMRDataset(Dataset):
    def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, augment_lesion_only=False):
        self.data_list = data_list
        self.image_size = int(image_size)
        self.crop_thr = float(crop_thr)
        self.crop_margin = int(crop_margin)
        self.augment_lesion_only = augment_lesion_only
        self.case_bbox = {}
        self.samples = self._build_samples()

    def _load(self, path):
        return nib.load(path).get_fdata().astype(np.float32)  

    def _resize2d(self, x2d, is_mask=False):
        return transform.resize(
            x2d,
            (self.image_size, self.image_size),
            order=0 if is_mask else 3,
            preserve_range=True,
            anti_aliasing=False if is_mask else True,
        ).astype(np.float32)

    def _norm_ct(self, x2d):
        x2d = np.clip(x2d, -100, 200)
        return ((x2d + 100) / 300.0).astype(np.float32) 

    def _norm_mr(self, x2d):
        mn, mx = float(x2d.min()), float(x2d.max())
        return ((x2d - mn) / (mx - mn + 1e-8)).astype(np.float32)

    def _to_3c(self, hw):
        return np.repeat(hw[None, ...], 3, axis=0).astype(np.float32)

    def _nonempty_slices(self, mask_vol):
        valid = np.any(mask_vol > 0, axis=(0, 1))  
        return np.flatnonzero(valid).astype(int)

    def _crop_bbox_from_ct(self, ct_vol, valid_slices):
        H, W, _ = ct_vol.shape
        if valid_slices.size == 0:
            return (0, H - 1, 0, W - 1)

        ct_sub = ct_vol[:, :, valid_slices]
        nz = np.argwhere(ct_sub > self.crop_thr)
        if nz.size == 0:
            return (0, H - 1, 0, W - 1)

        x0, x1 = int(nz[:, 0].min()), int(nz[:, 0].max())
        y0, y1 = int(nz[:, 1].min()), int(nz[:, 1].max())

        x0 = max(0, x0 - self.crop_margin)
        y0 = max(0, y0 - self.crop_margin)
        x1 = min(H - 1, x1 + self.crop_margin)
        y1 = min(W - 1, y1 + self.crop_margin)
        return (x0, x1, y0, y1)

    @staticmethod
    def _apply_crop(x2d, bbox):
        x0, x1, y0, y1 = bbox
        return x2d[x0:x1+1, y0:y1+1]

    def _apply_augmentation(self, ct2d, mr2d, mk2d):
        if random.random() > 0.5:
            ct2d = np.fliplr(ct2d)
            mr2d = np.fliplr(mr2d)
            mk2d = np.fliplr(mk2d)
        if random.random() > 0.5:
            ct2d = np.flipud(ct2d)
            mr2d = np.flipud(mr2d)
            mk2d = np.flipud(mk2d)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            ct2d = transform.rotate(ct2d, angle, resize=False, preserve_range=True, order=3)
            mr2d = transform.rotate(mr2d, angle, resize=False, preserve_range=True, order=3)
            mk2d = transform.rotate(mk2d, angle, resize=False, preserve_range=True, order=0)
        return ct2d, mr2d, mk2d

    def _build_samples(self):
        samples = []

        for case_idx, case in enumerate(self.data_list):
            mask_vol = self._load(case["label"])
            valid_z = self._nonempty_slices(mask_vol)
            if valid_z.size == 0:
                continue

            ct_vol = self._load(case["ct"])
            bbox = self._crop_bbox_from_ct(ct_vol, valid_z)
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

        ct_vol = self._load(case["ct"])
        mr_vol = self._load(case["mr"])
        mk_vol = self._load(case["label"]) 
        
        ct2d = ct_vol[:, :, z]
        mr2d = mr_vol[:, :, z]
        mk2d = (mk_vol[:, :, z] > 0).astype(np.float32)

        ct2d = self._apply_crop(ct2d, bbox)
        mr2d = self._apply_crop(mr2d, bbox)
        mk2d = self._apply_crop(mk2d, bbox)

        crop_h, crop_w = ct2d.shape
        metric_size = 256
        spacing = (float(sx) * (crop_w / metric_size),
                    float(sy) * (crop_h / metric_size))

        ct2d = self._resize2d(ct2d, is_mask=False)
        mr2d = self._resize2d(mr2d, is_mask=False)
        mk2d = self._resize2d(mk2d, is_mask=True)

        class_name = case.get("class_name", "unknown")
        if self.augment_lesion_only and class_name == "lesion":
            ct2d, mr2d, mk2d = self._apply_augmentation(ct2d, mr2d, mk2d)

        ct2d = self._norm_ct(ct2d)
        mr2d = self._norm_mr(mr2d)

        ct3 = self._to_3c(ct2d)
        mr3 = self._to_3c(mr2d)
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
