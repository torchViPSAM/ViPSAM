import random
import numpy as np
import nibabel as nib
from skimage import transform


def load_volume(path):
    return nib.load(path).get_fdata().astype(np.float32)

def resize2d(x2d, size, is_mask=False):
    return transform.resize(
        x2d,
        (size, size),
        order=0 if is_mask else 3,
        preserve_range=True,
        anti_aliasing=False if is_mask else True,
    ).astype(np.float32)

def norm_ct(x2d):
    x2d = np.clip(x2d, -100, 200)
    return ((x2d + 100) / 300.0).astype(np.float32)

def norm_mr(x2d):
    mn, mx = float(x2d.min()), float(x2d.max())
    return ((x2d - mn) / (mx - mn + 1e-8)).astype(np.float32)

def to_3c(hw):
    return np.repeat(hw[None, ...], 3, axis=0).astype(np.float32)

def nonempty_slices(mask_vol):
    valid = np.any(mask_vol > 0, axis=(0, 1))
    return np.flatnonzero(valid).astype(int)

def crop_bbox_from_ct(ct_vol, valid_slices, crop_thr, crop_margin):
    H, W, _ = ct_vol.shape
    if valid_slices.size == 0:
        return (0, H - 1, 0, W - 1)
    ct_sub = ct_vol[:, :, valid_slices]
    nz = np.argwhere(ct_sub > crop_thr)
    if nz.size == 0:
        return (0, H - 1, 0, W - 1)
    x0, x1 = int(nz[:, 0].min()), int(nz[:, 0].max())
    y0, y1 = int(nz[:, 1].min()), int(nz[:, 1].max())
    x0 = max(0, x0 - crop_margin)
    y0 = max(0, y0 - crop_margin)
    x1 = min(H - 1, x1 + crop_margin)
    y1 = min(W - 1, y1 + crop_margin)
    return (x0, x1, y0, y1)

def apply_crop(x2d, bbox):
    x0, x1, y0, y1 = bbox
    return x2d[x0 : x1 + 1, y0 : y1 + 1]

def apply_augmentation(ct2d, mr2d, mk2d):
    if random.random() > 0.5:
        ct2d, mr2d, mk2d = np.fliplr(ct2d), np.fliplr(mr2d), np.fliplr(mk2d)
    if random.random() > 0.5:
        ct2d, mr2d, mk2d = np.flipud(ct2d), np.flipud(mr2d), np.flipud(mk2d)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        ct2d = transform.rotate(ct2d, angle, resize=False, preserve_range=True, order=3)
        mr2d = transform.rotate(mr2d, angle, resize=False, preserve_range=True, order=3)
        mk2d = transform.rotate(mk2d, angle, resize=False, preserve_range=True, order=0)
    return ct2d, mr2d, mk2d