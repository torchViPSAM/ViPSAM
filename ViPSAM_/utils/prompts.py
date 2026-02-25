import random
import numpy as np
import torch

def generate_box_prompt(mask, bbox_shift=20, image_size=None, fixed_shift=None):
    if image_size is None:
        if len(mask.shape) == 4:
            _, _, H, W = mask.shape
        else:
            _, H, W = mask.shape
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
                shift_x = shift_y = fixed_shift
            elif bbox_shift > 0:
                shift_x = random.randint(0, bbox_shift)
                shift_y = random.randint(0, bbox_shift)
            else:
                shift_x = shift_y = 0
            x_min = max(0, x_min - shift_x)
            x_max = min(W, x_max + shift_x)
            y_min = max(0, y_min - shift_y)
            y_max = min(H, y_max + shift_y)
            boxes.append([x_min, y_min, x_max, y_max])
        else:
            H, W = mask[b].shape
            boxes.append([0, 0, W, H])
    return torch.tensor(boxes)


def generate_click_prompt(mask, pt_label=1, image_size=None):
    if image_size is None:
        if len(mask.shape) == 4:
            _, _, H, W = mask.shape
        else:
            _, H, W = mask.shape
        image_size = H
    if len(mask.shape) == 4:
        mask = mask[:, 0, :, :]
    B, H, W = mask.shape
    points_list = []
    labels_list = []
    for b in range(B):
        mask_2d = mask[b].detach().cpu().numpy()
        mask_binary = (mask_2d > 0.5).astype(np.uint8)
        y_indices, x_indices = np.where(mask_binary > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            idx = random.randint(0, len(x_indices) - 1)
            x_point, y_point = float(x_indices[idx]), float(y_indices[idx])
        else:
            x_point, y_point = float(W // 2), float(H // 2)
        points_list.append([x_point, y_point])
        labels_list.append(pt_label)
    points = torch.tensor(points_list, dtype=torch.float32, device=mask.device).unsqueeze(1)
    point_labels = torch.tensor(labels_list, dtype=torch.long, device=mask.device).unsqueeze(1)
    return points, point_labels