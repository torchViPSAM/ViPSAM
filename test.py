import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.join(_script_dir, "ViPSAM_")
if os.path.isdir(_code_dir) and _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)
import numpy as np
import torch
import argparse
import json
from datetime import datetime
from collections import OrderedDict
from torch.utils.data import DataLoader

from config import Cfg, cfg_from_dict
from module import CrossAttentionFusion
from model import CTMRISegModel
from dataset import CTMRDataset
from utils import scan_cases,round_floats,get_test_loader
from tester import validation_loop

def parse_args():
    parser = argparse.ArgumentParser(description='ViPSAM Test')
    
    parser.add_argument('--test_ct_mr_root', type=str, default="./test")
    parser.add_argument('--test_mask_root', type=str, default="./test/mask")

    parser.add_argument('--sam_ckpt', type=str, default="./ViPSAM/work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--prompt_type', type=str, default=None,choices=["box", "point"])
    
    parser.add_argument('--results_dir', type=str, default="./ViPSAM/ViPSAM_/output/result")
    parser.add_argument('--model_name', type=str, default=None)
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_device', type=int, default=0)
    
    parser.add_argument('--use_lora', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=8)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})

    cfg = cfg_from_dict(checkpoint_config)
    cfg.sam_ckpt = args.sam_ckpt
    cfg.img_size = checkpoint_config.get("img_size", args.image_size)
    cfg.prompt_type = args.prompt_type or checkpoint.get("prompt_type", "box")
    cfg.use_lora = checkpoint_config.get("use_lora", args.use_lora)
    cfg.lora_rank = checkpoint_config.get("lora_rank", args.lora_rank)
    cfg.lora_alpha = checkpoint_config.get("lora_alpha", args.lora_alpha)
    cfg.use_bce = checkpoint_config.get("use_bce", True)

    fusion = CrossAttentionFusion(c=cfg.out_chans, heads=cfg.fusion_heads)
    model = CTMRISegModel(cfg, fusion).to(device)
        
    start_epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('best_dice', 0.0)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    test_loader = get_test_loader(
        args.test_ct_mr_root,
        args.test_mask_root,
        image_size=cfg.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    
    if args.model_name is None:
        checkpoint_name = os.path.basename(args.checkpoint_path).replace('.pth', '')
        args.model_name = f"test_{checkpoint_name}"
    
    save_dir = os.path.join(args.results_dir, args.model_name)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    test_results = validation_loop(
        model=model,
        val_loader=test_loader,
        cfg=cfg,
        device=device,
        prompt_type=cfg.prompt_type,
        vis_dir=vis_dir,
        epoch=None,
        save_vis=True
    )
    
    summary = {
        "experiment_name": args.model_name,
        "checkpoint": args.checkpoint_path,
        "epoch": start_epoch,
        "test_data_path": args.test_ct_mr_root,
        "image_size": cfg.img_size,
        "prompt_type": cfg.prompt_type,
        "metrics": {
            "overall": {
                "total_loss": float(test_results["loss"]),
                "iou": {
                    "mean": float(test_results["iou"]),
                    "std": float(test_results["iou_std"])
                },
                "dice": {
                    "mean": float(test_results["dice"]),
                    "std": float(test_results["dice_std"])
                },
                "hd95": {  
                    "mean": float(test_results["hd95"]),
                    "std": float(test_results["hd95_std"])
                }
            },
            "liver": {
                "iou": {
                    "mean": float(test_results.get("iou_liver", 0.0)),
                    "std": float(test_results.get("iou_std_liver", 0.0))
                },
                "dice": {
                    "mean": float(test_results.get("dice_liver", 0.0)),
                    "std": float(test_results.get("dice_std_liver", 0.0))
                },
                "hd95": {  
                    "mean": float(test_results.get("hd95_liver", 0.0)),
                    "std": float(test_results.get("hd95_std_liver", 0.0))
                },
                "num_samples": int(test_results.get("num_samples_liver", 0))
            },
            "lesion": {
                "iou": {
                    "mean": float(test_results.get("iou_lesion", 0.0)),
                    "std": float(test_results.get("iou_std_lesion", 0.0))
                },
                "dice": {
                    "mean": float(test_results.get("dice_lesion", 0.0)),
                    "std": float(test_results.get("dice_std_lesion", 0.0))
                },
                "hd95": {  
                    "mean": float(test_results.get("hd95_lesion", 0.0)),
                    "std": float(test_results.get("hd95_std_lesion", 0.0))
                },
                "num_samples": int(test_results.get("num_samples_lesion", 0))
            }
        },
        "num_slices": len(test_loader),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    json_path = os.path.join(save_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(round_floats(summary, 3), f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'test done!')
    print(f'{"="*60}')
    print(f'\nEach class results:')
    print(f'   Liver:')
    print(f'     IOU:  {test_results.get("iou_liver", 0.0):.4f} ± {test_results.get("iou_std_liver", 0.0):.4f})')
    print(f'     DICE: {test_results.get("dice_liver", 0.0):.4f} ± {test_results.get("dice_std_liver", 0.0):.4f}')
    print(f'     HD95: {test_results.get("hd95_liver", 0.0):.4f} ± {test_results.get("hd95_std_liver", 0.0):.4f}') 

    print(f'   Lesion:')
    print(f'     IOU:  {test_results.get("iou_lesion", 0.0):.4f} ± {test_results.get("iou_std_lesion", 0.0):.4f})')
    print(f'     DICE: {test_results.get("dice_lesion", 0.0):.4f} ± {test_results.get("dice_std_lesion", 0.0):.4f}')
    print(f'     HD95: {test_results.get("hd95_lesion", 0.0):.4f} ± {test_results.get("hd95_std_lesion", 0.0):.4f}')

if __name__ == "__main__":
    main()
