import json
from configs import cfg_from_dict

def load_config(config_path, overrides=None):
    with open(config_path) as f:
        cfg = json.load(f)
    overrides = overrides or []
    for s in overrides:
        if "=" in s: 
            k, v = s.split("=", 1)
            k = k.strip()
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
            cfg[k] = v
    return cfg


def build_cfg_from_checkpoint_and_json(checkpoint, j):
    cfg = cfg_from_dict(checkpoint.get("config", {}))
    cfg.sam_ckpt = j.get("sam_ckpt", cfg.sam_ckpt)
    cfg.img_size = j.get("image_size", getattr(cfg, "img_size", 1024))
    cfg.prompt_type = j.get("prompt_type", getattr(cfg, "prompt_type", "box"))
    cfg.use_lora = j.get("use_lora", getattr(cfg, "use_lora", False))
    cfg.lora_rank = j.get("lora_rank", getattr(cfg, "lora_rank", 8))
    cfg.lora_alpha = j.get("lora_alpha", getattr(cfg, "lora_alpha", 8))
    cfg.use_bce = checkpoint.get("config", {}).get("use_bce", True)
    return cfg