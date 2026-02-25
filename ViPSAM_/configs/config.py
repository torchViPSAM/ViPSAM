from dataclasses import dataclass, fields

@dataclass
class Cfg:
    sam_ckpt: str
    img_size: int = 1024
    embed_dim: int = 768
    out_chans: int = 256
    fusion_heads: int = 8

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 4
    val_split: float = 0.1
    use_bce: bool = False
    prompt_type: str = "box"
    seed: int = 42

    use_sampler: bool = False
    lesion_boost: float = 1.0
    sampler_epoch_len: int = 0

    use_lora: bool = False # True: use LoRA, False: use full model
    lora_rank: int = 8
    lora_alpha: int = 8

def cfg_from_dict(d: dict) -> Cfg:
    names = {f.name for f in fields(Cfg)}
    kwargs = {k: v for k, v in d.items() if k in names}
    if "medsam_ckpt" in d and "sam_ckpt" not in kwargs:
        kwargs["sam_ckpt"] = d["medsam_ckpt"]
    return Cfg(**kwargs)