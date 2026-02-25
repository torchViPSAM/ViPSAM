import torch, torch.nn as nn
from typing import Type

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class CrossAttentionFusion(nn.Module):
    def __init__(self, c=256, heads=8, mlp_dim=1024):
        super().__init__()
        self.ct_ln = nn.LayerNorm(c)
        self.mri_ln = nn.LayerNorm(c)
        self.mha = nn.MultiheadAttention(c, heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.3))
        
        self.mlp = MLPBlock(c, mlp_dim, act=nn.GELU)
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, ct, mri):
        B, C, H, W = ct.shape
        q = ct.flatten(2).transpose(1, 2)
        k = mri.flatten(2).transpose(1, 2)
        v = mri.flatten(2).transpose(1, 2)
        
        q = self.ct_ln(q)
        k = self.mri_ln(k)
        v = self.mri_ln(v)
        
        attn_out, _ = self.mha(q, k, v)
        fused = self.norm1(q + self.gate * attn_out)
        
        mlp_out = self.mlp(fused)
        fused = self.norm2(fused + mlp_out)
        
        ct_flat = ct.flatten(2).transpose(1, 2)
        fused = self.residual_weight * ct_flat + (1 - self.residual_weight) * fused
        
        return fused.transpose(1, 2).reshape(B, C, H, W)

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_dim = base.in_features
        out_dim = base.out_features

        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

def add_lora_to_attention(attn, rank=8, alpha=8, target="all"):
    if target == "all":
        attn.q_proj = LoRALinear(attn.q_proj, rank, alpha)
        attn.k_proj = LoRALinear(attn.k_proj, rank, alpha)
        attn.v_proj = LoRALinear(attn.v_proj, rank, alpha)
        attn.out_proj = LoRALinear(attn.out_proj, rank, alpha)
        print(f"Added LoRA to all of attention layers")
    elif target =="kv_out":
        attn.k_proj = LoRALinear(attn.k_proj, rank, alpha)
        attn.v_proj = LoRALinear(attn.v_proj, rank, alpha)
        attn.out_proj = LoRALinear(attn.out_proj, rank, alpha)
        print(f"Added LoRA to KV_out of attention layers")
    elif target == "kv_only": 
        attn.k_proj = LoRALinear(attn.k_proj, rank, alpha)
        attn.v_proj = LoRALinear(attn.v_proj, rank, alpha)
    elif target == "q_out": 
        attn.q_proj = LoRALinear(attn.q_proj, rank, alpha)
        attn.out_proj = LoRALinear(attn.out_proj, rank, alpha)
        print(f"Added LoRA to Q_out of attention layers")
    return attn

def apply_lora_to_decoder(mask_decoder, rank=8, alpha=8):
    for layer in mask_decoder.transformer.layers:
        add_lora_to_attention(layer.cross_attn_token_to_image, rank, alpha, target="kv_out")
        add_lora_to_attention(layer.cross_attn_image_to_token, rank, alpha, target="q_out")
    add_lora_to_attention(mask_decoder.transformer.final_attn_token_to_image, rank, alpha, target="kv_out")
    return mask_decoder

def get_lora_params(mask_decoder):
    lora_params = []
    for module in mask_decoder.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params