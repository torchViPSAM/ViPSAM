import torch, torch.nn as nn
from segment_anything import sam_model_registry
from module import apply_lora_to_decoder

class CTMRISegModel(nn.Module):
    def __init__(self, cfg, fusion):
        super().__init__()
        image_size = getattr(cfg, "img_size", 1024)
        encoder_ct = sam_model_registry["vit_b"](checkpoint=cfg.sam_ckpt, image_size=image_size)
        encoder_mri = sam_model_registry["vit_b"](checkpoint=cfg.sam_ckpt, image_size=image_size)
        
        self.ct_encoder = encoder_ct.image_encoder
        self.prompt_encoder = encoder_ct.prompt_encoder
        self.mask_decoder = encoder_ct.mask_decoder
        self.mri_encoder = encoder_mri.image_encoder
        self.fusion = fusion
        self.cfg = cfg
        self.image_size = image_size
        self.register_buffer(
            "pixel_mean", 
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", 
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        )
        for p in self.ct_encoder.parameters():
            p.requires_grad = False
        for p in self.mri_encoder.parameters():
            p.requires_grad = False
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.mask_decoder.parameters():
            p.requires_grad = False
        
        if cfg.use_lora:
            self.mask_decoder = apply_lora_to_decoder(
                self.mask_decoder, rank=cfg.lora_rank, alpha=cfg.lora_alpha
            )
        
        for p in self.fusion.parameters():
            p.requires_grad = True

    def _preprocess_sam(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 255.0
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def forward(self, ct_img, mri_img, boxes=None, points=None, point_labels=None, masks=None):
        ct_img = self._preprocess_sam(ct_img)
        mri_img = self._preprocess_sam(mri_img)
        ct_feat = self.ct_encoder(ct_img)      
        mri_feat = self.mri_encoder(mri_img)   
        img_embed = self.fusion(ct_feat, mri_feat)

        if boxes is not None:
            boxes = boxes.float() 
            if len(boxes.shape) == 2:
                boxes = boxes.unsqueeze(1) 
        else:
            boxes = None
        
        if points is not None and point_labels is not None:
            point_coords = points 
            point_labels_tensor = point_labels  
        else:
            point_coords = None
            point_labels_tensor = None

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels_tensor) if point_coords is not None else None,
                boxes=boxes,
                masks=masks,
            )

        logits, iou = self.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return logits, iou



