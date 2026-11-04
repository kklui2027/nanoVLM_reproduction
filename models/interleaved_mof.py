import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modality_projector import ModalityProjector


class DINOVisionTower(nn.Module):
    """DINOv2 branch from MMVP, adapted to nanoVLM's image tiles."""

    def __init__(self, model_name="dinov2_vitl14", dino_model=None):
        super().__init__()
        self.vision_tower = dino_model or torch.hub.load(
            "facebookresearch/dinov2", model_name
        )
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

    @torch.no_grad()
    def forward(self, images, grid_size):
        # nanoVLM's 512px tiles are not divisible by DINOv2's 14px patches.
        # Resize the same image tile so both towers produce the same grid.
        dino_size = (grid_size[0] * 14, grid_size[1] * 14)
        images = F.interpolate(images.float(), size=dino_size, mode="bilinear", align_corners=False)
        parameter = next(self.vision_tower.parameters())
        outputs = self.vision_tower.forward_features(
            images.to(device=parameter.device, dtype=parameter.dtype)
        )
        return outputs["x_prenorm"][:, 1:].to(dtype=images.dtype)


class InterleavedVisionTower(nn.Module):
    def __init__(self, vision_tower, dino_tower, cfg):
        super().__init__()
        self.vision_tower = vision_tower
        self.dino_tower = dino_tower
        self.cfg = cfg
        self.vision_tower.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        image_features = self.vision_tower(images)
        grid = self.cfg.vit_img_size // self.cfg.vit_patch_size
        if image_features.ndim != 3 or image_features.shape[1] != grid * grid:
            raise ValueError("Expected nanoVLM patch features without a CLS token.")
        dino_features = self.dino_tower(images, (grid, grid))
        if dino_features.shape[:2] != image_features.shape[:2]:
            raise ValueError("SigLIP and DINO patch grids do not match.")
        return image_features, dino_features.to(image_features.dtype)


class InterleavedProjector(nn.Module):
    def __init__(self, projector, dino_projector):
        super().__init__()
        self.mm_projector = projector
        self.dino_mm_projector = dino_projector

    def forward(self, image_features):
        image_features_clip, image_features_dino = image_features
        image_features_clip = self.mm_projector(image_features_clip)
        image_features_dino = self.dino_mm_projector(image_features_dino)
        if image_features_clip.shape != image_features_dino.shape:
            raise ValueError("The two projected feature sequences must have the same shape.")

        # MMVP's original I-MoF ordering: clip_1, dino_1, clip_2, dino_2, ...
        batch, num_patches, hidden_size = image_features_clip.shape
        merged_features = torch.empty(
            batch, 2 * num_patches, hidden_size,
            device=image_features_clip.device, dtype=image_features_clip.dtype,
        )
        merged_features[:, 0::2] = image_features_clip
        merged_features[:, 1::2] = image_features_dino.to(image_features_clip.dtype)
        return merged_features


class InterleavedMoF(nn.Module):
    """Wrap an already-loaded nanoVLM; no original source files are changed.

    Use a separately loaded base model: its modules are replaced in memory.
    Construct image placeholders with model.cfg.mp_image_token_length.
    """

    def __init__(self, model, dino_model=None, dino_hidden_size=1024):
        super().__init__()
        self.model = model
        cfg = model.cfg
        if cfg.vit_cls_flag:
            raise ValueError("This nanoVLM adapter expects patch features without CLS.")

        dino_tower = DINOVisionTower(dino_model=dino_model)
        base_parameter = next(model.vision_encoder.parameters())
        dino_tower.to(device=base_parameter.device)

        dino_cfg = copy.copy(cfg)
        dino_cfg.vit_hidden_dim = dino_hidden_size
        dino_projector = ModalityProjector(dino_cfg)
        projector_parameter = next(model.MP.parameters())
        dino_projector.to(device=projector_parameter.device, dtype=projector_parameter.dtype)

        model.vision_encoder = InterleavedVisionTower(model.vision_encoder, dino_tower, cfg)
        model.MP = InterleavedProjector(model.MP, dino_projector)
        model.cfg = copy.copy(cfg)
        model.cfg.mp_image_token_length = cfg.mp_image_token_length * 2

    @property
    def cfg(self):
        return self.model.cfg

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def vision_encoder(self):
        return self.model.vision_encoder

    @property
    def MP(self):
        return self.model.MP

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        return self.model(input_ids, images, attention_mask=attention_mask, targets=targets)

    def generate(self, input_ids, images, **kwargs):
        return self.model.generate(input_ids, images, **kwargs)
