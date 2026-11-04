import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoHEALVisionTower(nn.Module):
    """The supplied DINO-HEAL tower, adapted to nanoVLM's SigLIP tiles."""

    def __init__(self, vision_tower, cfg, dino_model=None, feature_extractor=None):
        super().__init__()
        self.vision_tower = vision_tower
        self.cfg = cfg
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        self.dinov2_feature_extractor = feature_extractor
        self.dinov2_model = dino_model

        if dino_model is None and feature_extractor is None:
            self.load_model()
        elif dino_model is None or feature_extractor is None:
            raise ValueError("Provide both DINO model and processor, or neither.")
        else:
            self.dinov2_model.requires_grad_(False)
            self.dinov2_model.eval()

    def load_model(self):
        try:
            from transformers import AutoImageProcessor, AutoModel

            self.dinov2_feature_extractor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-large"
            )
            dino_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.dinov2_model = AutoModel.from_pretrained(
                "facebook/dinov2-large", torch_dtype=dino_dtype,
                attn_implementation="eager",
            )
            self.dinov2_model.requires_grad_(False)
            self.dinov2_model.eval()
            self.dinov2_model.to(device=next(self.vision_tower.parameters()).device)
        except Exception as error:
            # Preserve the supplied implementation's visible baseline fallback.
            print(f"[DinoHEALVisionTower] Warning: failed to load DINOv2: {error}")
            self.dinov2_feature_extractor = None
            self.dinov2_model = None

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        image_features = self.vision_tower(images).to(images.dtype)
        if self.dinov2_model is None or self.dinov2_feature_extractor is None:
            return image_features

        # nanoVLM's image processor supplies RGB tiles in [0, 1], not CLIP-normalized pixels.
        images_pil = self._to_pil_list(images.detach().float().cpu().clamp(0, 1))
        dino_inputs = self.dinov2_feature_extractor(images=images_pil, return_tensors="pt")
        parameter = next(self.dinov2_model.parameters())
        dino_inputs = {
            key: value.to(device=parameter.device, dtype=parameter.dtype)
            if value.is_floating_point() else value.to(device=parameter.device)
            for key, value in dino_inputs.items()
        }
        dino_outputs = self.dinov2_model(
            **dino_inputs, output_hidden_states=True, output_attentions=True, return_dict=True
        )
        dino_atts = dino_outputs.attentions
        if dino_atts is None:
            raise RuntimeError("DINOv2 did not return attention weights; eager attention is required.")
        last_att = dino_atts[-1] if isinstance(dino_atts, (list, tuple)) else dino_atts
        if last_att.dim() == 5:
            last_att = last_att[-1]
        att = last_att.mean(dim=1)
        if att.size(1) < 2 or att.size(2) < 2:
            return image_features
        cls_to_patches = att[:, 0, 1:]

        batch = images.shape[0]
        height, width = self._infer_dino_grid(cls_to_patches, dino_inputs)
        dino_saliency_2d = cls_to_patches.view(batch, height, width)
        dino_saliency_2d = (dino_saliency_2d - dino_saliency_2d.amin(dim=(1, 2), keepdim=True)) / (
            dino_saliency_2d.amax(dim=(1, 2), keepdim=True)
            - dino_saliency_2d.amin(dim=(1, 2), keepdim=True) + 1e-6
        )

        grid = self.cfg.vit_img_size // self.cfg.vit_patch_size
        dino_saliency_up = F.interpolate(
            dino_saliency_2d.unsqueeze(1), size=(grid, grid),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        dino_saliency_flat = torch.sigmoid(dino_saliency_up.reshape(batch, -1))
        if image_features.ndim != 3 or image_features.size(1) != grid * grid:
            raise ValueError("Expected nanoVLM patch features without a CLS token.")

        image_features_norm = (image_features - image_features.mean()) / (image_features.std() + 1e-6)
        saliency_expanded = dino_saliency_flat.unsqueeze(-1).repeat(1, 1, image_features.size(-1))
        return image_features_norm * 0.3 + saliency_expanded * 0.7

    @staticmethod
    def _to_pil_list(pixel_values):
        from torchvision.transforms.functional import to_pil_image
        return [to_pil_image(image) for image in pixel_values]

    def _infer_dino_grid(self, cls_to_patches, dino_inputs):
        count = cls_to_patches.shape[1]
        patch = self.dinov2_model.config.patch_size
        height, width = dino_inputs["pixel_values"].shape[-2:]
        height, width = height // patch, width // patch
        if height * width == count:
            return height, width
        side = int(math.sqrt(count))
        if side * side == count:
            return side, side
        raise ValueError("Cannot determine DINOv2 patch grid from attention output.")


class DinoHEAL(nn.Module):
    """Wrap an already-loaded nanoVLM; the original source files stay unchanged."""

    def __init__(self, model, dino_model=None, feature_extractor=None):
        super().__init__()
        if model.cfg.vit_cls_flag:
            raise ValueError("This nanoVLM adapter expects patch features without CLS.")
        self.model = model
        model.vision_encoder = DinoHEALVisionTower(
            model.vision_encoder, model.cfg, dino_model, feature_extractor
        )

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
