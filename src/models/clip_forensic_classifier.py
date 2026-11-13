#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP-ViT Forensic Classifier (UnivFD + RINE Multi-Layer + Patch Top-K Pooling)
==============================================================================
Designed for fast training on consumer GPUs (e.g., RTX 5060 8GB):
- Freezes the pretrained CLIP Vision Transformer (or torchvision ViT) backbone
- Extracts both global [CLS] representations and spatial Patch Token representations
  across intermediate + final transformer blocks
- Uses Top-K Patch Anomaly Pooling alongside [CLS] so localized edits (like OpenCut
  handwriting/text replacement covering <5% of the image) are not diluted by global averaging
- Optional fusion with 6-dim SRM Noise & Reference-Debiasing statistics to suppress
  false positives on simple backgrounds (white walls, clean paper documents)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

# Default to hf-mirror for mainland network stability
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMResidualExtractor(nn.Module):
    """Fixed (non-trainable) 3-filter Spatial Rich Model (SRM) high-pass layer."""

    def __init__(self) -> None:
        super().__init__()
        k1 = [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]]
        k2 = [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]]
        k3 = [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]]
        kernels = torch.tensor([k1, k2, k3], dtype=torch.float32)
        kernels[0] /= 2.0
        kernels[1] /= 2.0
        kernels[2] /= 4.0
        # Shape: (3, 1, 3, 3) applied to luminance channel
        self.register_buffer("weight", kernels.unsqueeze(1))

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: Normalized or [0,1] RGB tensor of shape (B, 3, H, W)
        Returns:
            6-dim statistical vector per image: (B, 6) capturing global mean/std,
            patch-variance disparity (sensitive to OpenCut inpainting boxes), and
            90-degree rotation self-reference delta.
        """
        gray = 0.299 * x_rgb[:, 0:1] + 0.587 * x_rgb[:, 1:2] + 0.114 * x_rgb[:, 2:3]
        res = F.conv2d(gray, self.weight, padding=1).abs().clamp(max=5.0)  # (B, 3, H, W)

        # Global mean & std per image
        feat_mean = res.mean(dim=(1, 2, 3), keepdim=False)
        feat_std = res.std(dim=(1, 2, 3), keepdim=False)

        # Patch-level noise variance disparity (captures localized "noise vacuum" from OpenCut)
        patch_energy = F.avg_pool2d(res.mean(dim=1, keepdim=True), kernel_size=16, stride=16)
        patch_max = patch_energy.amax(dim=(2, 3)).squeeze(1)
        patch_min = patch_energy.amin(dim=(2, 3)).squeeze(1)
        patch_std = patch_energy.std(dim=(2, 3)).squeeze(1)

        # 90-degree rotation self-reference difference (RDD debiasing signal)
        gray_rot = torch.rot90(gray, k=1, dims=(2, 3))
        res_rot = F.conv2d(gray_rot, self.weight, padding=1).abs().clamp(max=5.0)
        rot_delta = (res.mean(dim=(1, 2, 3)) - res_rot.mean(dim=(1, 2, 3))).abs()

        return torch.stack([feat_mean, feat_std, patch_max, patch_min, patch_std, rot_delta], dim=1)


class CLIPForensicClassifier(nn.Module):
    """
    UnivFD-style CLIP-ViT Forensic Classifier enhanced with:
    1. Multi-layer intermediate feature tap (RINE principle)
    2. Patch-token Top-K pooling (captures localized AI edits / OpenCut handwriting tampering)
    3. SRM & rotation-debiased statistical side-channel fusion
    """

    def __init__(
        self,
        backbone_name: str = "openai/clip-vit-base-patch16",
        num_classes: int = 2,
        top_k_patches: int = 8,
        use_srm_sidechannel: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.top_k_patches = top_k_patches
        self.use_srm_sidechannel = use_srm_sidechannel

        self.use_hf_clip = False
        self.embed_dim = 768

        if backbone_name.startswith("torchvision/"):
            import torchvision.models as tvm
            tv_arch = backbone_name.split("/", 1)[1]
            if tv_arch == "vit_b_16":
                weights = tvm.ViT_B_16_Weights.DEFAULT
                self.backbone = tvm.vit_b_16(weights=weights)
                self.embed_dim = self.backbone.hidden_dim
                self.backbone.heads = nn.Identity()
            else:
                raise ValueError(f"Unsupported torchvision ViT backbone: {tv_arch}")
        else:
            from transformers import CLIPVisionModel
            self.backbone = CLIPVisionModel.from_pretrained(backbone_name)
            self.embed_dim = self.backbone.config.hidden_size
            self.use_hf_clip = True

        # Freeze the ViT backbone parameters (UnivFD paradigm: fast training, strong generalization)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.srm_extractor = SRMResidualExtractor() if use_srm_sidechannel else None
        srm_dim = 6 if use_srm_sidechannel else 0

        # Trainable Patch Anomaly Scorer (1x1 projection over patch tokens to find tampered patches)
        self.patch_scorer = nn.Linear(self.embed_dim, 1)

        # Feature vector combines:
        # [CLS] final (D) + [CLS] mid-layer (D) + Top-K patch mean (D) + SRM stats (6)
        in_features = self.embed_dim * 3 + srm_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def train(self, mode: bool = True) -> "CLIPForensicClassifier":
        super().train(mode)
        # Keep frozen backbone in eval mode at all times
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_vit_tokens(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_final: (B, D) final layer [CLS] token
            cls_mid:   (B, D) intermediate layer [CLS] token
            patches:   (B, N, D) final layer spatial patch tokens
        """
        if self.use_hf_clip:
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (B, 1+N, D)
            mid_idx = len(hidden_states) // 2
            final_tokens = hidden_states[-1]
            mid_tokens = hidden_states[mid_idx]

            cls_final = final_tokens[:, 0, :]
            cls_mid = mid_tokens[:, 0, :]
            patches = final_tokens[:, 1:, :]
            return cls_final, cls_mid, patches
        else:
            # Torchvision ViT fallback
            x = self.backbone._process_input(pixel_values)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = x + self.backbone.encoder.pos_embedding
            x = self.backbone.encoder.dropout(x)

            num_layers = len(self.backbone.encoder.layers)
            mid_idx = num_layers // 2
            cls_mid = None
            for idx, layer in enumerate(self.backbone.encoder.layers):
                x = layer(x)
                if idx == mid_idx:
                    cls_mid = x[:, 0, :]
            x = self.backbone.encoder.ln(x)
            cls_final = x[:, 0, :]
            if cls_mid is None:
                cls_mid = cls_final
            patches = x[:, 1:, :]
            return cls_final, cls_mid, patches

    def forward(
        self, pixel_values: torch.Tensor, return_patch_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: Normalized image tensor (B, 3, 224, 224)
            return_patch_scores: Whether to return spatial patch anomaly weights (for localizing edits)
        """
        with torch.no_grad():
            cls_final, cls_mid, patches = self.extract_vit_tokens(pixel_values)
            srm_feats = self.srm_extractor(pixel_values) if self.srm_extractor is not None else None

        # Score each spatial patch token to catch localized tampering (OpenCut / inpainting)
        patch_logits = self.patch_scorer(patches).squeeze(-1)  # (B, N)
        k = min(self.top_k_patches, patches.shape[1])
        topk_vals, topk_idx = torch.topk(patch_logits, k=k, dim=1)  # (B, k)
        topk_weights = F.softmax(topk_vals, dim=1).unsqueeze(-1)  # (B, k, 1)

        # Gather Top-K suspicious patch embeddings
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        topk_patches = torch.gather(patches, dim=1, index=idx_expanded)  # (B, k, D)
        patch_pooled = (topk_patches * topk_weights).sum(dim=1)  # (B, D)

        feats = [cls_final, cls_mid, patch_pooled]
        if srm_feats is not None:
            feats.append(srm_feats)

        combined = torch.cat(feats, dim=1)
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=1)

        out = {
            "logits": logits,
            "probs": probs,
            "ai_prob": probs[:, 1:].sum(dim=1),  # Total probability of AI-generated or AI-modified
        }
        if return_patch_scores:
            out["patch_scores"] = torch.sigmoid(patch_logits)
        return out
