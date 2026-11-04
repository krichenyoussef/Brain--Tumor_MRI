"""Model builders and the layer Grad-CAM should look at for each backbone."""
from __future__ import annotations

import timm
import torch
from torch import nn


def build_model(timm_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """A timm backbone with a fresh classification head (transfer learning)."""
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def count_params(model: nn.Module) -> float:
    """Trainable parameters, in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def is_vit(timm_name: str) -> bool:
    return timm_name.startswith(("vit", "deit"))


def gradcam_target(model: nn.Module, timm_name: str) -> nn.Module:
    """Last spatial layer, where Grad-CAM gets a meaningful heatmap."""
    if is_vit(timm_name):
        return model.blocks[-1].norm1
    if timm_name.startswith("resnet"):
        return model.layer4[-1]
    if timm_name.startswith("efficientnet"):
        return model.conv_head
    raise ValueError(f"No Grad-CAM target defined for {timm_name}")


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Turn ViT tokens (B, 1+H*W, C) into a feature map (B, C, H, W), dropping the class token."""
    tokens = tensor[:, 1:, :]
    side = int(tokens.shape[1] ** 0.5)
    return tokens.reshape(tensor.size(0), side, side, tensor.size(2)).permute(0, 3, 1, 2)
