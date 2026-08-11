from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class DINOv3BinaryClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        input_dim = int(backbone.embed_dim) * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone.forward_features(images)
        cls_token = outputs["x_norm_clstoken"]
        patch_mean = outputs["x_norm_patchtokens"].mean(dim=1)
        features = torch.cat([cls_token, patch_mean], dim=1)
        return self.classifier(features)


def build_model(repo: Path, weights: Path, dropout: float = 0.1) -> DINOv3BinaryClassifier:
    repo = repo.resolve()
    weights = weights.resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"Official DINOv3 repository not found: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 ViT-L/16 weight file not found: {weights}")
    backbone = torch.hub.load(
        str(repo),
        "dinov3_vitl16",
        source="local",
        pretrained=True,
        weights=str(weights),
    )
    model = DINOv3BinaryClassifier(backbone, dropout=dropout)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def parameter_summary(model: nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total,
    }
