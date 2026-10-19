"""MobileNetV2 with a 4-class head."""
import torch.nn as nn
from torchvision import models

from config import CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    # Original classifier = Dropout + Linear(1280 -> 1000). Replace the Linear.
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze / unfreeze the feature extractor (everything except classifier)."""
    for p in model.features.parameters():
        p.requires_grad = trainable
