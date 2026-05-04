"""src/models/segmentation.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def _make_layer_dilated(layer: nn.Module, dilation: int) -> nn.Module:
    """ResNet layer의 stride를 제거하고 dilation 적용 (output stride 유지)."""
    for module in layer.modules():
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (3, 3):
                module.dilation = (dilation, dilation)
                module.padding = (dilation, dilation)
                module.stride = (1, 1)
            elif module.stride == (2, 2):  # downsample 1x1 conv
                module.stride = (1, 1)
    return layer


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_ch: int, out_ch: int, rates: list[int]) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            for r in rates
        ])

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            conv1x1(out_ch * (len(rates) + 2), out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = F.interpolate(self.global_pool(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        feats = [pool, self.branch1(x)] + [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class SemanticSegmentationModel(nn.Module):
    """
    ResNet101 (output stride=16) + ASPP + DeepLabV3+ Decoder

    Standard DeepLabV3+: layer1(1/4) skip connection만 사용.
    layer2 mid-level fusion 제거 → decoder 안정성 향상.
    """
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()

        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 1/4,  256ch  ← skip connection
        self.layer2 = backbone.layer2   # 1/8,  512ch
        self.layer3 = backbone.layer3   # 1/16, 1024ch
        self.layer4 = _make_layer_dilated(backbone.layer4, dilation=2)  # 1/16, 2048ch

        self.aspp = ASPP(in_ch=2048, out_ch=256, rates=[6, 12, 18])

        # Low-level projection: layer1 (1/4, 256ch) → 48ch
        self.low_level_proj = nn.Sequential(
            conv1x1(256, 48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder: ASPP(256) upsample + low(48) → 256ch
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def backbone_params(self):
        return (list(self.stem.parameters()) + list(self.layer1.parameters()) +
                list(self.layer2.parameters()) + list(self.layer3.parameters()) +
                list(self.layer4.parameters()))

    def decoder_params(self):
        return (list(self.aspp.parameters()) + list(self.low_level_proj.parameters()) +
                list(self.decoder.parameters()) + list(self.head.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x  = self.stem(x)
        l1 = self.layer1(x)    # 1/4,  256ch
        x  = self.layer2(l1)   # 1/8
        x  = self.layer3(x)    # 1/16
        x  = self.layer4(x)    # 1/16, dilated

        x = self.aspp(x)       # 1/16, 256ch

        # 1/16 → 1/4: upsample + layer1 skip
        x = F.interpolate(x, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.decoder(torch.cat([x, self.low_level_proj(l1)], dim=1))

        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


def build_model(num_classes: int = 21) -> SemanticSegmentationModel:
    return SemanticSegmentationModel(num_classes=num_classes)
