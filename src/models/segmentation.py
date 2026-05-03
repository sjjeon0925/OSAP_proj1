"""src/models/segmentation.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def _make_layer_dilated(layer: nn.Module, dilation: int) -> nn.Module:
    """ResNet layer의 stride를 제거하고 dilation을 적용 (output stride 유지)."""
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
    """Atrous Spatial Pyramid Pooling (DeepLabV3 style, standard dilated conv)."""
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
    DeepLabV3+ style: ResNet50 (output stride=16) + ASPP + Decoder
    - layer4를 dilated conv로 변환해 output stride=16 유지 (v3 대비 2배 해상도)
    - layer1 저해상도 feature를 decoder에서 재활용해 경계 복원력 향상
    """
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Stem: 1/4 resolution
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 1/4,  256ch  ← low-level features
        self.layer2 = backbone.layer2  # 1/8,  512ch
        self.layer3 = backbone.layer3  # 1/16, 1024ch
        # layer4: stride 제거 + dilation=2 → output stride=16 유지, 2048ch
        self.layer4 = _make_layer_dilated(backbone.layer4, dilation=2)

        # ASPP (1/16 해상도에서 적용)
        self.aspp = ASPP(in_ch=2048, out_ch=256, rates=[6, 12, 18])

        # Low-level feature projection: layer1(256ch) → 48ch
        self.low_level_proj = nn.Sequential(
            conv1x1(256, 48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder: ASPP(256) + low-level(48) → 256ch → num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self.stem(x)
        low_level = self.layer1(x)   # 1/4,  256ch
        x = self.layer2(low_level)   # 1/8
        x = self.layer3(x)           # 1/16
        x = self.layer4(x)           # 1/16, dilated

        x = self.aspp(x)             # 1/16, 256ch

        # ASPP output → 1/4 해상도로 upsample
        x = F.interpolate(x, size=low_level.shape[-2:], mode='bilinear', align_corners=False)

        # Low-level feature와 concat → decode
        x = torch.cat([x, self.low_level_proj(low_level)], dim=1)
        x = self.decoder(x)
        x = self.head(x)

        # 원본 해상도로 upsample
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


def build_model(num_classes: int = 21) -> SemanticSegmentationModel:
    return SemanticSegmentationModel(num_classes=num_classes)
