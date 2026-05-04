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
    ResNet101 (output stride=8) + ASPP + Multi-scale Decoder

    output stride=8: layer3(dilation=2) + layer4(dilation=4) 로 고해상도 feature 유지.
    Multi-scale decoder: layer2(1/8, 512ch)를 ASPP와 동해상도에서 먼저 합치고,
                         layer1(1/4, 256ch)과 한 번 더 합쳐 경계 복원력을 높임.
    """
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()

        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        # Stem: 1/4 resolution
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # 1/4,  256ch  ← low-level (fine)
        self.layer2 = backbone.layer2   # 1/8,  512ch  ← low-level (mid)
        # layer3: stride 제거 + dilation=2  → 1/8 해상도 유지, 1024ch
        self.layer3 = _make_layer_dilated(backbone.layer3, dilation=2)
        # layer4: stride 제거 + dilation=4  → 1/8 해상도 유지, 2048ch
        self.layer4 = _make_layer_dilated(backbone.layer4, dilation=4)

        # ASPP: output stride=8이므로 rate를 2배 스케일 적용
        self.aspp = ASPP(in_ch=2048, out_ch=256, rates=[12, 24, 36])

        # Mid-level projection: layer2 (1/8, 512ch) → 64ch
        self.mid_level_proj = nn.Sequential(
            conv1x1(512, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ASPP(256) + mid(64) → 1/8 fusion
        self.fuse_mid = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Low-level projection: layer1 (1/4, 256ch) → 48ch
        self.low_level_proj = nn.Sequential(
            conv1x1(256, 48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # fuse_mid(256) upsample + low(48) → 1/4 fusion → decode
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
        l1 = self.layer1(x)    # 1/4,  256ch
        l2 = self.layer2(l1)   # 1/8,  512ch
        x  = self.layer3(l2)   # 1/8,  1024ch (dilated)
        x  = self.layer4(x)    # 1/8,  2048ch (dilated)

        x = self.aspp(x)       # 1/8,  256ch

        # 1/8: ASPP + mid-level layer2 feature 합산
        x = self.fuse_mid(torch.cat([x, self.mid_level_proj(l2)], dim=1))

        # 1/4로 upsample 후 low-level layer1 feature 합산
        x = F.interpolate(x, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, self.low_level_proj(l1)], dim=1)
        x = self.decoder(x)

        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


def build_model(num_classes: int = 21) -> SemanticSegmentationModel:
    return SemanticSegmentationModel(num_classes=num_classes)
