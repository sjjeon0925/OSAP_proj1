"""src/models/segmentation.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def _make_dilated(stage: nn.Module, dilation: int) -> nn.Module:
    """stage 내 stride=2 conv를 dilation으로 대체해 output stride 유지."""
    for module in stage.modules():
        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
            module.stride = (1, 1)
            kh, kw = module.kernel_size
            module.dilation = (dilation, dilation)
            module.padding = (dilation * (kh // 2), dilation * (kw // 2))
    return stage


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: list[int]) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            conv1x1(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            conv1x1(out_ch * (len(rates) + 2), out_ch),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = F.interpolate(self.global_pool(x), size=x.shape[-2:],
                             mode='bilinear', align_corners=False)
        feats = [pool, self.branch1(x)] + [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class SemanticSegmentationModel(nn.Module):
    """
    EfficientNet-B3 (output stride=16) + ASPP(128ch) + DeepLabV3+ Decoder

    EfficientNet-B3 stage 구성:
      stage01 : features[0-1], 1/2,  24ch
      stage2  : features[2],   1/4,  32ch  ← skip connection
      stage3  : features[3],   1/8,  48ch
      stage4  : features[4],   1/16, 96ch
      stage5  : features[5],   1/16, 136ch
      stage6  : features[6],   1/16, 232ch (stride→dilation=2)
      stage7  : features[7],   1/16, 384ch → ASPP 입력
    """
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        feats = backbone.features

        self.stage01 = nn.Sequential(feats[0], feats[1])          # 1/2,  24ch
        self.stage2  = feats[2]                                    # 1/4,  32ch
        self.stage3  = feats[3]                                    # 1/8,  48ch
        self.stage4  = feats[4]                                    # 1/16, 96ch
        self.stage5  = feats[5]                                    # 1/16, 136ch
        self.stage6  = _make_dilated(feats[6], dilation=2)        # 1/16, 232ch
        self.stage7  = feats[7]                                    # 1/16, 384ch

        self.aspp = ASPP(in_ch=384, out_ch=128, rates=[6, 12, 18])

        self.low_level_proj = nn.Sequential(
            conv1x1(32, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128 + 16, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(128, num_classes, 1)

    def backbone_params(self):
        params = []
        for s in [self.stage01, self.stage2, self.stage3,
                  self.stage4, self.stage5, self.stage6, self.stage7]:
            params += list(s.parameters())
        return params

    def decoder_params(self):
        return (list(self.aspp.parameters()) +
                list(self.low_level_proj.parameters()) +
                list(self.decoder.parameters()) +
                list(self.head.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x  = self.stage01(x)   # 1/2
        l1 = self.stage2(x)    # 1/4, 32ch ← skip
        x  = self.stage3(l1)   # 1/8
        x  = self.stage4(x)    # 1/16
        x  = self.stage5(x)    # 1/16
        x  = self.stage6(x)    # 1/16, dilated
        x  = self.stage7(x)    # 1/16, 384ch

        x = self.aspp(x)
        x = F.interpolate(x, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.decoder(torch.cat([x, self.low_level_proj(l1)], dim=1))
        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


def build_model(num_classes: int = 21) -> SemanticSegmentationModel:
    return SemanticSegmentationModel(num_classes=num_classes)
