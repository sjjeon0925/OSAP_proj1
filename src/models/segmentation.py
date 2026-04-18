"""src/models/segmentation.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

def conv3x3(in_ch: int, out_ch: int, dilation: int) -> nn.Module:
    if dilation > 1:
        # 연산량 최적화(FLOPs 감소)를 위한 Depthwise Separable Convolution
        return nn.Sequential(
            # 1. Depthwise Conv: 채널별로 따로 연산 (groups=in_ch)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation, groups=in_ch, bias=False),
            # 2. Pointwise Conv: 1x1 Conv로 채널 수 축소 (in_ch -> out_ch)
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
    else:
        # 일반 Convolution
        return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    다양한 Receptive Field를 통해 다중 스케일 문맥 정보를 추출합니다.
    """
    def __init__(self, in_ch: int, out_ch: int, rates: list[int]) -> None:
        super().__init__()

        # 1x1 Conv branch
        self.branch1 = nn.Sequential(
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Dilated Conv branches
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(
                nn.Sequential(
                    conv3x3(in_ch, out_ch, dilation=rate),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        # Global Avg Pool branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Projection (1x1 + Dilated * len(rates) + Global Pool)
        self.project = nn.Sequential(
            conv1x1(out_ch * (len(rates) + 2), out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) # 과적합 방지
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global pooling & upsample
        pool_feat = self.global_pool(x)
        pool_feat = F.interpolate(pool_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # 1x1 and Dilated Convs
        branch1_feat = self.branch1(x)
        branch_feats = [branch(x) for branch in self.branches]
        
        # Concat all features
        concat_feat = torch.cat([pool_feat, branch1_feat] + branch_feats, dim=1)
        
        # Projection
        out = self.project(concat_feat)
        return out

class SemanticSegmentationModel(nn.Module):
    """
    Project 1: MobileNetV2 Backbone + ASPP Neck 구조
    FLOPs 점수를 높게 받기 위해 연산량이 적은 MobileNetV2를 사용합니다.
    """
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()

        # 1. Backbone: TorchVision pretrained ImageNet-1K weights (Segmentation weights 금지 규정 준수)
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        mobilenet = mobilenet_v2(weights=weights)
        
        # Classifier 제외하고 Feature Extractor만 사용
        self.backbone = mobilenet.features
        
        # MobileNetV2의 최종 feature channel은 1280
        backbone_out_channels = 1280

        # 2. Neck: ASPP
        aspp_out_channels = 256
        # 모바일넷에 맞게 rate를 조금 줄여 FLOPs 최적화 (필요에 따라 6, 12, 18 등 조정 가능)
        self.neck = ASPP(in_ch=backbone_out_channels, out_ch=aspp_out_channels, rates=[3, 6, 9])

        # 3. Head: Pixel-wise Classification
        self.head = nn.Conv2d(aspp_out_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:] # 원본 이미지 (H, W) 저장

        # Feature Extraction
        feat = self.backbone(x)
        
        # Context Aggregation
        feat = self.neck(feat)
        
        # Prediction
        out = self.head(feat)
        
        # Upsample to original input resolution
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out

def build_model(num_classes: int = 21) -> SemanticSegmentationModel:
    return SemanticSegmentationModel(num_classes=num_classes)