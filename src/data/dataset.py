"""src/data/dataset.py"""

import os
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import v2
from torchvision import tv_tensors

class VOCSegmentationDataset(Dataset):
    """
    Pascal-VOC 2012 Segmentation 데이터셋 래퍼 클래스입니다.
    torchvision.transforms.v2를 사용하여 이미지와 마스크에 동일한 변환을 적용합니다.
    """
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        super().__init__()
        # validation 어노테이션은 학습에 사용 금지 규정이 있으므로 image_set 확인 주의
        self.voc_data = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=download
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.voc_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.voc_data[index]

        # transforms.v2에서 이미지와 마스크를 함께 처리하기 위해 tv_tensors로 래핑
        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # 마스크는 Long 타입의 1D 채널로 변환하여 반환 (CrossEntropyLoss 사용을 위함)
        # mask shape: (1, H, W) -> (H, W)
        mask = mask.to(torch.long).squeeze(0)
        
        return img, mask


def get_transforms(is_train: bool) -> v2.Compose:
    """
    학습 및 평가용 데이터 증강 파이프라인을 반환합니다.
    """
    if is_train:
        return v2.Compose([
            # 랜덤 리사이즈 및 크롭 (입력 해상도는 성능/FLOPs 트레이드오프에 따라 조정 가능)
            v2.RandomResize(min_size=320, max_size=640),
            # 마스크의 빈 공간(패딩)은 ignore_label인 255로 채움
            v2.RandomCrop(size=(480, 480), pad_if_needed=True, fill={tv_tensors.Mask: 255}),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            # ImageNet 정규화 수치
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return v2.Compose([
            # 검증 시 배치 단위 처리를 위해 크기를 고정 (학습과 동일한 크기)
            v2.Resize((480, 480)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloader(
    data_dir: str,
    batch_size: int,
    image_set: str = "train",
    num_workers: int = 4,
    download: bool = False
) -> DataLoader:
    """
    데이터 로더를 생성하여 반환합니다.
    """
    is_train = image_set == "train"
    
    dataset = VOCSegmentationDataset(
        root=data_dir,
        image_set=image_set,
        transform=get_transforms(is_train=is_train),
        download=download
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return loader