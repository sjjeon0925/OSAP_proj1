"""src/data/dataset.py"""

import os
import numpy as np
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import VOCSegmentation, CocoDetection
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image

# COCO category_id → VOC class index 매핑 (배경=0 제외, 20개 클래스)
COCO_TO_VOC = {
    1:  15,  # person
    2:  2,   # bicycle
    3:  7,   # car
    4:  14,  # motorcycle  → motorbike
    5:  1,   # airplane    → aeroplane
    6:  6,   # bus
    7:  19,  # train
    9:  4,   # boat
    16: 3,   # bird
    17: 8,   # cat
    18: 12,  # dog
    19: 13,  # horse
    20: 17,  # sheep
    21: 10,  # cow
    44: 5,   # bottle
    62: 9,   # chair
    63: 18,  # couch       → sofa
    64: 16,  # potted plant → pottedplant
    67: 11,  # dining table → diningtable
    72: 20,  # tv           → tvmonitor
}


class VOCSegmentationDataset(Dataset):
    """Pascal-VOC 2012 Segmentation 래퍼."""
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        super().__init__()
        self.voc_data = VOCSegmentation(root=root, year="2012",
                                        image_set=image_set, download=download)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.voc_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.voc_data[index]
        img  = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask.to(torch.long).squeeze(0)


class COCOVOCSegmentation(CocoDetection):
    """
    torchvision.datasets.CocoDetection 기반.
    MS-COCO 2017 train → VOC 21-class 시맨틱 마스크 변환.
    - COCO_TO_VOC 매핑에 해당하는 카테고리만 반영 (나머지는 배경=0).
    - iscrowd=1 어노테이션 무시.
    - VOC 클래스 어노테이션이 없는 이미지 제외.
    """
    def __init__(self, coco_root: str, transform: Optional[Callable] = None) -> None:
        img_dir  = os.path.join(coco_root, 'train2017')
        ann_file = os.path.join(coco_root, 'annotations', 'instances_train2017.json')
        super().__init__(root=img_dir, annFile=ann_file, transforms=None)
        self._seg_transform = transform

        # VOC 카테고리가 포함된 이미지 ID만 수집
        voc_cat_ids = list(COCO_TO_VOC.keys())
        img_ids: set = set()
        for cat_id in voc_cat_ids:
            img_ids.update(self.coco.getImgIds(catIds=[cat_id]))
        self.ids = sorted(img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from pycocotools import mask as coco_mask

        img_id   = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        H, W     = img_info['height'], img_info['width']

        img        = self._load_image(img_id)  # PIL Image
        mask_array = np.zeros((H, W), dtype=np.uint8)

        for ann in self._load_target(img_id):
            if ann.get('iscrowd', 0):
                continue
            cat_id = ann['category_id']
            if cat_id not in COCO_TO_VOC:
                continue
            seg = ann['segmentation']
            if isinstance(seg, list):
                rle    = coco_mask.frPyObjects(seg, H, W)
                binary = coco_mask.decode(coco_mask.merge(rle))
            else:
                binary = coco_mask.decode(seg)
            mask_array[binary > 0] = COCO_TO_VOC[cat_id]

        img  = tv_tensors.Image(img)
        mask = tv_tensors.Mask(Image.fromarray(mask_array))

        if self._seg_transform is not None:
            img, mask = self._seg_transform(img, mask)

        return img, mask.to(torch.long).squeeze(0)


def get_transforms(is_train: bool) -> v2.Compose:
    if is_train:
        return v2.Compose([
            v2.RandomResize(min_size=320, max_size=640),
            v2.RandomCrop(size=(480, 480), pad_if_needed=True, fill={tv_tensors.Mask: 255}),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return v2.Compose([
            v2.Resize((480, 480)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloader(data_config: dict, image_set: str = "train",
                   download: bool = False) -> DataLoader:
    """
    data_config: config['data'] 딕셔너리
    - 학습 시 use_coco=true이면 VOC + COCO 합산 데이터셋 사용
    - 검증은 항상 VOC val만 사용
    """
    is_train   = image_set == "train"
    transform  = get_transforms(is_train)

    voc_dataset = VOCSegmentationDataset(
        root=data_config['root'],
        image_set=image_set,
        transform=transform,
        download=download,
    )

    if is_train and data_config.get('use_coco', False):
        coco_root = data_config['coco_root']
        if not os.path.isdir(os.path.join(coco_root, 'train2017')):
            raise FileNotFoundError(
                f"COCO train2017 not found at '{coco_root}/train2017'.\n"
                f"MS-COCO 2017 train 데이터를 해당 경로에 준비해주세요.\n"
                f"  이미지: {coco_root}/train2017/\n"
                f"  어노테이션: {coco_root}/annotations/instances_train2017.json"
            )
        coco_dataset = COCOVOCSegmentation(coco_root=coco_root, transform=transform)
        dataset = ConcatDataset([voc_dataset, coco_dataset])
        print(f"Dataset: VOC({len(voc_dataset)}) + COCO({len(coco_dataset)}) = {len(dataset)} samples")
    else:
        dataset = voc_dataset

    return DataLoader(
        dataset,
        batch_size=data_config['batch_size'],
        shuffle=is_train,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        drop_last=is_train,
    )
