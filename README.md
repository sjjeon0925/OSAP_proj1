# Project 1: Semantic Segmentation

EfficientNet-B0 backbone + DeepLabV3+ decoder를 이용한 Pascal VOC 21-class 시맨틱 세그멘테이션.

## 실행 환경

- Google Colab (A100 GPU)
- Python 3.11+

## 실행 방법

- VSCode의 Colab 확장은 Assigned Colab Server에서 자주 끊김이 발생
- 코드 작성은 VSCode에서 작성 후 깃허브 push
- Colab 웹에서 노트북 파일 실행, 노트북 셀에서 git clone하여 모델 파일 로드 후 사용

`notebooks/main.ipynb`를 Google Colab에서 열고 셀을 순서대로 실행한다.

**Cell 0 - 레포지토리 클론 및 의존성 설치:**
```bash
!git clone https://github.com/sjjeon0925/OSAP_proj1 proj1
%cd proj1
!pip install fvcore pycocotools wandb tqdm
```

**학습:**
노트북의 학습 셀을 실행한다. `src/config/config.yaml`의 설정이 적용되며,  
`train.resume: true`로 설정 시 `latest_model.pth`에서 이어서 학습한다.

**평가 (mIoU + FLOPs):**
노트북의 평가 셀을 실행한다 (`src/eval.py` 호출).  
`best_model.pth`를 로드하여 Pascal VOC 2012 val mIoU와 FLOPs를 출력한다.

**추론:**
노트북의 추론 셀을 실행한다 (`src/inference.py` 호출).  
`submit/img/`의 이미지를 읽어 `submit/pred/`에 예측 결과를 저장한다 (예: `0001.jpg` → `0001.png`).
- `submit/img/` 폴더에 이미지 저장 필요

## FLOPs 측정 방법

`src/eval.py`에서 `fvcore.nn.FlopCountAnalysis`를 사용하여 `1×3×480×640` 입력 기준으로 측정한다:

```python
from fvcore.nn import FlopCountAnalysis
input_tensor = torch.randn(1, 3, 480, 640)
flops = FlopCountAnalysis(model, input_tensor)
print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
```

노트북의 평가 셀 실행 시 자동으로 측정된다.

## 제출 결과 재현

1. Google Drive에 저장된 `best_model.pth`를 `config.yaml`의 `train.checkpoint_dir` 경로에 준비한다.
2. 테스트 이미지(`.jpg`)를 `submit/img/`에 넣는다.
3. `config.yaml`의 `submit.pred_dir`를 `submit/pred`로 설정한다.
4. 노트북의 추론 셀을 실행한다.

입력 파일명과 동일한 이름으로 `.png` 예측 결과가 `submit/pred/`에 저장된다.

## 최종 결과

| 지표 | Val (VOC 2012) | Test (Closed) |
|------|---------------|---------------|
| mIoU | 0.7493 | 0.688 |
| FLOPs | 6.62 GFLOPs (로컬) | 13.118 GFLOPs (채점 사이트) |

- Backbone: EfficientNet-B0 (ImageNet-1K pretrained)
- 학습 데이터: Pascal VOC 2012 train (1,464장) + MS-COCO 2017 필터링 (60,000장)
- 학습 에포크: 100 (best checkpoint: epoch 71)
