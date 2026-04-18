"""src/eval.py"""
import os
import yaml
import torch
from models.segmentation import build_model
# fvcore 또는 ptflops 사용 권장 (수업 시간에 배운 도구 활용)
from fvcore.nn import FlopCountAnalysis

def evaluate_efficiency(model):
    # 과제 요구 입력 크기: 3 x 480 x 640
    # 모델과 동일한 디바이스로 텐서 이동
    device = next(model.parameters()).device
    input_tensor = torch.randn(1, 3, 480, 640).to(device)
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return flops.total()

def main():
    # 1. 설정 파일 로드
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=config['model']['num_classes']).to(device)

    # 2. 최고 성능 가중치(best_model.pth) 로드
    ckpt_path = os.path.join(config['train']['checkpoint_dir'], 'best_model.pth')

    if os.path.isfile(ckpt_path):
        print(f"=> Loading best model from '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Error: Checkpoint not found at '{ckpt_path}'")
        print("학습을 먼저 완료하여 best_model.pth를 생성해주세요.")
        return

    model.eval()

    print("--- Efficiency Evaluation ---")
    evaluate_efficiency(model)

if __name__ == "__main__":
    main()
