"""src/inference.py"""
import os
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors

from models.segmentation import build_model

def main():
    # 1. 설정 파일 로드
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 모델 초기화 및 최고 성능 가중치(best_model.pth) 로드
    model = build_model(num_classes=config['model']['num_classes']).to(device)
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

    # 3. 추론용 데이터 변환 (Normalize만 수행, 크기 변경은 하지 않음)
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 제출 폴더 생성 및 이미지 리스트업
    img_dir = "submit/img"
    pred_dir = "submit/pred"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"No images found in '{img_dir}'. 테스트할 이미지를 폴더에 넣어주세요.")
        return

    print(f"Starting inference on {len(img_files)} images...")
    
    with torch.no_grad():
        for img_name in img_files:
            img_path = os.path.join(img_dir, img_name)
            
            # 이미지를 읽고 Tensor로 변환
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = tv_tensors.Image(img_pil)
            img_tensor = transform(img_tensor).unsqueeze(0).to(device) # Batch 차원 추가 (1, 3, H, W)
            
            # 예측 (모델 내부의 F.interpolate가 원본 해상도로 자동 리사이즈 해줌)
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # (H, W)
            
            # PNG 이미지로 저장 (채널 값 자체가 클래스 인덱스인 Grayscale 형태)
            pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
            
            # 확장자를 항상 .png로 변경하여 저장 (평가 스크립트 대응)
            save_name = os.path.splitext(img_name)[0] + ".png"
            save_path = os.path.join(pred_dir, save_name)
            pred_pil.save(save_path)
            print(f"Saved: {save_path}")

    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
