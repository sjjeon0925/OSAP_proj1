"""src/inference.py"""
import os
import yaml
import zipfile
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors

from models.segmentation import build_model


def load_model(config, device):
    model = build_model(num_classes=config['model']['num_classes']).to(device)
    ckpt_path = os.path.join(config['train']['checkpoint_dir'], 'best_model.pth')

    if not os.path.isfile(ckpt_path):
        print(f"Error: Checkpoint not found at '{ckpt_path}'")
        print("학습을 먼저 완료하여 best_model.pth를 생성해주세요.")
        return None

    print(f"=> Loading best model from '{ckpt_path}'")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def prepare_images(img_zip: str, img_dir: str) -> None:
    """Drive의 zip을 img_dir에 압축 해제. 이미 이미지가 있으면 건너뜀."""
    os.makedirs(img_dir, exist_ok=True)
    existing = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if existing:
        print(f"Images already exist in '{img_dir}' ({len(existing)} files). Skipping extraction.")
        return

    if not os.path.isfile(img_zip):
        print(f"Error: zip file not found at '{img_zip}'")
        return

    print(f"Extracting '{img_zip}' → '{img_dir}' ...")
    with zipfile.ZipFile(img_zip, 'r') as zf:
        zf.extractall(img_dir)
    print(f"Extraction complete. ({len(os.listdir(img_dir))} files)")


def run_inference(model, device, img_dir, pred_dir):
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(pred_dir, exist_ok=True)

    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    if not img_files:
        print(f"No images found in '{img_dir}'. 테스트할 이미지를 폴더에 넣어주세요.")
        return

    print(f"Starting inference on {len(img_files)} images...")
    with torch.no_grad():
        for img_name in img_files:
            img_pil = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
            img_tensor = transform(tv_tensors.Image(img_pil)).unsqueeze(0).to(device)

            pred_mask = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy()

            save_name = os.path.splitext(img_name)[0] + ".png"
            Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(pred_dir, save_name))
            print(f"Saved: {os.path.join(pred_dir, save_name)}")

    print("Inference completed successfully!")


def export_onnx(model, save_path="model.onnx"):
    try:
        import onnx
    except ImportError:
        print("onnx 패키지가 없습니다. `pip install onnx` 후 재실행하세요.")
        return

    struct_path = os.path.splitext(save_path)[0] + "_structure.onnx"

    dummy = torch.randn(1, 3, 480, 640, device=next(model.parameters()).device)
    torch.onnx.export(
        model, dummy, save_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
    print(f"ONNX model exported: {save_path}")

    # 가중치 제거 (제출 사이트 요구: 구조만 포함, 10MB 이하)
    m = onnx.load(save_path)
    for init in m.graph.initializer:
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.ClearField("int32_data")
        init.ClearField("int64_data")
    onnx.save(m, struct_path)
    print(f"Structure-only ONNX saved: {struct_path}")


def main():
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    if model is None:
        return

    prepare_images(img_zip=config['submit']['img_zip'], img_dir=config['submit']['img_dir'])
    run_inference(model, device, img_dir=config['submit']['img_dir'], pred_dir=config['submit']['pred_dir'])
    export_onnx(model, save_path=config['submit']['onnx_path'])


if __name__ == "__main__":
    main()
