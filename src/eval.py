"""src/eval.py"""
import os
import yaml
import torch
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from models.segmentation import build_model
from data.dataset import get_dataloader
from utils.metrics import SegmentationMetric


def evaluate_efficiency(model):
    device = next(model.parameters()).device
    input_tensor = torch.randn(1, 3, 480, 640).to(device)
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return flops.total()


def evaluate_miou(model, val_loader, device, num_classes):
    model.eval()
    metric = SegmentationMetric(n_classes=num_classes)
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating mIoU"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            metric.add_batch(preds.cpu().numpy(), masks.cpu().numpy())
    return metric.mean_iou()


def main():
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=config['model']['num_classes']).to(device)

    ckpt_path = os.path.join(config['train']['checkpoint_dir'], 'best_model.pth')
    if os.path.isfile(ckpt_path):
        print(f"=> Loading best model from '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   (saved at epoch {checkpoint['epoch'] + 1}, best mIoU {checkpoint['best_miou']:.4f})")
    else:
        print(f"Error: Checkpoint not found at '{ckpt_path}'")
        return

    model.eval()

    print("\n--- Performance Evaluation ---")
    val_loader = get_dataloader(
        config['data']['root'],
        config['data']['batch_size'],
        image_set="val",
        num_workers=config['data']['num_workers'],
        download=True,
    )
    miou = evaluate_miou(model, val_loader, device, config['model']['num_classes'])
    print(f"Val mIoU: {miou:.4f}")

    print("\n--- Efficiency Evaluation ---")
    evaluate_efficiency(model)


if __name__ == "__main__":
    main()
