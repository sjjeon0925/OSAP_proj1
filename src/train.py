"""src/train.py"""
import os
import yaml
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from models.segmentation import build_model
from data.dataset import get_dataloader
from utils.metrics import SegmentationMetric

def save_checkpoint(state, is_best, save_dir):
    """체크포인트를 저장하는 헬퍼 함수"""
    os.makedirs(save_dir, exist_ok=True)

    # 항상 최신 상태 저장 (이어하기 용도)
    latest_path = os.path.join(save_dir, 'latest_model.pth')
    torch.save(state, latest_path)

    # 최고 성능일 경우 따로 저장 (평가 및 제출 용도)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)

def train():
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config['wandb']['use']:
        wandb.init(project=config['project_name'], name=config['wandb']['name'], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Colab 서버에 데이터가 없을 경우를 대비해 항상 다운로드를 확인하도록 수정
    train_loader = get_dataloader(config['data']['root'], config['data']['batch_size'], "train",
                                  num_workers=config['data']['num_workers'], download=True)
    val_loader = get_dataloader(config['data']['root'], config['data']['batch_size'], "val",
                                num_workers=config['data']['num_workers'], download=True)
    model = build_model(num_classes=config['model']['num_classes']).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['train']['epochs'], eta_min=config['train']['lr'] * 0.01
    )

    start_epoch = 0
    best_miou = 0.0

    # 학습 재실행(Resume) 로직
    if config['train'].get('resume', False):
        latest_ckpt_path = os.path.join(config['train']['checkpoint_dir'], 'latest_model.pth')
        if os.path.isfile(latest_ckpt_path):
            print(f"=> Loading checkpoint '{latest_ckpt_path}'")
            checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch'] + 1
            best_miou = checkpoint['best_miou']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{latest_ckpt_path}', starting from scratch.")

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        train_loss = 0.0

        # 스텝(Batch) 단위 손실 기록 및 Tqdm 진행바 개선
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        for step, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 진행바에 현재 Loss 표시
            pbar.set_postfix({'loss': loss.item()})

            # WandB에 step 단위 기록 (선택 사항: 에포크 단위만 필요하다면 삭제 가능)
            if config['wandb']['use']:
                wandb.log({"step_loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        miou = validate(model, val_loader, device, config['model']['num_classes'])
        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Val mIoU: {miou:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if config['wandb']['use']:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_train_loss": avg_train_loss,
                "val_miou": miou,
                "lr": current_lr,
            })

        # 옵티마이저 상태와 에포크를 포함하여 체크포인트 저장
        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
        }

        save_checkpoint(checkpoint_state, is_best, config['train']['checkpoint_dir'])

def validate(model, loader, device, num_classes):
    model.eval()
    metric = SegmentationMetric(n_classes=num_classes)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            metric.add_batch(preds.cpu().numpy(), masks.cpu().numpy())

    return metric.mean_iou()

if __name__ == "__main__":
    train()
