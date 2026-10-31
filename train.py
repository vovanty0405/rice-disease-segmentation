import os
import sys
import argparse

# Đảm bảo Python luôn nhận diện được thư mục hiện tại làm module root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import pandas as pd
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from configs.config import Config
from datasets.dataset import GDRGBMSDataset
from models.unet_model import build_model
from utils.visualizer import save_loss_curve, save_prediction_samples

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân vùng hạt lem lép lúa (GD-RGBMS-DB)")
    parser.add_argument("--mode", type=str, default=Config.MODE, choices=["early_fusion", "rgb_only", "ms_only"],
                        help="Phương thức dữ liệu: early_fusion (6 kênh), rgb_only (3 kênh), ms_only (3 kênh)")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    # 1. Cấu hình thư mục lưu trữ thực nghiệm
    experiment_name = f"UNet_{args.mode.upper()}"
    exp_dir = os.path.join(Config.DRIVE_EXPERIMENT_ROOT, experiment_name)
    weights_dir = os.path.join(exp_dir, "weights")
    logs_dir = os.path.join(exp_dir, "logs")
    vis_dir = os.path.join(exp_dir, "visualizes")

    for d in [weights_dir, logs_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"=== BẮT ĐẦU HUẤN LUYỆN: {experiment_name} ===")
    print(f"Lưu kết quả tại: {exp_dir}")

    # 2. Chuẩn bị DataLoader
    in_channels = 6 if args.mode == "early_fusion" else 3
    train_dataset = GDRGBMSDataset(Config.DATA_DIR, "train_list.txt", mode=args.mode, img_size=Config.IMG_SIZE)
    val_dataset = GDRGBMSDataset(Config.DATA_DIR, "val_list.txt", mode=args.mode, img_size=Config.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=Config.NUM_WORKERS)

    # 3. Khởi tạo Model, Loss, Optimizer
    device = torch.device(Config.DEVICE)
    model = build_model(arch=Config.MODEL_ARCH, backbone=Config.BACKBONE, in_channels=in_channels, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)

    # 4. Training Loop
    history_losses = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history_losses.append(avg_loss)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Dice Loss: {avg_loss:.4f}")

        # Lưu model có loss tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))

    # Lưu model ở epoch cuối cùng
    torch.save(model.state_dict(), os.path.join(weights_dir, "last_model.pth"))
    print("--> Đã lưu thành công các file weights (.pth)")

    # 5. Lưu bảng nhật ký chỉ số (metrics csv)
    df_history = pd.DataFrame({"Epoch": list(range(1, args.epochs + 1)), "Dice_Loss": history_losses})
    df_history.to_csv(os.path.join(logs_dir, "training_history.csv"), index=False)
    print("--> Đã lưu lịch sử huấn luyện vào file CSV")

    # 6. Vẽ và lưu biểu đồ Loss & Ảnh mẫu AI dự đoán
    save_loss_curve(history_losses, os.path.join(vis_dir, "loss_curve.png"), title=f"Quá trình học - {experiment_name}")
    save_prediction_samples(model, val_loader, device, os.path.join(vis_dir, "sample_predictions.png"))
    print("--> Đã lưu biểu đồ Loss và ảnh dự đoán trực quan thành công!")
    print(f"=== HOÀN TẤT THỰC NGHIỆM {experiment_name} ===")

if __name__ == "__main__":
    main()
