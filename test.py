import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
from torch.utils.data import DataLoader

from configs.config import Config
from datasets.dataset import GDRGBMSDataset
from models.unet_model import build_model
from utils.metrics import compute_iou_and_dice

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập Test (GD-RGBMS-DB)")
    parser.add_argument("--mode", type=str, default="early_fusion", choices=["early_fusion", "rgb_only", "ms_only"])
    parser.add_argument("--weight_path", type=str, required=True, help="Đường dẫn đến file .pth cần đánh giá")
    args = parser.parse_args()

    device = torch.device(Config.DEVICE)
    in_channels = 6 if args.mode == "early_fusion" else 3

    print(f"=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ({args.mode.upper()}) ===")
    print(f"Trọng số: {args.weight_path}")

    # 1. Nạp tập Test
    test_dataset = GDRGBMSDataset(Config.DATA_DIR, "test_list.txt", mode=args.mode, img_size=Config.IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Khởi tạo mô hình và load weights
    model = build_model(arch=Config.MODEL_ARCH, backbone=Config.BACKBONE, in_channels=in_channels, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Tính điểm IoU và Dice Score trên toàn tập test
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)

            iou, dice = compute_iou_and_dice(preds, masks)
            total_iou += iou
            total_dice += dice

    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)

    print("\n---------------- KẾT QUẢ ĐÁNH GIÁ ----------------")
    print(f"Số lượng mẫu kiểm thử (Test batches): {len(test_loader)}")
    print(f"Mean IoU (Jaccard Score)            : {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"Dice Score (F1-Score)               : {avg_dice:.4f} ({avg_dice*100:.2f}%)")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
