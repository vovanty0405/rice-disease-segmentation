import matplotlib
matplotlib.use("Agg")  # Vẽ không cần màn hình (chạy bằng !python trên Colab)
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_loss_curve(history_losses, save_path, title="Training Loss Curve"):
    """(Hàm cũ) Vẽ và lưu biểu đồ Loss qua các Epochs"""
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(history_losses) + 1))
    plt.plot(epochs, history_losses, marker="o", color="royalblue", linewidth=2, label="Dice Loss")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_history_curves(history, save_path, title="Training history"):
    """Vẽ 2 biểu đồ: (1) Train/Val loss, (2) Val mIoU, IoU(GD), F1(GD) theo epoch."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="Train loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "s-", label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, [h["val_mIoU"] for h in history], "o-", label="Val mIoU")
    axes[1].plot(epochs, [h["val_IoU_GD"] for h in history], "s-", label="Val IoU (GD)")
    axes[1].plot(epochs, [h["val_F1_GD"] for h in history], "^-", label="Val F1 (GD)")
    axes[1].set_title("Chỉ số trên tập Validation (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


_INPUT_TITLES = {
    "rgb_only": "Ảnh RGB",
    "ms_only": "Ảnh MS (G-RE-NIR, giả màu)",
    "early_fusion": "Ảnh RGB (3/6 kênh đầu vào)",
}


def save_prediction_samples(model, loader, device, save_path, num_samples=3, mode="early_fusion"):
    """
    Lưu ảnh so sánh 4 cột: Ảnh đầu vào | Ground-truth | AI dự đoán | Bản đồ sai số
    Bản đồ sai số: xanh lá = TP (đúng), đỏ = FP (báo nhầm), vàng = FN (bỏ sót)
    """
    model.eval()
    images, masks = next(iter(loader))

    with torch.no_grad():
        logits = model(images.to(device))
        preds = (torch.sigmoid(logits) > 0.5).float()

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    input_title = _INPUT_TITLES.get(mode, "Ảnh đầu vào")
    for i in range(num_samples):
        show = np.clip(np.transpose(images[i, :3, :, :], (1, 2, 0)), 0, 1)
        gt = masks[i, 0] > 0.5
        pr = preds[i, 0] > 0.5

        err = np.zeros(gt.shape + (3,), dtype=np.float32)
        err[gt & pr] = (0.0, 0.8, 0.0)     # TP
        err[~gt & pr] = (0.9, 0.1, 0.1)    # FP
        err[gt & ~pr] = (1.0, 0.85, 0.0)   # FN

        axes[i, 0].imshow(show)
        axes[i, 0].set_title(f"Mẫu {i + 1}: {input_title}")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title(f"Mẫu {i + 1}: Nhãn thực tế (Ground-Truth)")
        axes[i, 2].imshow(pr, cmap="gray")
        axes[i, 2].set_title(f"Mẫu {i + 1}: AI dự đoán")
        axes[i, 3].imshow(err)
        axes[i, 3].set_title("Sai số: TP xanh | FP đỏ | FN vàng")
        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
