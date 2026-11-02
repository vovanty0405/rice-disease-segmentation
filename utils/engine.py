"""
Bộ máy huấn luyện / đánh giá DÙNG CHUNG cho 3 kịch bản:
    train_rgb_only.py  -> mode = 'rgb_only'      (3 kênh R-G-B)
    train_ms_only.py   -> mode = 'ms_only'       (3 kênh G-RE-NIR)
    train_rgb_ms.py    -> mode = 'early_fusion'  (6 kênh RGB + MS)

Mỗi kịch bản được chạy với nhiều seed (mặc định 5). Mỗi lần chạy:
    1. Huấn luyện, chọn checkpoint tốt nhất theo IoU(GD) trên tập Validation
    2. Đánh giá checkpoint tốt nhất trên tập Test (mIoU, Pixel Acc, mF1, mPrecision, mRecall, ...)
    3. Đo thời gian suy luận (s/ảnh) và FPS
    4. Lưu results.json, history.csv, biểu đồ, ảnh dự đoán vào:
       <output_dir>/<arch>_<backbone>_<MODE>/seed_<seed>/
benchmark.py sẽ đọc các results.json này để lập bảng tổng hợp.
"""
import os
import csv
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from configs.config import Config
from rice_data.dataset import GDRGBMSDataset, resolve_data_dir
from models.unet_model import build_model, arch_display_name
from utils.metrics import BinarySegMetric
from utils.table import print_table, fmt_num, fmt_mean_std, force_utf8_stdout
from utils.visualizer import save_history_curves, save_prediction_samples


# ======================================================================
# Tiện ích chung
# ======================================================================
def set_seed(seed, deterministic=False):
    """Cố định seed để kết quả có thể lặp lại."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def experiment_name(arch, backbone, mode):
    return f"{arch}_{backbone}_{Config.EXPERIMENTS[mode]['suffix']}"


class DiceBCELoss(nn.Module):
    """Dice (tốt cho đối tượng nhỏ, mất cân bằng lớp) + BCE (ổn định gradient)."""

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


class FocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode="binary")

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.focal(logits, targets)


def build_criterion(name):
    name = name.lower()
    if name == "dice":
        return smp.losses.DiceLoss(mode="binary")
    if name == "dice_bce":
        return DiceBCELoss()
    if name == "focal_dice":
        return FocalDiceLoss()
    raise ValueError(f"Loss '{name}' không hỗ trợ (dice | dice_bce | focal_dice)")


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_parser(mode):
    info = Config.EXPERIMENTS[mode]
    p = argparse.ArgumentParser(
        description=f"Huấn luyện mô hình {info['label']} ({info['in_channels']} kênh) trên GD-RGBMS-DB"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=Config.SEEDS,
                   help="Danh sách seed, mỗi seed = 1 lần chạy (mặc định 5 seed)")
    p.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    p.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    p.add_argument("--img_size", type=int, default=Config.IMG_SIZE[0], help="Kích thước ảnh vuông (mặc định 512)")
    p.add_argument("--arch", default=Config.MODEL_ARCH, help="unet | unetplusplus | deeplabv3plus | fpn | ...")
    p.add_argument("--backbone", default=Config.BACKBONE, help="resnet34 | resnet50 | efficientnet-b0 | ...")
    p.add_argument("--encoder_weights", default=Config.ENCODER_WEIGHTS, help="imagenet | none")
    p.add_argument("--loss", default=Config.LOSS, choices=["dice", "dice_bce", "focal_dice"])
    p.add_argument("--no_augment", action="store_true", help="Tắt tăng cường dữ liệu")
    p.add_argument("--no_amp", action="store_true", help="Tắt mixed precision (AMP)")
    p.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    p.add_argument("--data_dir", default=Config.DATA_DIR)
    p.add_argument("--output_dir", default=Config.BENCHMARK_ROOT)
    p.add_argument("--cache_dir", default=Config.CACHE_DIR, help="'none' để tắt cache")
    p.add_argument("--overwrite", action="store_true", help="Chạy lại cả những seed đã có kết quả")
    p.add_argument("--no_save_weights", action="store_true", help="Không lưu file .pth (tiết kiệm Drive)")
    return p


# ======================================================================
# Đánh giá & đo tốc độ
# ======================================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion=None, use_amp=False):
    model.eval()
    metric = BinarySegMetric(class_names=Config.CLASS_NAMES)
    total_loss, n = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        if criterion is not None:
            total_loss += criterion(logits, masks).item() * images.size(0)
            n += images.size(0)
        metric.update(logits, masks)
    res = metric.compute()
    res["loss"] = total_loss / n if n else float("nan")
    return res


@torch.no_grad()
def measure_inference_time(model, dataset, device, warmup=5):
    """Thời gian suy luận trung bình cho 1 ảnh (batch = 1, FP32), chỉ tính phần forward."""
    model.eval()
    n = len(dataset)
    if n == 0:
        return float("nan"), float("nan")
    x0 = dataset[0][0].unsqueeze(0).to(device)
    for _ in range(warmup):
        model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for i in range(n):
        x = dataset[i][0].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_t = float(np.mean(times))
    return mean_t, (1.0 / mean_t if mean_t > 0 else float("nan"))


# ======================================================================
# In log theo epoch dạng bảng
# ======================================================================
_EPOCH_COLS = [("Epoch", 7), ("Train Loss", 10), ("Val Loss", 9), ("Val mIoU", 9),
               ("Val IoU(GD)", 11), ("Val F1(GD)", 10), ("LR", 9), ("Time", 6), ("", 6)]


def _epoch_line(left, mid, right):
    return left + mid.join("─" * (w + 2) for _, w in _EPOCH_COLS) + right


def _epoch_row(values):
    return "│" + "│".join(" " + str(v).center(w) + " " for v, (_, w) in zip(values, _EPOCH_COLS)) + "│"


# ======================================================================
# Một lần chạy (1 mode, 1 seed)
# ======================================================================
def run_single(mode, seed, args, datasets, device):
    info = Config.EXPERIMENTS[mode]
    exp_name = experiment_name(args.arch, args.backbone, mode)
    run_dir = os.path.join(args.output_dir, exp_name, f"seed_{seed}")
    result_path = os.path.join(run_dir, "results.json")

    if os.path.exists(result_path) and not args.overwrite:
        print(f"\n[BỎ QUA] {exp_name} | seed={seed} đã có kết quả: {result_path}")
        print("         (thêm --overwrite nếu muốn chạy lại)")
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    weights_dir = os.path.join(run_dir, "weights")
    vis_dir = os.path.join(run_dir, "visualizes")
    for d in (weights_dir, vis_dir):
        os.makedirs(d, exist_ok=True)

    set_seed(seed, deterministic=True)
    train_ds, val_ds, test_ds = datasets

    g = torch.Generator()
    g.manual_seed(seed)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                     worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = build_model(args.arch, args.backbone, info["in_channels"], Config.NUM_CLASSES,
                        encoder_weights=args.encoder_weights).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    criterion = build_criterion(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = _make_scaler(use_amp)

    print("\n" + "=" * 96)
    print(f" {exp_name}  |  seed = {seed}  |  {info['label']} ({info['in_channels']} kênh: {info['input']})")
    print(f" Kiến trúc: {arch_display_name(args.arch, args.backbone)}  |  Params: {params_m:.2f} M  |  "
          f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} ảnh")
    print("=" * 96)
    print(_epoch_line("┌", "┬", "┐"))
    print(_epoch_row([c for c, _ in _EPOCH_COLS]))
    print(_epoch_line("├", "┼", "┤"), flush=True)

    history = []
    best_score, best_epoch, best_state = -1.0, 0, None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            loss = criterion(logits.float(), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            n += images.size(0)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss = running / max(n, 1)

        val = evaluate(model, val_loader, device, criterion, use_amp=False)
        is_best = val["IoU_GD"] > best_score
        if is_best:
            best_score, best_epoch = val["IoU_GD"], epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        ep_time = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
            "val_mIoU": val["mIoU"], "val_IoU_GD": val["IoU_GD"], "val_F1_GD": val["F1_GD"],
            "lr": lr_now, "time_s": ep_time,
        })
        print(_epoch_row([f"{epoch}/{args.epochs}", f"{train_loss:.4f}", f"{val['loss']:.4f}",
                          f"{val['mIoU']:.2f}", f"{val['IoU_GD']:.2f}", f"{val['F1_GD']:.2f}",
                          f"{lr_now:.1e}", f"{ep_time:.0f}s", "★ best" if is_best else ""]), flush=True)

    print(_epoch_line("└", "┴", "┘"))
    train_time_min = (time.time() - t_start) / 60

    # ---- Đánh giá checkpoint tốt nhất trên tập Test ----
    model.load_state_dict(best_state)
    if not args.no_save_weights:
        torch.save(best_state, os.path.join(weights_dir, "best_model.pth"))
    test = evaluate(model, test_loader, device, None, use_amp=False)
    inf_time, fps = measure_inference_time(model, test_ds, device)

    result = {
        "exp_name": exp_name,
        "mode": mode,
        "input_label": info["label"],
        "input_bands": info["input"],
        "in_channels": info["in_channels"],
        "model_label": arch_display_name(args.arch, args.backbone),
        "arch": args.arch,
        "backbone": args.backbone,
        "encoder_weights": args.encoder_weights,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "img_size": args.img_size,
        "augment": not args.no_augment,
        "best_epoch": best_epoch,
        "best_val_IoU_GD": best_score,
        "params_M": params_m,
        "train_time_min": train_time_min,
        "inference_time_s": inf_time,
        "fps": fps,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "test": {k: v for k, v in test.items() if k != "loss"},
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    try:
        save_history_curves(history, os.path.join(vis_dir, "training_curves.png"),
                            title=f"{exp_name} | seed {seed}")
        save_prediction_samples(model, test_loader, device, os.path.join(vis_dir, "test_predictions.png"),
                                mode=mode)
    except Exception as e:  # Không để lỗi vẽ hình làm hỏng kết quả
        print(f"[CẢNH BÁO] Không vẽ được hình: {e}")

    t = result["test"]
    print_table(
        ["Chỉ số (Test)", "mIoU", "Pixel Acc", "mF1", "mPrecision", "mRecall",
         "IoU(GD)", "F1(GD)", "s/ảnh", "FPS"],
        [[f"seed {seed} | best ep {best_epoch}", fmt_num(t["mIoU"]), fmt_num(t["PixelAcc"]), fmt_num(t["mF1"]),
          fmt_num(t["mPrecision"]), fmt_num(t["mRecall"]), fmt_num(t["IoU_GD"]), fmt_num(t["F1_GD"]),
          fmt_num(inf_time, 4), fmt_num(fps, 1)]],
        title=f"KẾT QUẢ TEST - {exp_name} (seed {seed}) - huấn luyện {train_time_min:.1f} phút",
    )
    print(f"--> Đã lưu kết quả tại: {run_dir}")

    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_seed_summary(results, mode):
    info = Config.EXPERIMENTS[mode]
    keys = ["mIoU", "PixelAcc", "mF1", "mPrecision", "mRecall", "IoU_GD", "F1_GD"]
    rows = []
    for r in sorted(results, key=lambda x: x["seed"]):
        t = r["test"]
        rows.append([r["seed"], r["best_epoch"]] + [fmt_num(t[k]) for k in keys]
                    + [fmt_num(r["inference_time_s"], 4)])
    footer = [["Mean ± Std", "—"] + [fmt_mean_std([r["test"][k] for r in results]) for k in keys]
              + [fmt_mean_std([r["inference_time_s"] for r in results], 4)]]
    print_table(
        ["Seed", "Best ep", "mIoU (%)", "Pixel Acc (%)", "mF1 (%)", "mPrecision (%)", "mRecall (%)",
         "IoU GD (%)", "F1 GD (%)", "s/ảnh"],
        rows, footer=footer,
        title=f"TỔNG HỢP {len(results)} LẦN CHẠY - {info['label']} ({info['in_channels']} kênh)",
        note="Chạy 'python benchmark.py' để so sánh các mô hình và kiểm định thống kê.",
    )


# ======================================================================
# Hàm chính cho mỗi file train_*.py
# ======================================================================
def main_for_mode(mode):
    force_utf8_stdout()
    args = build_parser(mode).parse_args()
    info = Config.EXPERIMENTS[mode]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data_dir = resolve_data_dir(args.data_dir)
    cache_dir = None if str(args.cache_dir).lower() == "none" else args.cache_dir
    if not os.path.exists(os.path.join(args.data_dir, "train_list.txt")):
        raise SystemExit(f"[LỖI] Không tìm thấy dataset tại '{args.data_dir}'. "
                         f"Hãy giải nén GD-RGBMS-DB.zip hoặc dùng --data_dir.")

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (sẽ rất chậm!)"
    print("#" * 96)
    print(f"  THỰC NGHIỆM : {info['label']}  ({info['in_channels']} kênh: {info['input']})")
    print(f"  Mô hình     : {arch_display_name(args.arch, args.backbone)}  | encoder weights: {args.encoder_weights}")
    print(f"  Seeds       : {args.seeds}  | Epochs: {args.epochs}  | Batch: {args.batch_size}  | LR: {args.lr}"
          f"  | Loss: {args.loss}")
    print(f"  Thiết bị    : {gpu}")
    print(f"  Kết quả lưu : {os.path.join(args.output_dir, experiment_name(args.arch, args.backbone, mode))}")
    print("#" * 96)

    img_size = (args.img_size, args.img_size)
    print("Đang nạp dữ liệu (lần đầu sẽ tạo cache ảnh đã resize, mất vài phút)...", flush=True)
    t0 = time.time()
    train_ds = GDRGBMSDataset(args.data_dir, "train_list.txt", mode, img_size,
                              augment=not args.no_augment, cache_dir=cache_dir, preload=True, verbose=True)
    val_ds = GDRGBMSDataset(args.data_dir, "val_list.txt", mode, img_size,
                            augment=False, cache_dir=cache_dir, preload=True, verbose=True)
    test_ds = GDRGBMSDataset(args.data_dir, "test_list.txt", mode, img_size,
                             augment=False, cache_dir=cache_dir, preload=True, verbose=True)
    print(f"Nạp dữ liệu xong sau {time.time() - t0:.0f}s.")

    results = []
    for seed in args.seeds:
        results.append(run_single(mode, seed, args, (train_ds, val_ds, test_ds), device))

    print_seed_summary(results, mode)
