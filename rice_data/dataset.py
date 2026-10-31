import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Màu vùng hạt lem lép (GD) trong ảnh nhãn thư mục Mark/ (RGB)
GD_COLOR_RGB = (240, 128, 128)
# Nền có màu (173, 216, 230). Cho phép sai lệch nhỏ để an toàn.
COLOR_TOLERANCE = 10


def load_filenames(txt_file_path):
    """Đọc danh sách tên file từ train_list.txt / val_list.txt / test_list.txt"""
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"Không tìm thấy file danh sách tại: {txt_file_path}")
    with open(txt_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def resolve_data_dir(data_dir):
    """Tự tìm thư mục chứa train_list.txt (phòng khi giải nén bị lồng thêm 1 cấp)."""
    if os.path.exists(os.path.join(data_dir, "train_list.txt")):
        return data_dir
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            if "train_list.txt" in files:
                return root
            if root.count(os.sep) - data_dir.count(os.sep) >= 2:
                break
    return data_dir


def rgb_mask_to_binary(mask_rgb):
    """Chuyển ảnh nhãn màu -> mask nhị phân (1 = hạt lem lép, 0 = nền)."""
    diff = np.abs(mask_rgb.astype(np.int16) - np.array(GD_COLOR_RGB, dtype=np.int16))
    return (diff.max(axis=-1) <= COLOR_TOLERANCE).astype(np.uint8)


def _imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _find_file(folder, base, exts):
    """Tìm file theo nhiều đuôi (Linux phân biệt .PNG và .png)."""
    for ext in exts:
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            return p
    return os.path.join(folder, base + exts[0])


class GDRGBMSDataset(Dataset):
    """
    Bộ nạp dữ liệu đa phương thức GD-RGBMS-DB
      - RGB : 3 kênh (JPG)
      - MS  : 3 kênh Green, Red-Edge, NIR (PNG)
      - Mask: ảnh màu, vùng hạt lem lép có màu [240, 128, 128]

    mode:
      - 'rgb_only'     -> tensor 3 kênh  [R, G, B]
      - 'ms_only'      -> tensor 3 kênh  [G, RE, NIR]
      - 'early_fusion' -> tensor 6 kênh  [R, G, B, G, RE, NIR]

    cache_dir: nếu khác None, ảnh sau khi resize được lưu thành .npz để các lần
               chạy sau (seed khác / mô hình khác) đọc lại rất nhanh.
    preload  : nạp toàn bộ ảnh đã resize vào RAM ngay khi khởi tạo.
    augment  : lật ngang / lật dọc / xoay 90° ngẫu nhiên (chỉ dùng cho tập train).
    """

    VALID_MODES = ("early_fusion", "rgb_only", "ms_only")

    def __init__(self, data_dir, list_filename, mode="early_fusion", img_size=(512, 512),
                 augment=False, cache_dir=None, preload=True, verbose=False):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Chế độ mode='{mode}' không được hỗ trợ! Chọn: {self.VALID_MODES}")

        self.data_dir = resolve_data_dir(data_dir)
        self.list_filename = list_filename
        self.filenames = load_filenames(os.path.join(self.data_dir, list_filename))
        self.mode = mode
        self.img_size = tuple(img_size)
        self.augment = augment
        self.preload = preload

        self.cache_dir = None
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, f"{self.img_size[0]}x{self.img_size[1]}")
            os.makedirs(self.cache_dir, exist_ok=True)

        self._mem = {}
        if preload:
            n = len(self.filenames)
            for i in range(n):
                self._mem[i] = self._load_sample(i)
                if verbose and ((i + 1) % 50 == 0 or i + 1 == n):
                    print(f"   [{list_filename}] đã nạp {i + 1}/{n} ảnh", flush=True)

    def __len__(self):
        return len(self.filenames)

    def get_filename(self, idx):
        return self.filenames[idx]

    # ------------------------------------------------------------------
    # Đọc ảnh gốc + resize
    # ------------------------------------------------------------------
    def _read_and_resize(self, base, rgb_name):
        w, h = self.img_size
        rgb = _imread_rgb(os.path.join(self.data_dir, "RGB", rgb_name))
        ms = _imread_rgb(_find_file(os.path.join(self.data_dir, "MS"), base, (".PNG", ".png")))
        mask_rgb = _imread_rgb(_find_file(os.path.join(self.data_dir, "Mark"), base, (".png", ".PNG")))

        # INTER_AREA cho ảnh khi thu nhỏ; INTER_NEAREST cho nhãn để không làm nhòe màu
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(ms, (w, h), interpolation=cv2.INTER_AREA)
        mask_rgb = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        return rgb, ms, rgb_mask_to_binary(mask_rgb)

    def _load_sample(self, idx):
        rgb_name = self.filenames[idx]
        base = os.path.splitext(rgb_name)[0]

        if self.cache_dir is None:
            return self._read_and_resize(base, rgb_name)

        cache_path = os.path.join(self.cache_dir, base + ".npz")
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as d:
                    return d["rgb"], d["ms"], d["mask"]
            except Exception:
                pass  # File cache hỏng -> đọc lại từ ảnh gốc

        rgb, ms, mask = self._read_and_resize(base, rgb_name)
        tmp_path = cache_path[:-4] + ".tmp.npz"
        np.savez(tmp_path, rgb=rgb, ms=ms, mask=mask)
        os.replace(tmp_path, cache_path)
        return rgb, ms, mask

    def _get(self, idx):
        if self.preload:
            return self._mem[idx]
        return self._load_sample(idx)

    # ------------------------------------------------------------------
    # Tăng cường dữ liệu (áp dụng giống nhau cho mọi kênh và mask)
    # ------------------------------------------------------------------
    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        if img.shape[0] == img.shape[1]:
            k = random.randint(0, 3)
            if k:
                img, mask = np.rot90(img, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
        return img, mask

    def __getitem__(self, idx):
        rgb, ms, mask = self._get(idx)

        if self.mode == "early_fusion":
            img = np.concatenate([rgb, ms], axis=-1)   # (H, W, 6)
        elif self.mode == "rgb_only":
            img = rgb                                  # (H, W, 3)
        else:
            img = ms                                   # (H, W, 3)

        if self.augment:
            img, mask = self._augment(img, mask)

        # (H, W, C) -> (C, H, W), chuẩn hóa về [0, 1]
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        mask = np.ascontiguousarray(mask[None], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask)
