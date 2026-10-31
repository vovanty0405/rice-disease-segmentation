import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def load_filenames(txt_file_path):
    """Đọc danh sách tên file từ các file train_list.txt, val_list.txt, test_list.txt"""
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"Không tìm thấy file danh sách tại: {txt_file_path}")
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

class GDRGBMSDataset(Dataset):
    """
    Bộ nạp dữ liệu đa phương thức ảnh lúa (GD-RGBMS-DB)
    - RGB: 3 kênh (JPG)
    - MS: 3 kênh Green, Red-Edge, NIR (PNG)
    - Mask: Ảnh màu RGB với nhãn hạt lép có màu [240, 128, 128]
    """
    def __init__(self, data_dir, list_filename, mode='early_fusion', img_size=(512, 512)):
        self.data_dir = data_dir
        txt_path = os.path.join(data_dir, list_filename)
        self.filenames = load_filenames(txt_path)
        self.mode = mode
        self.img_size = img_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        rgb_name = self.filenames[idx]
        base_name = os.path.splitext(rgb_name)[0]

        rgb_path = os.path.join(self.data_dir, "RGB", rgb_name)
        ms_path = os.path.join(self.data_dir, "MS", base_name + ".PNG")
        mask_path = os.path.join(self.data_dir, "Mark", base_name + ".png")

        # 1. Đọc và chuẩn hóa ảnh RGB (3 kênh)
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise ValueError(f"Không thể đọc ảnh RGB: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.img_size).astype(np.float32) / 255.0

        # 2. Đọc và chuẩn hóa ảnh Đa phổ MS (3 kênh: Green, Red-Edge, NIR)
        ms = cv2.imread(ms_path)
        if ms is None:
            raise ValueError(f"Không thể đọc ảnh MS: {ms_path}")
        ms = cv2.cvtColor(ms, cv2.COLOR_BGR2RGB)
        ms = cv2.resize(ms, self.img_size).astype(np.float32) / 255.0

        # 3. Đọc Mask nhãn và lọc mã màu bệnh [240, 128, 128]
        mask_rgb = cv2.imread(mask_path)
        if mask_rgb is None:
            raise ValueError(f"Không thể đọc ảnh Mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Hạt lem lép có kênh Red=240, Green=128
        disease_mask = (mask_rgb[:, :, 0] == 240) & (mask_rgb[:, :, 1] == 128)
        mask = disease_mask.astype(np.float32)

        # 4. Hợp nhất hoặc lựa chọn phương thức dữ liệu
        if self.mode == 'early_fusion':
            # Ghép thành 6 kênh: [RGB + MS]
            img = np.concatenate([rgb, ms], axis=-1)
        elif self.mode == 'rgb_only':
            img = rgb
        elif self.mode == 'ms_only':
            img = ms
        else:
            raise ValueError(f"Chế độ mode='{self.mode}' không được hỗ trợ!")

        # 5. Chuyển sang định dạng PyTorch tensor (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
