"""
Script huấn luyện mô hình KẾT HỢP ĐA PHƯƠNG THỨC (RGB + MS: Early Fusion 6 kênh)
Chạy:
    python train_rgb_ms.py
Hoặc tùy biến tham số:
    python train_rgb_ms.py --epochs 20 --batch_size 4 --lr 1e-4 --seeds 42 123 2024
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.engine import main_for_mode

if __name__ == "__main__":
    main_for_mode("early_fusion")

