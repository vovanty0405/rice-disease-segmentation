import os

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # Cho phép chạy benchmark.py trên máy không cài PyTorch
    _HAS_CUDA = False


class Config:
    # ------------------------------------------------------------------
    # 1. Đường dẫn (có thể ghi đè bằng biến môi trường hoặc tham số dòng lệnh)
    # ------------------------------------------------------------------
    # Thư mục dataset sau khi giải nén trên Colab
    DATA_DIR = os.environ.get("GD_DATA_DIR", "/content/GD-RGBMS-DB")

    # Thư mục gốc lưu kết quả thực nghiệm trên Google Drive
    DRIVE_EXPERIMENT_ROOT = os.environ.get(
        "GD_EXP_DIR", "/content/drive/MyDrive/Khóa Luận TN/Experiments"
    )

    # Thư mục riêng cho bộ benchmark nhiều seed (RGB / MS / RGB+MS)
    BENCHMARK_ROOT = os.environ.get(
        "GD_BENCHMARK_DIR", os.path.join(DRIVE_EXPERIMENT_ROOT, "Benchmark")
    )

    # Cache ảnh đã resize (lưu trên ổ máy ảo Colab để các lần chạy sau nhanh hơn)
    CACHE_DIR = os.environ.get("GD_CACHE_DIR", "/content/cache_gd")

    # ------------------------------------------------------------------
    # 2. Chế độ dữ liệu (dùng cho train.py / test.py cũ)
    # ------------------------------------------------------------------
    # Các chế độ hỗ trợ: 'early_fusion' (6 kênh), 'rgb_only' (3 kênh), 'ms_only' (3 kênh)
    MODE = "early_fusion"
    IN_CHANNELS = 6 if MODE == "early_fusion" else 3

    # Kích thước ảnh resize để đưa vào mô hình huấn luyện
    IMG_SIZE = (512, 512)

    # ------------------------------------------------------------------
    # 3. Cấu hình mô hình
    # ------------------------------------------------------------------
    MODEL_ARCH = "unet"          # unet | unetplusplus | deeplabv3plus | fpn | ...
    BACKBONE = "resnet34"        # Mạng trích xuất đặc trưng (encoder)
    NUM_CLASSES = 1              # Nhị phân: Hạt lem lép (1) vs Nền (0)
    ENCODER_WEIGHTS = "imagenet" # "imagenet" hoặc "none" (dùng cho bộ benchmark)
    CLASS_NAMES = ("Background", "GD")

    # ------------------------------------------------------------------
    # 4. Siêu tham số huấn luyện (Hyperparameters)
    # ------------------------------------------------------------------
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LOSS = "dice_bce"            # dice | dice_bce | focal_dice

    # 5 seed khác nhau -> chạy 5 lần để tính mean ± std, p-value, Cohen's d
    SEEDS = [42, 123, 2024, 3407, 777]

    # ------------------------------------------------------------------
    # 5. Đăng ký các kịch bản thực nghiệm của bộ benchmark
    # ------------------------------------------------------------------
    EXPERIMENTS = {
        "rgb_only": {
            "suffix": "RGB_ONLY",
            "label": "RGB only",
            "input": "R-G-B",
            "in_channels": 3,
        },
        "ms_only": {
            "suffix": "MS_ONLY",
            "label": "MS only",
            "input": "G-RE-NIR",
            "in_channels": 3,
        },
        "early_fusion": {
            "suffix": "RGB_MS",
            "label": "RGB + MS (Early Fusion)",
            "input": "R-G-B + G-RE-NIR",
            "in_channels": 6,
        },
    }
    MODE_ORDER = ["rgb_only", "ms_only", "early_fusion"]

    # ------------------------------------------------------------------
    # 6. Thiết bị
    # ------------------------------------------------------------------
    DEVICE = "cuda" if _HAS_CUDA else "cpu"
