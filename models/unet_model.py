import segmentation_models_pytorch as smp

# Tên kiến trúc (viết thường, bỏ dấu '-' '_') -> class trong segmentation_models_pytorch
_ARCHS = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "deeplabv3": "DeepLabV3",
    "deeplabv3plus": "DeepLabV3Plus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "manet": "MAnet",
    "linknet": "Linknet",
    "pan": "PAN",
}

ARCH_DISPLAY = {
    "unet": "U-Net",
    "unetplusplus": "U-Net++",
    "deeplabv3": "DeepLabV3",
    "deeplabv3plus": "DeepLabV3+",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "manet": "MA-Net",
    "linknet": "LinkNet",
    "pan": "PAN",
}


def _arch_key(arch):
    return arch.lower().replace("-", "").replace("_", "").replace("+", "plus")


def arch_display_name(arch, backbone=None):
    name = ARCH_DISPLAY.get(_arch_key(arch), arch)
    return f"{name} ({backbone})" if backbone else name


def build_model(arch="unet", backbone="resnet34", in_channels=6, num_classes=1,
                pretrained_encoder=False, encoder_weights="auto"):
    """
    Hàm khởi tạo mô hình phân vùng ảnh (Semantic Segmentation).
    Hỗ trợ thay đổi số kênh đầu vào linh hoạt (3 kênh hoặc 6 kênh).

    encoder_weights:
      - "auto"     : giữ hành vi cũ (chỉ dùng ImageNet khi pretrained_encoder=True và 3 kênh)
      - "imagenet" : dùng trọng số ImageNet. Với 6 kênh, smp tự điều chỉnh lớp conv đầu tiên
      - None/"none": khởi tạo ngẫu nhiên
    """
    if encoder_weights == "auto":
        encoder_weights = "imagenet" if (pretrained_encoder and in_channels == 3) else None
    if isinstance(encoder_weights, str) and encoder_weights.lower() in ("none", "null", ""):
        encoder_weights = None

    key = _arch_key(arch)
    if key not in _ARCHS:
        raise ValueError(f"Kiến trúc '{arch}' chưa được khai báo! Chọn một trong: {list(_ARCHS)}")

    model_cls = getattr(smp, _ARCHS[key])
    return model_cls(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
