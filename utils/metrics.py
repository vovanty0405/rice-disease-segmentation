import numpy as np
import torch

CLASS_NAMES = ("Background", "GD")


def compute_iou_and_dice(preds, targets, threshold=0.5, smooth=1e-6):
    """
    (Hàm cũ, giữ lại cho test.py) Tính IoU và Dice Score cho lớp đối tượng trên 1 batch.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum().item()
    total_pred = preds.sum().item()
    total_target = targets.sum().item()

    dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
    union = total_pred + total_target - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou, dice


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b > 0)
    return out


class BinarySegMetric:
    """
    Tích lũy ma trận nhầm lẫn (confusion matrix) trên TOÀN BỘ tập dữ liệu rồi mới tính chỉ số
    (cách tính chuẩn của các framework segmentation như MMSegmentation).

    Với mỗi lớp i (Background, GD):
        IoU_i       = TP / (TP + FP + FN)
        Precision_i = TP / (TP + FP)
        Recall_i    = TP / (TP + FN)
        F1_i        = 2 * Precision * Recall / (Precision + Recall)   (= Dice)
    Chỉ số trung bình:
        mIoU = (1/N) * Σ IoU_i ,   mF1, mPrecision, mRecall tương tự
        Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """

    def __init__(self, threshold=0.5, class_names=CLASS_NAMES):
        self.threshold = threshold
        self.class_names = tuple(class_names)
        self.reset()

    def reset(self):
        # Hàng = nhãn thật, cột = dự đoán. [0]=Background, [1]=GD
        self.cm = np.zeros((2, 2), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits, targets):
        preds = (torch.sigmoid(logits.float()) > self.threshold).long().view(-1)
        gts = (targets > 0.5).long().view(-1)
        cm = torch.bincount(gts * 2 + preds, minlength=4).reshape(2, 2)
        self.cm += cm.cpu().numpy().astype(np.int64)

    def compute(self):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = _safe_div(tp, tp + fp + fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        total = cm.sum()
        pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

        res = {
            "mIoU": float(iou.mean() * 100),
            "PixelAcc": pixel_acc * 100,
            "mF1": float(f1.mean() * 100),
            "mPrecision": float(precision.mean() * 100),
            "mRecall": float(recall.mean() * 100),
            # Lối tắt cho lớp GD (hạt lem lép) - lớp quan trọng nhất của bài toán
            "IoU_GD": float(iou[1] * 100),
            "F1_GD": float(f1[1] * 100),
            "Precision_GD": float(precision[1] * 100),
            "Recall_GD": float(recall[1] * 100),
            "per_class": {},
            "confusion_matrix": self.cm.tolist(),
        }
        for i, name in enumerate(self.class_names):
            res["per_class"][name] = {
                "IoU": float(iou[i] * 100),
                "F1": float(f1[i] * 100),
                "Precision": float(precision[i] * 100),
                "Recall": float(recall[i] * 100),
            }
        return res
