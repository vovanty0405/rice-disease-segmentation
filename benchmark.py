"""
Script BENCHMARK tổng hợp số liệu, xuất bảng so sánh ra terminal chuẩn bài báo khoa học.
Chức năng:
  - Tự động quét các kết quả `results.json` trong thư mục Benchmark
  - Bảng 1: So sánh hiệu năng các phương thức (mIoU, Pixel Accuracy, mF1, mPrecision, mRecall, Inference Time s/it) theo định dạng Paper Table 4
  - Bảng 2: Phân tích thống kê qua các lần chạy (mean ± std), kiểm định p-value (t-test) và Cohen's d so với mô hình tốt nhất (hoặc RGB+MS) theo Paper Table 2
  - Bảng 3: Chi tiết riêng cho lớp Hạt lem lép (GD) - đối tượng chính của bài toán
  - Tự động xuất ra file Markdown và CSV để bạn chèn thẳng vào báo cáo khóa luận / Word / LaTeX.

Cách chạy:
    python benchmark.py
    python benchmark.py --benchmark_dir "/content/drive/MyDrive/Khóa Luận TN/Experiments/Benchmark"
"""
import os
import sys
import glob
import json
import math
import argparse
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from configs.config import Config
from utils.table import print_table, to_markdown, mean_std, fmt_num, fmt_mean_std, force_utf8_stdout


def compute_p_value_and_cohen_d(baseline_vals, target_vals):
    """
    Tính p-value (Paired Student's t-test hoặc Independent t-test) và Cohen's d
    để kiểm định xem sự vượt trội có ý nghĩa thống kê hay chỉ là ngẫu nhiên.
    """
    b = np.array(baseline_vals, dtype=np.float64)
    t = np.array(target_vals, dtype=np.float64)

    if len(b) < 2 or len(t) < 2:
        return float("nan"), float("nan")

    # 1. Tính p-value bằng scipy.stats
    if _HAS_SCIPY:
        if len(b) == len(t):
            # Nếu cùng số seed -> paired t-test
            _, p_val = stats.ttest_rel(target_vals, baseline_vals)
        else:
            _, p_val = stats.ttest_ind(target_vals, baseline_vals, equal_var=False)
    else:
        p_val = float("nan")

    # 2. Tính Cohen's d: (Mean_target - Mean_baseline) / Pooled_std
    n1, n2 = len(b), len(t)
    var1, var2 = np.var(b, ddof=1), np.var(t, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1e-6
    cohen_d = (np.mean(t) - np.mean(b)) / (pooled_std if pooled_std > 0 else 1e-6)

    return p_val, cohen_d


def fmt_p_value(p):
    if p is None or math.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def load_all_results(benchmark_root):
    """Quét và gom toàn bộ file results.json theo từng thực nghiệm"""
    data = {}
    pattern = os.path.join(benchmark_root, "*", "seed_*", "results.json")
    json_files = glob.glob(pattern)

    if not json_files:
        # Tìm thử ở cấp con khác nếu có
        pattern = os.path.join(benchmark_root, "**", "results.json")
        json_files = glob.glob(pattern, recursive=True)

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                res = json.load(f)
            exp_name = res.get("exp_name") or os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            if exp_name not in data:
                data[exp_name] = []
            data[exp_name].append(res)
        except Exception as e:
            print(f"Bỏ qua file lỗi {fpath}: {e}")

    return data


def generate_benchmark_reports(benchmark_root, save_dir=None):
    force_utf8_stdout()
    if save_dir is None:
        save_dir = benchmark_root
    os.makedirs(save_dir, exist_ok=True)

    data = load_all_results(benchmark_root)
    if not data:
        print(f"\n[THÔNG BÁO] Chưa tìm thấy kết quả chạy nào tại: {benchmark_root}")
        print("Vui lòng chạy ít nhất một trong các lệnh sau trước khi xem benchmark:")
        print("   python train_rgb_only.py")
        print("   python train_ms_only.py")
        print("   python train_rgb_ms.py\n")
        return

    all_markdown = []
    print("\n" + "=" * 105)
    print("                    BÁO CÁO BENCHMARK VÀ PHÂN TÍCH THỐNG KÊ (GD-RGBMS-DB)                    ")
    print(f"                    Thư mục dữ liệu: {benchmark_root}")
    print("=" * 105)

    # Xác định mô hình cơ sở để so sánh (ưu tiên mô hình Early Fusion / RGB_MS)
    baseline_exp = None
    for k in data.keys():
        if "RGB_MS" in k.upper() or "EARLY" in k.upper():
            baseline_exp = k
            break
    if not baseline_exp and data:
        baseline_exp = list(data.keys())[0]

    # ------------------------------------------------------------------
    # BẢNG 1: Tổng hợp hiệu năng theo mẫu Table 4 (Paper mẫu)
    # Các cột: Model | Input Modality | mIoU (%) | mAccuracy (%) | mF1 (%) | mPrecision (%) | mRecall (%) | Inference Time (s/it)
    # ------------------------------------------------------------------
    headers_t1 = ["Mô hình", "Phương thức vào", "Số kênh", "mIoU (%)", "mAccuracy (%)", "mF1 (%)", "mPrecision (%)", "mRecall (%)", "Inference (s/it)", "FPS"]
    rows_t1 = []
    
    for exp_name, runs in data.items():
        sample = runs[0]
        model_name = sample.get("model_label", exp_name)
        input_desc = sample.get("input_label", sample.get("mode", "Custom"))
        in_ch = sample.get("in_channels", 3)

        miou_str = fmt_mean_std([r["test"]["mIoU"] for r in runs])
        acc_str  = fmt_mean_std([r["test"]["PixelAcc"] for r in runs])
        mf1_str  = fmt_mean_std([r["test"]["mF1"] for r in runs])
        prec_str = fmt_mean_std([r["test"]["mPrecision"] for r in runs])
        rec_str  = fmt_mean_std([r["test"]["mRecall"] for r in runs])
        time_str = fmt_mean_std([r.get("inference_time_s", float("nan")) for r in runs], 4)
        fps_str  = fmt_mean_std([r.get("fps", float("nan")) for r in runs], 1)

        rows_t1.append([model_name, input_desc, in_ch, miou_str, acc_str, mf1_str, prec_str, rec_str, time_str, fps_str])

    title_t1 = "Table 1. Performance of models using different combinations of input datasets (mean ± std)"
    print_table(headers_t1, rows_t1, title=title_t1)
    all_markdown.append(to_markdown(headers_t1, rows_t1, title=title_t1))

    # ------------------------------------------------------------------
    # BẢNG 2: Kiểm định thống kê so với mô hình Đa phương thức (Table 2 của Paper)
    # Model | mIoU (mean ± std) | F1 (mean ± std) | p-Value (mIoU) | p-Value (F1) | Cohen's d
    # ------------------------------------------------------------------
    headers_t2 = ["Mô hình & Phương thức", "Số lần chạy", "mIoU (%) mean ± std", "F1 (%) mean ± std", f"p-Value (mIoU)", f"p-Value (F1)", "Cohen's d"]
    rows_t2 = []

    target_runs = data.get(baseline_exp, [])
    target_mious = [r["test"]["mIoU"] for r in target_runs]
    target_f1s = [r["test"]["mF1"] for r in target_runs]

    for exp_name, runs in data.items():
        model_name = f"{runs[0].get('model_label', exp_name)} - {runs[0].get('input_label', '')}"
        n_runs = len(runs)
        miou_str = fmt_mean_std([r["test"]["mIoU"] for r in runs])
        mf1_str  = fmt_mean_std([r["test"]["mF1"] for r in runs])

        if exp_name == baseline_exp:
            p_miou_str, p_f1_str, cohen_str = "— (Chuẩn)", "— (Chuẩn)", "— (Baseline)"
        else:
            cur_mious = [r["test"]["mIoU"] for r in runs]
            cur_f1s = [r["test"]["mF1"] for r in runs]
            p_miou, d_miou = compute_p_value_and_cohen_d(cur_mious, target_mious)
            p_f1, _ = compute_p_value_and_cohen_d(cur_f1s, target_f1s)
            p_miou_str = fmt_p_value(p_miou)
            p_f1_str   = fmt_p_value(p_f1)
            cohen_str  = f"{d_miou:.2f}" if not math.isnan(d_miou) else "—"

        rows_t2.append([model_name, n_runs, miou_str, mf1_str, p_miou_str, p_f1_str, cohen_str])

    title_t2 = f"Table 2. Statistical analysis over multiple runs and significance testing against {baseline_exp}"
    print_table(headers_t2, rows_t2, title=title_t2, note="Ghi chú: p-value < 0.05 thể hiện sự khác biệt có ý nghĩa thống kê; Cohen's d > 0.8 biểu thị mức độ tác động rất lớn.")
    all_markdown.append(to_markdown(headers_t2, rows_t2, title=title_t2, note="p-value < 0.05 thể hiện sự khác biệt có ý nghĩa thống kê."))

    # ------------------------------------------------------------------
    # BẢNG 3: Bảng chỉ số chuyên sâu cho lớp Bệnh lem lép hạt (GD Class Only)
    # ------------------------------------------------------------------
    headers_t3 = ["Mô hình", "Phương thức", "IoU Lớp GD (%)", "F1/Dice Lớp GD (%)", "Precision GD (%)", "Recall GD (%)"]
    rows_t3 = []
    for exp_name, runs in data.items():
        m_name = runs[0].get("model_label", exp_name)
        inp_desc = runs[0].get("input_label", "")
        iou_gd = fmt_mean_std([r["test"]["IoU_GD"] for r in runs])
        f1_gd  = fmt_mean_std([r["test"]["F1_GD"] for r in runs])
        p_gd   = fmt_mean_std([r["test"]["Precision_GD"] for r in runs])
        r_gd   = fmt_mean_std([r["test"]["Recall_GD"] for r in runs])
        rows_t3.append([m_name, inp_desc, iou_gd, f1_gd, p_gd, r_gd])

    title_t3 = "Table 3. Detailed Segmentation Performance on Target Class (Grain Discoloration - GD)"
    print_table(headers_t3, rows_t3, title=title_t3)
    all_markdown.append(to_markdown(headers_t3, rows_t3, title=title_t3))

    # Lưu toàn bộ báo cáo Markdown vào thư mục
    md_path = os.path.join(save_dir, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO KẾT QUẢ THỰC NGHIỆM ĐA PHƯƠNG THỨC GD-RGBMS-DB\n\n")
        f.write(f"*Thời gian xuất báo cáo:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n\n".join(all_markdown))

    print(f"\n--> Đã tự động lưu toàn bộ báo cáo dưới dạng Markdown tại: {md_path}")
    print("    (Bạn có thể mở file này để copy thẳng bảng vào Word hoặc slide thuyết trình!)")


def main():
    parser = argparse.ArgumentParser(description="Tổng hợp benchmark hiệu năng các phương thức dữ liệu GD-RGBMS-DB")
    parser.add_argument("--benchmark_dir", default=Config.BENCHMARK_ROOT,
                        help="Đường dẫn thư mục chứa các kết quả run (mặc định lấy từ Config.BENCHMARK_ROOT)")
    parser.add_argument("--save_dir", default=None, help="Thư mục lưu báo cáo markdown (mặc định lưu cùng benchmark_dir)")
    args = parser.parse_args()

    generate_benchmark_reports(args.benchmark_dir, args.save_dir)


if __name__ == "__main__":
    main()

