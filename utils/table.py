"""Tiện ích in bảng đẹp ra terminal (không cần thư viện ngoài) và xuất Markdown."""
import math
import sys


def force_utf8_stdout():
    """Tránh lỗi in tiếng Việt / ký tự kẻ bảng trên terminal Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_num(x, digits=2):
    if is_nan(x):
        return "—"
    return f"{x:.{digits}f}"


def mean_std(values):
    vals = [float(v) for v in values if not is_nan(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)   # độ lệch chuẩn mẫu (ddof=1)
    return m, math.sqrt(var)


def fmt_mean_std(values, digits=2):
    vals = [v for v in values if not is_nan(v)]
    if not vals:
        return "—"
    m, s = mean_std(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _fmt_cell(text, width, align):
    if align == "l":
        return text.ljust(width)
    if align == "r":
        return text.rjust(width)
    return text.center(width)


def render_table(headers, rows, title=None, align=None, footer=None, note=None):
    headers = [str(h) for h in headers]
    rows = [[str(c) for c in r] for r in rows]
    footer = [[str(c) for c in r] for r in (footer or [])]
    n = len(headers)
    if align is None:
        align = ["l"] + ["c"] * (n - 1)

    widths = [len(h) for h in headers]
    for r in rows + footer:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(left, mid, right, ch="─"):
        return left + mid.join(ch * (w + 2) for w in widths) + right

    def row_str(r):
        return "│" + "│".join(" " + _fmt_cell(c, widths[i], align[i]) + " " for i, c in enumerate(r)) + "│"

    total_w = sum(widths) + 3 * n + 1
    out = []
    if title:
        out.append("")
        out.append(title.center(total_w) if len(title) < total_w else title)
    out.append(line("┌", "┬", "┐"))
    out.append(row_str(headers))
    out.append(line("╞", "╪", "╡", "═"))
    for r in rows:
        out.append(row_str(r))
    if footer:
        out.append(line("├", "┼", "┤"))
        for r in footer:
            out.append(row_str(r))
    out.append(line("└", "┴", "┘"))
    if note:
        out.append(note)
    return "\n".join(out)


def print_table(headers, rows, title=None, align=None, footer=None, note=None):
    text = render_table(headers, rows, title=title, align=align, footer=footer, note=note)
    print(text, flush=True)
    return text


def to_markdown(headers, rows, title=None, footer=None, note=None):
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("|" + "|".join([":---"] + [":---:"] * (len(headers) - 1)) + "|")
    for r in list(rows) + list(footer or []):
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    return "\n".join(lines)
