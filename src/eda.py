"""EDA cho DDR Dataset: thống kê class, kích thước, độ sáng, tương phản, ảnh chất lượng thấp.

Kết quả là list[dict] (mỗi ảnh một dòng), lưu được ra CSV/JSON để dùng lại về sau.
"""
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import config
from .dataset import Sample, read_image_rgb
from .preprocessing import find_retina_bbox, foreground_mask

QUALITY_FLAGS = ("too_dark", "too_bright", "low_contrast", "blurry", "crop_failed")
_NUMERIC_INT = ("label", "width", "height", "crop_width", "crop_height")
_NUMERIC_FLOAT = ("file_size_kb", "aspect_ratio", "brightness_full", "contrast_full",
                  "brightness_fg", "contrast_fg", "fg_ratio", "sharpness")
_BOOL = ("crop_ok",) + QUALITY_FLAGS


def compute_image_stats(sample: Sample) -> dict:
    """Thống kê cho một ảnh.

    Độ sáng = mean grayscale, độ tương phản = std grayscale (thang 0-255), đo trên ảnh thu nhỏ
    (cạnh dài `EDA_ANALYSIS_SIZE`). Hậu tố `_full` tính trên toàn ảnh, `_fg` chỉ tính trong vùng
    võng mạc (bỏ nền đen). Sharpness = phương sai Laplacian trong vùng võng mạc.
    """
    image = read_image_rgb(sample.path)
    h, w = image.shape[:2]

    scale = max(h, w) / config.EDA_ANALYSIS_SIZE
    small = cv2.resize(image, (round(w / scale), round(h / scale)), interpolation=cv2.INTER_AREA) \
        if scale > 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask, _ = foreground_mask(small)
    has_fg = bool(mask.any())
    fg = gray[mask] if has_fg else gray.ravel()

    # bỏ viền võng mạc khi đo sharpness, vì mép võng mạc/nền đen luôn cho Laplacian lớn
    inner = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian[inner].var()) if inner.any() else float(laplacian.var())

    bbox = find_retina_bbox(image)
    crop_w, crop_h = (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else (w, h)

    return {
        "split": sample.split,
        "filename": sample.filename,
        "label": sample.label,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 4),
        "file_size_kb": round(sample.path.stat().st_size / 1024, 1),
        "brightness_full": round(float(gray.mean()), 3),
        "contrast_full": round(float(gray.std()), 3),
        "brightness_fg": round(float(fg.mean()), 3),
        "contrast_fg": round(float(fg.std()), 3),
        "fg_ratio": round(float(mask.mean()), 4),
        "sharpness": round(sharpness, 3),
        "crop_ok": bbox is not None,
        "crop_width": crop_w,
        "crop_height": crop_h,
    }


def compute_dataset_stats(samples: list, workers: int = 8) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(compute_image_stats, samples), total=len(samples), desc="EDA"))


# ----------------------------------------------------------------------------- Chất lượng ảnh
def _tukey_fences(values, k: float):
    q1, q3 = np.percentile(values, [25, 75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def flag_low_quality(rows: list, k: float = config.QUALITY_IQR_K) -> dict:
    """Gắn cờ ảnh chất lượng thấp (thêm cột bool vào từng row) và trả về các ngưỡng đã dùng.

    Dataset không có nhãn chất lượng, nên đây là heuristic: ảnh là ngoại lệ (Tukey fence,
    Q1 - k*IQR / Q3 + k*IQR) so với phân phối của chính dataset về độ sáng, tương phản,
    độ nét trong vùng võng mạc; hoặc ảnh không tìm được vùng võng mạc để crop.
    """
    bright_lo, bright_hi = _tukey_fences([r["brightness_fg"] for r in rows], k)
    contrast_lo, _ = _tukey_fences([r["contrast_fg"] for r in rows], k)
    # sharpness lệch phải mạnh -> tính fence trên thang log
    log_sharp_lo, _ = _tukey_fences(np.log([max(r["sharpness"], 1e-6) for r in rows]), k)
    sharp_lo = float(np.exp(log_sharp_lo))

    for r in rows:
        r["too_dark"] = r["brightness_fg"] < bright_lo
        r["too_bright"] = r["brightness_fg"] > bright_hi
        r["low_contrast"] = r["contrast_fg"] < contrast_lo
        r["blurry"] = r["sharpness"] < sharp_lo
        r["crop_failed"] = not r["crop_ok"]

    return {
        "method": f"Tukey fences, k={k}",
        "brightness_fg_low": round(bright_lo, 3),
        "brightness_fg_high": round(bright_hi, 3),
        "contrast_fg_low": round(contrast_lo, 3),
        "sharpness_low": round(sharp_lo, 3),
    }


def is_low_quality(row: dict) -> bool:
    return any(row[f] for f in QUALITY_FLAGS)


# ----------------------------------------------------------------------------- Tổng hợp
def _describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {k: round(float(x), 3) for k, x in
            dict(min=v.min(), q1=q1, median=med, q3=q3, max=v.max(), mean=v.mean(), std=v.std()).items()}


def class_distribution(rows: list) -> list:
    """Số ảnh và tỉ lệ % của từng class, tổng và theo từng split."""
    total = len(rows)
    out = []
    for label, name in config.CLASS_NAMES.items():
        sub = [r for r in rows if r["label"] == label]
        entry = {"label": label, "class_name": name, "count": len(sub),
                 "percent": round(100 * len(sub) / total, 2)}
        for split in config.SPLITS:
            entry[split] = sum(r["split"] == split for r in sub)
        out.append(entry)
    return out


def summarize(rows: list, thresholds: dict) -> dict:
    sizes = Counter((r["width"], r["height"]) for r in rows)
    low = [r for r in rows if is_low_quality(r)]
    per_class = {}
    for label, name in config.CLASS_NAMES.items():
        sub = [r for r in rows if r["label"] == label]
        per_class[name] = {
            "brightness_fg": _describe([r["brightness_fg"] for r in sub]),
            "contrast_fg": _describe([r["contrast_fg"] for r in sub]),
        }
    return {
        "total_images": len(rows),
        "images_per_split": {s: sum(r["split"] == s for r in rows) for s in config.SPLITS},
        "class_distribution": class_distribution(rows),
        "image_size": {
            "width": _describe([r["width"] for r in rows]),
            "height": _describe([r["height"] for r in rows]),
            "aspect_ratio": _describe([r["aspect_ratio"] for r in rows]),
            "unique_resolutions": len(sizes),
            "most_common": [{"width": w, "height": h, "count": c} for (w, h), c in sizes.most_common(10)],
        },
        "brightness_full": _describe([r["brightness_full"] for r in rows]),
        "contrast_full": _describe([r["contrast_full"] for r in rows]),
        "brightness_fg": _describe([r["brightness_fg"] for r in rows]),
        "contrast_fg": _describe([r["contrast_fg"] for r in rows]),
        "sharpness": _describe([r["sharpness"] for r in rows]),
        "fg_ratio": _describe([r["fg_ratio"] for r in rows]),
        "brightness_contrast_per_class": per_class,
        "low_quality": {
            "thresholds": thresholds,
            "total_flagged": len(low),
            "per_flag": {f: sum(r[f] for r in rows) for f in QUALITY_FLAGS},
            "per_class": {name: sum(r["label"] == label for r in low)
                          for label, name in config.CLASS_NAMES.items()},
        },
    }


# ----------------------------------------------------------------------------- Lưu / đọc
def save_csv(rows: list, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_stats(path=None) -> list:
    """Đọc lại `image_stats.csv` đã lưu, ép kiểu các cột số/bool."""
    path = path or config.EDA_DIR / "image_stats.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in _NUMERIC_INT:
            r[k] = int(r[k])
        for k in _NUMERIC_FLOAT:
            r[k] = float(r[k])
        for k in _BOOL:
            if k in r:
                r[k] = r[k] == "True"
    return rows
