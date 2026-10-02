"""Biểu đồ EDA và visualization cho các bước preprocessing.

Mỗi hàm trả về Figure; truyền `save_path` để lưu PNG.
"""
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .augmentation import TrainAugmentation
from .dataset import read_image_rgb
from .eda import QUALITY_FLAGS, is_low_quality
from .normalization import Normalizer
from .pipelines import CONFIGS, build_pipeline
from .preprocessing import apply_clahe, crop_retina, resize_image

# Bảng màu: 3 slot đầu của palette categorical (phân biệt được với người mù màu)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SPLIT_COLORS = {"train": BLUE, "valid": ORANGE, "test": AQUA}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def _finish(fig, save_path=None):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def _show(ax, image, title=None, fontsize=9, max_side=640):
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        # thu nhỏ ảnh chỉ để vẽ cho nhanh; extent giữ nguyên hệ toạ độ pixel của ảnh gốc
        image = cv2.resize(image, (round(w * max_side / max(h, w)), round(h * max_side / max(h, w))),
                           interpolation=cv2.INTER_AREA)
    ax.imshow(image, extent=(0, w, h, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="normal", loc="center")


def _shape_str(image) -> str:
    return f"{image.shape[1]}x{image.shape[0]}"


def normalized_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Tensor đã normalize -> ảnh hiển thị (min-max về [0, 1]) để nhìn trực tiếp giá trị sau normalize."""
    x = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-8)


# ============================================================================= EDA
def plot_class_distribution(rows, save_path=None):
    labels = list(config.CLASS_NAMES)
    names = [f"{k}\n{v}" for k, v in config.CLASS_NAMES.items()]
    total = len(rows)
    counts = [sum(r["label"] == k for r in rows) for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    bars = ax1.bar(names, counts, width=0.6, color=BLUE)
    ax1.bar_label(bars, [f"{c} ({100 * c / total:.1f}%)" for c in counts], padding=3, color=INK_2, fontsize=9)
    ax1.set_title(f"Images per class (total = {total})")
    ax1.set_ylabel("Number of images")
    ax1.set_ylim(0, max(counts) * 1.15)
    ax1.grid(axis="x", visible=False)

    x = np.arange(len(labels))
    width = 0.26
    for i, split in enumerate(config.SPLITS):
        c = [sum(r["label"] == k and r["split"] == split for r in rows) for k in labels]
        b = ax2.bar(x + (i - 1) * width, c, width=width - 0.03, color=SPLIT_COLORS[split],
                    label=f"{split} (n={sum(c)})")
        ax2.bar_label(b, padding=2, color=INK_2, fontsize=8)
    ax2.set_xticks(x, names)
    ax2.set_title("Images per class by split")
    ax2.set_ylabel("Number of images")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.12)
    ax2.legend(ncols=3, loc="upper right")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_image_sizes(rows, save_path=None, top_n=10):
    sizes = Counter((r["width"], r["height"]) for r in rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})

    w, h, c = zip(*[(w, h, c) for (w, h), c in sizes.items()])
    ax1.scatter(w, h, s=[20 + 6 * n for n in c], color=BLUE, alpha=0.55, edgecolor=SURFACE, linewidth=1)
    lim = max(max(w), max(h)) * 1.05
    ax1.plot([0, lim], [0, lim], color=AXIS, linewidth=1, linestyle="--", zorder=0)
    ax1.text(lim * 0.90, lim * 0.95, "square (1:1)", color=MUTED, fontsize=8, ha="right")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.set_title(f"Image resolutions ({len(sizes)} unique, marker area = image count)")

    top = sizes.most_common(top_n)[::-1]
    bars = ax2.barh([f"{w}x{h}" for (w, h), _ in top], [n for _, n in top], height=0.6, color=BLUE)
    ax2.bar_label(bars, padding=3, color=INK_2, fontsize=9)
    ax2.set_xlabel("Number of images")
    ax2.set_title(f"Top {len(top)} resolutions")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_brightness_contrast(rows, save_path=None):
    """Phân phối độ sáng / tương phản (trong vùng võng mạc) - tổng thể và theo class."""
    metrics = [("brightness_fg", "Brightness (mean gray, retina region)"),
               ("contrast_fg", "Contrast (std gray, retina region)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for col, (key, label) in enumerate(metrics):
        values = np.array([r[key] for r in rows])
        ax = axes[0, col]
        ax.hist(values, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.axvline(np.median(values), color=INK_2, linewidth=1.2, linestyle="--")
        ax.text(np.median(values), ax.get_ylim()[1] * 0.97, f" median {np.median(values):.1f}",
                color=INK_2, fontsize=9, va="top")
        ax.set_title(f"{label} - all images")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of images")
        ax.grid(axis="x", visible=False)

        ax = axes[1, col]
        groups = [[r[key] for r in rows if r["label"] == k] for k in config.CLASS_NAMES]
        ax.boxplot(groups, tick_labels=[f"{k}\n{v}" for k, v in config.CLASS_NAMES.items()], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), medianprops=dict(color=INK, linewidth=1.5),
                   whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
                   flierprops=dict(marker="o", markersize=4, markerfacecolor=BLUE, markeredgecolor="none", alpha=0.6))
        ax.set_title(f"{label.split(' (')[0]} by class")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_low_quality(rows, thresholds, save_path=None):
    low = [r for r in rows if is_low_quality(r)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))

    flags = list(QUALITY_FLAGS)[::-1]
    bars = ax1.barh(flags, [sum(r[f] for r in rows) for f in flags], height=0.55, color=BLUE)
    ax1.bar_label(bars, padding=3, color=INK_2, fontsize=9)
    ax1.set_title(f"Heuristic low-quality flags: {len(low)} of {len(rows)} images flagged")
    ax1.set_xlabel("Number of images (an image can carry several flags)")
    ax1.set_xlim(0, max(1, max(b.get_width() for b in bars)) * 1.15)
    ax1.grid(axis="y", visible=False)

    names = [f"{k}\n{v}" for k, v in config.CLASS_NAMES.items()]
    bars = ax2.bar(names, [sum(r["label"] == k for r in low) for k in config.CLASS_NAMES], width=0.55, color=BLUE)
    ax2.bar_label(bars, padding=3, color=INK_2, fontsize=9)
    ax2.set_title("Flagged images by class")
    ax2.set_ylabel("Number of images")
    ax2.set_ylim(0, max(1, max(b.get_height() for b in bars)) * 1.2)
    ax2.grid(axis="x", visible=False)

    fig.text(0.01, -0.03, f"Thresholds ({thresholds['method']}): brightness < {thresholds['brightness_fg_low']} or "
             f"> {thresholds['brightness_fg_high']}, contrast < {thresholds['contrast_fg_low']}, "
             f"sharpness < {thresholds['sharpness_low']}", color=MUTED, fontsize=8)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_low_quality_examples(rows, save_path=None, max_images=20, cols=5):
    low = [r for r in rows if is_low_quality(r)]
    if not low:
        return None
    low = sorted(low, key=lambda r: r["sharpness"])[:max_images]   # ảnh mờ nhất trước
    n_rows = int(np.ceil(len(low) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.5, n_rows * 2.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), low):
        ax.axis("on")
        image = read_image_rgb(config.DATA_ROOT / r["split"] / r["filename"])
        flags = ", ".join(f for f in QUALITY_FLAGS if r[f])
        _show(ax, resize_image(image, (320, round(320 * r["height"] / r["width"]))),
              f"{r['filename']}\nclass {r['label']} | {flags}\nB={r['brightness_fg']:.0f} C={r['contrast_fg']:.0f} "
              f"S={r['sharpness']:.0f}", fontsize=7)
    fig.suptitle("Flagged images, blurriest first (B = brightness, C = contrast, S = sharpness)", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_class_samples(samples, per_class=4, seed=config.SEED, save_path=None):
    """Ảnh mẫu đại diện cho từng class (chọn ngẫu nhiên có seed)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(config.NUM_CLASSES, per_class, figsize=(per_class * 2.6, config.NUM_CLASSES * 2.3))
    for row, (label, name) in enumerate(config.CLASS_NAMES.items()):
        pool = [s for s in samples if s.label == label]
        chosen = rng.choice(len(pool), size=per_class, replace=False)
        for col, idx in enumerate(chosen):
            image = read_image_rgb(pool[idx].path)
            thumb = resize_image(image, (384, round(384 * image.shape[0] / image.shape[1])))
            _show(axes[row, col], thumb, f"{_shape_str(image)} | {pool[idx].split}", fontsize=8)
        axes[row, 0].set_ylabel(f"{label} - {name}", fontsize=10, color=INK, fontweight="bold")
    fig.suptitle("Sample images per class", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, save_path)


# ============================================================================= Preprocessing
def plot_crop_examples(samples, save_path=None):
    """Original (kèm bbox) -> Cropped."""
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.7, 6.4), squeeze=False, layout="constrained")
    for col, s in enumerate(samples):
        image = read_image_rgb(s.path)
        cropped, info = crop_retina(image, return_info=True)
        _show(axes[0, col], image, f"Original {_shape_str(image)}\nclass {s.label}")
        if info["bbox"]:
            x0, y0, x1, y1 = info["bbox"]
            axes[0, col].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ffffff",
                                                 linewidth=1.2, linestyle="--"))
        _show(axes[1, col], cropped, f"Cropped {_shape_str(cropped)}" if info["cropped"]
              else f"Fallback (no crop) {_shape_str(cropped)}")
    fig.suptitle("Retinal crop: Original (dashed = detected box) -> Cropped", fontsize=12, fontweight="bold",
                 x=0.01, ha="left")
    return _finish(fig, save_path)


def plot_resize_check(samples, save_path=None):
    """Original -> resize trực tiếp 224x224 (E1) -> crop + resize 224x224 (E2)."""
    n = len(samples)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.7, 8.6), squeeze=False, layout="constrained")
    for col, s in enumerate(samples):
        image = read_image_rgb(s.path)
        direct = resize_image(image)
        cropped = resize_image(crop_retina(image))
        _show(axes[0, col], image, f"Original {_shape_str(image)}")
        _show(axes[1, col], direct, f"Resize only {_shape_str(direct)}")
        _show(axes[2, col], cropped, f"Crop + Resize {_shape_str(cropped)}")
    fig.suptitle(f"Resize to {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}", fontsize=12, fontweight="bold",
                 x=0.01, ha="left")
    return _finish(fig, save_path)


def plot_clahe_examples(samples, clip_limit=config.CLAHE_CLIP_LIMIT, tile_grid_size=config.CLAHE_TILE_GRID_SIZE,
                        save_path=None):
    """Trước / sau CLAHE (trên ảnh đã crop) và histogram kênh L."""
    n = len(samples)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.7, 8), squeeze=False)
    for col, s in enumerate(samples):
        before = crop_retina(read_image_rgb(s.path))
        after = apply_clahe(before, clip_limit, tile_grid_size)
        _show(axes[0, col], before, f"Before CLAHE\nclass {s.label}")
        _show(axes[1, col], after, "After CLAHE")
        ax = axes[2, col]
        for img, color, label in ((before, BLUE, "before"), (after, ORANGE, "after")):
            l = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]
            hist = np.bincount(l[l > 5].ravel(), minlength=256) / max(1, (l > 5).sum())
            ax.plot(hist, color=color, linewidth=1.6, label=label)
        ax.set_title("L channel histogram", fontsize=9, fontweight="normal", loc="center")
        ax.set_yticks([])
        ax.set_xlabel("L value", fontsize=8)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"CLAHE on L channel (LAB), clipLimit={clip_limit}, tileGridSize={tuple(tile_grid_size)}",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_clahe_params(sample, clip_limits=(1.0, 2.0, 4.0), tile_grids=((4, 4), (8, 8), (16, 16)), save_path=None):
    """Ảnh hưởng của clipLimit và tileGridSize trên một ảnh."""
    image = resize_image(crop_retina(read_image_rgb(sample.path)), (512, 512))
    fig, axes = plt.subplots(len(tile_grids), len(clip_limits) + 1,
                             figsize=((len(clip_limits) + 1) * 2.7, len(tile_grids) * 2.8), squeeze=False)
    for row, grid in enumerate(tile_grids):
        _show(axes[row, 0], image, "No CLAHE")
        for col, clip in enumerate(clip_limits, start=1):
            _show(axes[row, col], apply_clahe(image, clip, grid), f"clip={clip}, grid={grid}")
    fig.suptitle("CLAHE parameters", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_pipeline_stages(samples, seed=config.SEED, save_path=None):
    """Toàn bộ các bước của pipeline đầy đủ (E4, train) theo đúng thứ tự thực thi:
    Original -> Crop -> CLAHE -> Resize -> Augmentation -> Normalize."""
    torch.manual_seed(seed)
    pipeline = build_pipeline("E4", split="train")
    order = ["original", "crop", "clahe", "resize", "augmentation", "normalize"]
    titles = {"original": "Original", "crop": "Crop", "clahe": "CLAHE", "resize": "Resize",
              "augmentation": "Augmentation", "normalize": "Normalize"}
    fig, axes = plt.subplots(len(samples), len(order), figsize=(len(order) * 2.5, len(samples) * 2.9), squeeze=False,
                             layout="constrained")
    for row, s in enumerate(samples):
        stages = pipeline.run_with_stages(read_image_rgb(s.path))
        for col, key in enumerate(order):
            x = stages[key]
            if key == "normalize":
                _show(axes[row, col], normalized_to_display(x),
                      f"{titles[key]}\nfloat32 {tuple(x.shape)}\nrange [{x.min():.2f}, {x.max():.2f}]", fontsize=8)
            else:
                _show(axes[row, col], x, f"{titles[key]}\nuint8 {_shape_str(x)}", fontsize=8)
        axes[row, 0].set_ylabel(f"class {s.label}", fontsize=9, color=INK)
    fig.suptitle("Full preprocessing pipeline, step by step (Normalize shown min-max rescaled)", fontsize=12,
                 fontweight="bold", x=0.01, ha="left")
    return _finish(fig, save_path)


def plot_config_comparison(samples, seed=config.SEED, save_path=None):
    """So sánh output của E1-E4 (split train; ảnh hiển thị trước bước Normalize)."""
    torch.manual_seed(seed)
    pipelines = {name: build_pipeline(name, split="train", normalize=False) for name in CONFIGS}
    cols = ["Original"] + list(CONFIGS)
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(len(cols) * 2.6, len(samples) * 2.8), squeeze=False,
                             layout="constrained")
    for row, s in enumerate(samples):
        image = read_image_rgb(s.path)
        _show(axes[row, 0], image, f"Original {_shape_str(image)}", fontsize=8)
        axes[row, 0].set_ylabel(f"class {s.label}", fontsize=9, color=INK)
        for col, (name, pipe) in enumerate(pipelines.items(), start=1):
            out = pipe(image)
            _show(axes[row, col], out, f"{name}: {CONFIGS[name].description}\n{_shape_str(out)}", fontsize=8)
    fig.suptitle("Preprocessing configurations E1-E4 (train split, before Normalize)", fontsize=12,
                 fontweight="bold", x=0.01, ha="left")
    return _finish(fig, save_path)


def plot_augmentation_examples(sample, n=7, seed=config.SEED, save_path=None):
    """Một ảnh E3 và n phiên bản augmentation ngẫu nhiên của nó."""
    torch.manual_seed(seed)
    base = build_pipeline("E3", split="train", normalize=False)(read_image_rgb(sample.path))
    augment = TrainAugmentation()
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 2.1, 2.6), squeeze=False)
    _show(axes[0, 0], base, "No augmentation")
    for i in range(1, n + 1):
        _show(axes[0, i], augment(base), f"Augmented #{i}")
    cfg = config.AUGMENTATION
    fig.suptitle(f"Training augmentation: hflip p={cfg.hflip_prob}, rotation +-{cfg.rotation_degrees:g} deg, "
                 f"zoom {cfg.zoom_range}, brightness +-{cfg.brightness}, contrast +-{cfg.contrast}",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_normalization_check(sample, save_path=None):
    """Ảnh uint8 -> Normalize -> denormalize, kèm histogram giá trị trước/sau."""
    image = build_pipeline("E3", split="valid", normalize=False)(read_image_rgb(sample.path))
    normalizer = Normalizer()
    tensor = normalizer(image)
    restored = normalizer.denormalize(tensor)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    _show(axes[0], image, f"Before normalize\nuint8 [{image.min()}, {image.max()}]")
    _show(axes[1], normalized_to_display(tensor), f"Normalized (rescaled for display)\n"
          f"float32 [{tensor.min():.2f}, {tensor.max():.2f}]")
    _show(axes[2], restored, f"Denormalized\nmax abs diff = {int(np.abs(restored.astype(int) - image).max())}")
    for c, (name, color) in enumerate((("R", "#e34948"), ("G", "#008300"), ("B", BLUE))):
        axes[3].hist(tensor[c].numpy().ravel(), bins=60, histtype="step", linewidth=1.5, color=color, label=name)
    axes[3].set_title("Normalized values per channel", fontsize=9, fontweight="normal", loc="center")
    axes[3].set_yticks([])
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    return _finish(fig, save_path)
