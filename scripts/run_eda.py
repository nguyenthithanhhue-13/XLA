"""Chạy EDA trên toàn bộ DDR Dataset, lưu thống kê và biểu đồ vào outputs/eda/.

    python scripts/run_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

from src import config, eda, visualization as viz  # noqa: E402
from src.dataset import load_all  # noqa: E402


def main() -> None:
    out = config.EDA_DIR
    out.mkdir(parents=True, exist_ok=True)
    viz.set_style()

    samples = load_all()
    missing = [s.path for s in samples if not s.path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) listed in the label files are missing, e.g. {missing[0]}")

    rows = eda.compute_dataset_stats(samples)
    thresholds = eda.flag_low_quality(rows)
    summary = eda.summarize(rows, thresholds)

    eda.save_csv(rows, out / "image_stats.csv")
    eda.save_csv(summary["class_distribution"], out / "class_distribution.csv")
    eda.save_json(summary, out / "eda_summary.json")
    low = [r for r in rows if eda.is_low_quality(r)]
    if low:
        eda.save_csv(low, out / "low_quality_images.csv")

    viz.plot_class_distribution(rows, out / "class_distribution.png")
    viz.plot_image_sizes(rows, out / "image_sizes.png")
    viz.plot_brightness_contrast(rows, out / "brightness_contrast.png")
    viz.plot_low_quality(rows, thresholds, out / "low_quality_summary.png")
    viz.plot_low_quality_examples(rows, out / "low_quality_examples.png")
    viz.plot_class_samples(samples, save_path=out / "class_samples.png")

    print(f"Total images: {summary['total_images']}  {summary['images_per_split']}")
    for c in summary["class_distribution"]:
        print(f"  class {c['label']} ({c['class_name']:<14}): {c['count']:>4}  {c['percent']:5.2f}%  "
              f"train={c['train']} valid={c['valid']} test={c['test']}")
    size = summary["image_size"]
    print(f"Width : min {size['width']['min']:.0f}  median {size['width']['median']:.0f}  max {size['width']['max']:.0f}")
    print(f"Height: min {size['height']['min']:.0f}  median {size['height']['median']:.0f}  max {size['height']['max']:.0f}")
    print(f"Unique resolutions: {size['unique_resolutions']}")
    for key in ("brightness_fg", "contrast_fg", "brightness_full", "contrast_full", "sharpness"):
        d = summary[key]
        print(f"{key:<16}: mean {d['mean']:.2f}  std {d['std']:.2f}  min {d['min']:.2f}  max {d['max']:.2f}")
    lq = summary["low_quality"]
    print(f"Low-quality flags ({thresholds['method']}): {lq['total_flagged']} image(s)  {lq['per_flag']}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
