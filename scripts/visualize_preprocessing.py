"""Tạo các hình kiểm tra preprocessing (crop, resize, CLAHE, normalize, augmentation, E1-E4)
trên ảnh thật, lưu vào outputs/preprocessing/.

    python scripts/visualize_preprocessing.py [--per-class 1] [--seed 42]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from src import config, visualization as viz  # noqa: E402
from src.dataset import load_split  # noqa: E402


def pick_samples(per_class: int, seed: int) -> list:
    """Chọn ngẫu nhiên (có seed) `per_class` ảnh train cho mỗi class."""
    rng = np.random.default_rng(seed)
    train = load_split("train")
    chosen = []
    for label in config.CLASS_NAMES:
        pool = [s for s in train if s.label == label]
        chosen += [pool[i] for i in rng.choice(len(pool), size=per_class, replace=False)]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    out = config.PREPROCESS_DIR
    out.mkdir(parents=True, exist_ok=True)
    viz.set_style()
    samples = pick_samples(args.per_class, args.seed)

    viz.plot_crop_examples(samples, out / "01_crop.png")
    viz.plot_resize_check(samples, out / "02_resize.png")
    viz.plot_clahe_examples(samples, save_path=out / "03_clahe.png")
    viz.plot_clahe_params(samples[2], save_path=out / "03_clahe_params.png")
    viz.plot_normalization_check(samples[2], out / "04_normalize.png")
    viz.plot_augmentation_examples(samples[2], seed=args.seed, save_path=out / "05_augmentation.png")
    viz.plot_pipeline_stages(samples, seed=args.seed, save_path=out / "06_pipeline_stages.png")
    viz.plot_config_comparison(samples, seed=args.seed, save_path=out / "07_configs_E1_E4.png")
    print(f"Saved {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
