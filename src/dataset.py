"""Đọc danh sách ảnh/nhãn của DDR Dataset và đọc ảnh (chỉ đọc, không ghi vào dataset)."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    split: str

    @property
    def filename(self) -> str:
        return self.path.name


def load_split(split: str, data_root: Path = config.DATA_ROOT) -> list:
    """Đọc `<split>.txt` (mỗi dòng: `<tên file> <nhãn>`) thành list Sample."""
    if split not in config.SPLITS:
        raise ValueError(f"split phải thuộc {config.SPLITS}, nhận được {split!r}")
    samples = []
    with open(data_root / f"{split}.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, label = line.rsplit(maxsplit=1)
            samples.append(Sample(data_root / split / filename, int(label), split))
    return samples


def load_all(data_root: Path = config.DATA_ROOT) -> list:
    return [s for split in config.SPLITS for s in load_split(split, data_root)]


def read_image_rgb(path) -> np.ndarray:
    """Đọc ảnh thành RGB uint8 (H, W, 3). Dùng imdecode để an toàn với đường dẫn Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
