"""Augmentation pipeline - chỉ dùng cho training data.

Gồm: horizontal flip, rotation, zoom, brightness, contrast (tham số ở `config.AUGMENTATION`).
Làm việc trên ảnh uint8, đặt TRƯỚC normalization.
"""
import numpy as np
import torch
from torchvision.transforms import v2

from . import config


class TrainAugmentation:
    """Ảnh RGB uint8 (H, W, 3) -> ảnh RGB uint8 (H, W, 3) đã augment, cùng kích thước."""

    def __init__(self, cfg: config.AugmentationConfig = config.AUGMENTATION):
        self.cfg = cfg
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=cfg.hflip_prob),
            # rotation + zoom gộp trong một phép affine (nội suy một lần), viền lấp nền đen
            v2.RandomAffine(degrees=cfg.rotation_degrees, scale=cfg.zoom_range,
                            interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
        ])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise TypeError("Augmentation cần ảnh RGB uint8 (chưa normalize)")
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        return self.transform(tensor).permute(1, 2, 0).contiguous().numpy()


def get_augmentation(split: str, cfg: config.AugmentationConfig = config.AUGMENTATION):
    """Trả về augmentation cho split "train"; valid/test trả về None (không augment)."""
    if split not in config.SPLITS:
        raise ValueError(f"split phải thuộc {config.SPLITS}, nhận được {split!r}")
    return TrainAugmentation(cfg) if split == "train" else None
