"""4 cấu hình preprocessing E1-E4, mỗi cấu hình gọi độc lập qua `build_pipeline`.

    E1: Resize
    E2: Crop + Resize
    E3: Crop + CLAHE + Resize
    E4: Crop + CLAHE + Resize + Augmentation (augmentation chỉ khi split == "train")

Thứ tự thực thi: Crop -> CLAHE -> Resize -> Augmentation (train) -> Normalize.
Ảnh giữ dạng uint8 đến bước Normalize cuối cùng, nên chỉ normalize đúng một lần.

Ví dụ tích hợp vào Dataset sau này:

    pipeline = build_pipeline("E4", split="train")
    tensor = pipeline(read_image_rgb(path))        # float32 (3, 224, 224)
"""
from dataclasses import dataclass

import numpy as np

from . import config
from .augmentation import get_augmentation
from .normalization import Normalizer
from .preprocessing import apply_clahe, crop_retina, resize_image


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    crop: bool
    clahe: bool
    augment: bool
    description: str


CONFIGS = {
    "E1": PreprocessConfig("E1", crop=False, clahe=False, augment=False, description="Resize"),
    "E2": PreprocessConfig("E2", crop=True, clahe=False, augment=False, description="Crop + Resize"),
    "E3": PreprocessConfig("E3", crop=True, clahe=True, augment=False, description="Crop + CLAHE + Resize"),
    "E4": PreprocessConfig("E4", crop=True, clahe=True, augment=True, description="Crop + CLAHE + Augmentation"),
}


class PreprocessPipeline:
    """Pipeline cho một cấu hình + một split.

    `normalize=False` trả về ảnh uint8 (H, W, 3) thay vì tensor - tiện cho visualization.
    """

    def __init__(self, cfg: PreprocessConfig, split: str = "train", normalize: bool = True,
                 img_size: tuple = config.IMG_SIZE,
                 clip_limit: float = config.CLAHE_CLIP_LIMIT,
                 tile_grid_size: tuple = config.CLAHE_TILE_GRID_SIZE):
        self.cfg = cfg
        self.split = split
        self.img_size = img_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.augmentation = get_augmentation(split) if cfg.augment else None
        self.normalizer = Normalizer() if normalize else None

    def run_with_stages(self, image: np.ndarray) -> dict:
        """Chạy pipeline và trả về kết quả từng bước (dict theo thứ tự thực thi)."""
        stages = {"original": image}
        x = image
        if self.cfg.crop:
            x = stages["crop"] = crop_retina(x)
        if self.cfg.clahe:
            x = stages["clahe"] = apply_clahe(x, self.clip_limit, self.tile_grid_size)
        x = stages["resize"] = resize_image(x, self.img_size)
        if self.augmentation is not None:
            x = stages["augmentation"] = self.augmentation(x)
        if self.normalizer is not None:
            x = stages["normalize"] = self.normalizer(x)
        stages["output"] = x
        return stages

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Phần tất định (Crop/CLAHE/Resize) -> ảnh uint8 224x224; có thể cache trước khi train."""
        x = image
        if self.cfg.crop:
            x = crop_retina(x)
        if self.cfg.clahe:
            x = apply_clahe(x, self.clip_limit, self.tile_grid_size)
        return resize_image(x, self.img_size)

    def __call__(self, image: np.ndarray):
        x = self.process_image(image)
        if self.augmentation is not None:
            x = self.augmentation(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x

    def __repr__(self) -> str:
        return (f"PreprocessPipeline({self.cfg.name}: {self.cfg.description}, split={self.split!r}, "
                f"augment={self.augmentation is not None}, normalize={self.normalizer is not None})")


def build_pipeline(name: str, split: str = "train", normalize: bool = True, **kwargs) -> PreprocessPipeline:
    """Tạo pipeline theo tên cấu hình ("E1".."E4") và split ("train"/"valid"/"test")."""
    if name not in CONFIGS:
        raise ValueError(f"Cấu hình không tồn tại: {name!r}. Chọn một trong {list(CONFIGS)}")
    return PreprocessPipeline(CONFIGS[name], split=split, normalize=normalize, **kwargs)
