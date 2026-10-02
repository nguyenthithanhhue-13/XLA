"""Normalization, tách riêng khỏi các bước xử lý ảnh.

Mean/std cấu hình tại `config.NORMALIZATION`. `Normalizer` chỉ nhận ảnh uint8, nên
gọi normalize lần thứ hai trên kết quả (float) sẽ báo lỗi thay vì normalize hai lần.
"""
import numpy as np
import torch

from . import config


class Normalizer:
    """uint8 RGB (H, W, 3) -> float32 tensor (3, H, W) đã chuẩn hoá theo mean/std."""

    def __init__(self, cfg: config.NormalizationConfig = config.NORMALIZATION):
        self.cfg = cfg
        self._mean = torch.tensor(cfg.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(cfg.std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise TypeError("Normalizer chỉ nhận ảnh uint8 - ảnh này có thể đã được normalize.")
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8:
                raise TypeError("Normalizer chỉ nhận tensor uint8 - tensor này có thể đã được normalize.")
            tensor = image
        else:
            raise TypeError(f"Kiểu dữ liệu không hỗ trợ: {type(image)}")
        return (tensor.float() / 255.0 - self._mean) / self._std

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Đảo normalization về ảnh RGB uint8 (H, W, 3) - dùng cho visualization."""
        image = (tensor.detach().cpu() * self._std + self._mean).clamp(0, 1)
        return (image * 255.0).round().byte().permute(1, 2, 0).numpy()

    def __repr__(self) -> str:
        return f"Normalizer(mean={self.cfg.mean}, std={self.cfg.std})"
