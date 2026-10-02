"""Các bước xử lý ảnh: retinal crop, resize, CLAHE.

Mọi function nhận ảnh RGB uint8 (H, W, 3), trả về mảng MỚI và không sửa ảnh đầu vào.
Normalization và augmentation nằm ở module riêng.
"""
import cv2
import numpy as np

from . import config


def _check_rgb_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Cần ảnh RGB uint8 có shape (H, W, 3)")


# ----------------------------------------------------------------------------- Retinal crop
def foreground_mask(image: np.ndarray, threshold: int = config.CROP_THRESHOLD,
                    work_size: int = config.CROP_WORK_SIZE):
    """Mask vùng võng mạc (grayscale > mức nền + threshold, lấy thành phần liên thông lớn nhất).

    Mức nền = trung vị grayscale ở 4 góc ảnh, vì nền của một số ảnh DDR không đen hẳn (~17).
    Mask được tính trên bản thu nhỏ có cạnh dài <= work_size.
    Trả về (mask bool ở độ phân giải thu nhỏ, scale = kích thước gốc / kích thước thu nhỏ).
    """
    _check_rgb_uint8(image)
    h, w = image.shape[:2]
    scale = max(h, w) / work_size if max(h, w) > work_size else 1.0
    small = image if scale == 1.0 else cv2.resize(
        image, (max(1, round(w / scale)), max(1, round(h / scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    k = max(1, min(gray.shape) // 20)
    background = np.median(np.concatenate([c.ravel() for c in
                                           (gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:])]))
    mask = (gray > background + threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, dtype=bool), scale
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest, scale


def find_retina_bbox(image: np.ndarray, threshold: int = config.CROP_THRESHOLD,
                     margin: float = config.CROP_MARGIN,
                     min_area_ratio: float = config.CROP_MIN_AREA_RATIO,
                     min_side_ratio: float = config.CROP_MIN_SIDE_RATIO):
    """Tìm bounding box (x0, y0, x1, y1) của võng mạc trên ảnh gốc.

    Trả về None nếu không xác định được vùng hợp lệ (foreground quá nhỏ hoặc bbox quá hẹp).
    """
    mask, scale = foreground_mask(image, threshold)
    h, w = image.shape[:2]
    if mask.mean() < min_area_ratio:
        return None

    ys, xs = np.where(mask)
    x0, x1 = xs.min() * scale, (xs.max() + 1) * scale
    y0, y1 = ys.min() * scale, (ys.max() + 1) * scale
    if (x1 - x0) < min_side_ratio * w or (y1 - y0) < min_side_ratio * h:
        return None

    pad = margin * max(x1 - x0, y1 - y0)
    x0, y0 = max(0, int(np.floor(x0 - pad))), max(0, int(np.floor(y0 - pad)))
    x1, y1 = min(w, int(np.ceil(x1 + pad))), min(h, int(np.ceil(y1 + pad)))
    return x0, y0, x1, y1


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pad nền đen đối xứng để ảnh thành hình vuông (giữ nguyên tỉ lệ võng mạc)."""
    h, w = image.shape[:2]
    if h == w:
        return image.copy()
    d = abs(h - w)
    a, b = d // 2, d - d // 2
    top, bottom, left, right = (0, 0, a, b) if h > w else (a, b, 0, 0)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def crop_retina(image: np.ndarray, threshold: int = config.CROP_THRESHOLD,
                margin: float = config.CROP_MARGIN,
                square: bool = config.CROP_PAD_TO_SQUARE,
                return_info: bool = False):
    """Crop vùng võng mạc, bỏ nền đen thừa.

    Nếu không tìm được vùng crop hợp lệ thì fallback: trả về bản copy của ảnh gốc.
    `square=True` pad nền đen cho ảnh crop thành hình vuông (chỉ khi crop thành công).
    `return_info=True` trả thêm dict {"bbox": (x0, y0, x1, y1) | None, "cropped": bool}.
    """
    bbox = find_retina_bbox(image, threshold, margin)
    if bbox is None:
        out = image.copy()
    else:
        x0, y0, x1, y1 = bbox
        out = image[y0:y1, x0:x1]
        out = pad_to_square(out) if square else out.copy()
    if return_info:
        return out, {"bbox": bbox, "cropped": bbox is not None}
    return out


# ----------------------------------------------------------------------------- Resize
def resize_image(image: np.ndarray, size: tuple = config.IMG_SIZE) -> np.ndarray:
    """Resize ảnh về `size` = (width, height), mặc định 224x224."""
    _check_rgb_uint8(image)
    w, h = size
    shrinking = image.shape[0] > h or image.shape[1] > w
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interpolation)


# ----------------------------------------------------------------------------- CLAHE
def apply_clahe(image: np.ndarray, clip_limit: float = config.CLAHE_CLIP_LIMIT,
                tile_grid_size: tuple = config.CLAHE_TILE_GRID_SIZE) -> np.ndarray:
    """CLAHE trên kênh L của không gian LAB; kênh A, B giữ nguyên."""
    _check_rgb_uint8(image)
    l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
