"""Cấu hình chung của project: đường dẫn, tham số preprocessing, normalization, augmentation.

Mọi tham số có thể điều chỉnh đều nằm ở đây để các module khác không hard-code.
"""
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------- Đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "DDR-dataset" / "DR_grading"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
PREPROCESS_DIR = OUTPUT_DIR / "preprocessing"

SPLITS = ("train", "valid", "test")

# ----------------------------------------------------------------------------- Nhãn
CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR",
}
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42

# ----------------------------------------------------------------------------- Retinal crop
CROP_THRESHOLD = 10          # võng mạc = grayscale > mức nền (đo ở 4 góc ảnh) + ngưỡng này
CROP_MARGIN = 0.01           # lề thêm quanh bbox, tính theo cạnh dài của bbox
CROP_MIN_AREA_RATIO = 0.10   # foreground < 10% diện tích ảnh -> coi như không hợp lệ
CROP_MIN_SIDE_RATIO = 0.25   # bbox hẹp hơn 25% cạnh ảnh gốc -> coi như không hợp lệ
CROP_PAD_TO_SQUARE = True    # pad nền đen cho vuông để resize không làm méo võng mạc
CROP_WORK_SIZE = 512         # mask được tính trên bản thu nhỏ (cạnh dài) cho nhanh

# ----------------------------------------------------------------------------- Resize
IMG_SIZE = (224, 224)        # (width, height)

# ----------------------------------------------------------------------------- CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# ----------------------------------------------------------------------------- Normalization
@dataclass(frozen=True)
class NormalizationConfig:
    """Mean/std theo kênh RGB, áp dụng trên ảnh đã scale về [0, 1]."""
    mean: tuple = (0.485, 0.456, 0.406)   # ImageNet (phù hợp backbone pretrained)
    std: tuple = (0.229, 0.224, 0.225)


NORMALIZATION = NormalizationConfig()


# ----------------------------------------------------------------------------- Augmentation
@dataclass(frozen=True)
class AugmentationConfig:
    """Tham số augmentation cho training. Giữ ở mức nhẹ để không làm méo võng mạc."""
    hflip_prob: float = 0.5
    rotation_degrees: float = 15.0        # xoay trong [-15, +15] độ
    zoom_range: tuple = (0.9, 1.1)        # hệ số zoom
    brightness: float = 0.1               # hệ số sáng trong [0.9, 1.1]
    contrast: float = 0.1                 # hệ số tương phản trong [0.9, 1.1]


AUGMENTATION = AugmentationConfig()

# ----------------------------------------------------------------------------- EDA
EDA_ANALYSIS_SIZE = 512      # brightness/contrast/sharpness đo trên ảnh thu nhỏ (cạnh dài)
QUALITY_IQR_K = 1.5          # hệ số Tukey fence dùng để gắn cờ ảnh chất lượng thấp
