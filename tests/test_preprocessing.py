"""Test preprocessing trên ảnh thật của DDR Dataset.

    python tests/test_preprocessing.py          (hoặc: python -m pytest tests -q)
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src import config
from src.augmentation import TrainAugmentation, get_augmentation
from src.dataset import load_all, read_image_rgb
from src.normalization import Normalizer
from src.pipelines import CONFIGS, build_pipeline
from src.preprocessing import apply_clahe, crop_retina, find_retina_bbox, resize_image

N_PER_CLASS_SPLIT = 2   # 2 ảnh x 5 class x 3 split = 30 ảnh thật


def _samples():
    rng = np.random.default_rng(config.SEED)
    all_samples = load_all()
    chosen = []
    for split in config.SPLITS:
        for label in config.CLASS_NAMES:
            pool = [s for s in all_samples if s.split == split and s.label == label]
            chosen += [pool[i] for i in rng.choice(len(pool), size=N_PER_CLASS_SPLIT, replace=False)]
    return chosen


SAMPLES = _samples()
IMAGES = [read_image_rgb(s.path) for s in SAMPLES]
W, H = config.IMG_SIZE


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def test_crop_removes_background_and_keeps_retina():
    n_reduced = 0
    for s, image in zip(SAMPLES, IMAGES):
        before = image.copy()
        bbox = find_retina_bbox(image)
        if bbox is None:
            # ảnh bất thường (vd. 007-7258-400.jpg nền trắng): phải fallback về đúng ảnh gốc
            out = crop_retina(image)
            assert np.array_equal(out, image) and not np.shares_memory(out, image)
            continue
        tight = crop_retina(image, square=False)
        assert np.array_equal(image, before), "crop modified the input array"
        assert tight.shape[0] <= image.shape[0] and tight.shape[1] <= image.shape[1]
        n_reduced += tight.shape[0] * tight.shape[1] < image.shape[0] * image.shape[1]

        # nền đen bị loại: sau khi crop, không còn dải nền đen ở cả hai phía trái/phải
        cols = (tight.max(axis=2) > 3 * config.CROP_THRESHOLD).any(axis=0)
        border = int(0.05 * tight.shape[1])
        assert cols[:border].any() and cols[-border:].any(), f"black border left in {s.filename}"

        # không mất võng mạc: gần như toàn bộ pixel sáng (> threshold) nằm trong bbox
        fg = image.max(axis=2) > 3 * config.CROP_THRESHOLD
        x0, y0, x1, y1 = bbox
        assert fg[y0:y1, x0:x1].sum() / fg.sum() > 0.999, f"retina cut off in {s.filename}"

        square = crop_retina(image, square=True)
        assert square.shape[0] == square.shape[1]
        assert not np.shares_memory(tight, image) and not np.shares_memory(square, image)
    # một số ảnh DDR đã sát khung (võng mạc chạm cả 4 cạnh) nên không còn gì để cắt
    assert n_reduced > len(SAMPLES) // 2, f"crop only reduced {n_reduced}/{len(SAMPLES)} images"


def test_crop_fallback_on_invalid_region():
    black = np.zeros((300, 400, 3), np.uint8)
    out, info = crop_retina(black, return_info=True)
    assert info["cropped"] is False and info["bbox"] is None
    assert np.array_equal(out, black) and not np.shares_memory(out, black)

    tiny = black.copy()
    tiny[140:160, 190:210] = 200          # đốm sáng quá nhỏ -> không coi là võng mạc
    out, info = crop_retina(tiny, return_info=True)
    assert info["cropped"] is False and out.shape == tiny.shape


def test_resize_is_224():
    for image in IMAGES:
        out = resize_image(image)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert (W, H) == (224, 224)


def test_clahe_changes_only_lightness():
    import cv2
    for image in IMAGES[:6]:
        small = resize_image(crop_retina(image), (512, 512))
        before = small.copy()
        out = apply_clahe(small)
        assert np.array_equal(small, before), "CLAHE modified the input array"
        assert out.shape == small.shape and out.dtype == np.uint8
        assert not np.array_equal(out, small), "CLAHE had no effect"

        lab_in = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(int)
        lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(int)
        retina = lab_in[..., 0] > 20
        # tương phản cục bộ kênh L tăng; A/B chỉ lệch do làm tròn / gamut khi đổi LAB <-> RGB uint8
        local_contrast = [np.abs(cv2.Laplacian(x[..., 0].astype(np.float64), cv2.CV_64F))[retina].mean()
                          for x in (lab_in, lab_out)]
        assert local_contrast[1] > local_contrast[0]
        assert np.abs(lab_out[..., 1:] - lab_in[..., 1:])[retina].mean() < 1.5

        stronger = apply_clahe(small, clip_limit=6.0, tile_grid_size=(4, 4))
        assert not np.array_equal(stronger, out), "clip_limit / tile_grid_size have no effect"


def test_normalize_once():
    normalizer = Normalizer()
    image = resize_image(IMAGES[0])
    tensor = normalizer(image)
    assert tensor.shape == (3, H, W) and tensor.dtype == torch.float32

    expected = (image.astype(np.float32) / 255.0 - np.array(config.NORMALIZATION.mean, np.float32)) \
        / np.array(config.NORMALIZATION.std, np.float32)
    assert np.allclose(tensor.permute(1, 2, 0).numpy(), expected, atol=1e-5)
    assert np.abs(normalizer.denormalize(tensor).astype(int) - image).max() <= 1

    for already_normalized in (tensor, tensor.numpy()):
        try:
            normalizer(already_normalized)
        except TypeError:
            continue
        raise AssertionError("normalizing twice must raise")


def test_augmentation_train_only():
    assert isinstance(get_augmentation("train"), TrainAugmentation)
    assert get_augmentation("valid") is None and get_augmentation("test") is None

    image = resize_image(crop_retina(IMAGES[0]))
    before = image.copy()
    augment = TrainAugmentation()
    torch.manual_seed(0)
    outs = [augment(image) for _ in range(8)]
    assert np.array_equal(image, before), "augmentation modified the input array"
    assert all(o.shape == image.shape and o.dtype == np.uint8 for o in outs)
    assert any(not np.array_equal(o, image) for o in outs), "augmentation had no effect"
    assert len({o.tobytes() for o in outs}) > 1, "augmentation is not random"
    # không biến dạng quá mức: độ sáng trung bình chỉ thay đổi nhẹ
    assert all(abs(float(o.mean()) - float(image.mean())) < 0.25 * image.mean() for o in outs)


def test_configs_e1_to_e4():
    assert list(CONFIGS) == ["E1", "E2", "E3", "E4"]
    for name in CONFIGS:
        for split in config.SPLITS:
            pipeline = build_pipeline(name, split=split)
            assert (pipeline.augmentation is not None) == (name == "E4" and split == "train")
            for image in IMAGES[:5]:
                out = pipeline(image)
                assert out.shape == (3, H, W) and out.dtype == torch.float32 and torch.isfinite(out).all()

    image = IMAGES[0]
    raw = {n: build_pipeline(n, split="valid", normalize=False)(image) for n in CONFIGS}
    assert all(o.shape == (H, W, 3) and o.dtype == np.uint8 for o in raw.values())
    assert not np.array_equal(raw["E1"], raw["E2"])
    assert not np.array_equal(raw["E2"], raw["E3"])
    assert np.array_equal(raw["E3"], raw["E4"]), "E4 must equal E3 on valid/test (no augmentation)"

    # valid/test tất định; train E4 ngẫu nhiên
    e4_valid = build_pipeline("E4", split="valid")
    assert torch.equal(e4_valid(image), e4_valid(image))
    e4_train = build_pipeline("E4", split="train")
    torch.manual_seed(0)
    assert len({e4_train(image).numpy().tobytes() for _ in range(6)}) > 1

    stages = build_pipeline("E4", split="train").run_with_stages(image)
    assert list(stages) == ["original", "crop", "clahe", "resize", "augmentation", "normalize", "output"]


def test_dataset_files_untouched():
    """Chạy toàn bộ pipeline không ghi gì vào file ảnh gốc."""
    before = {s.path: (_md5(s.path), s.path.stat().st_mtime_ns) for s in SAMPLES}
    for s in SAMPLES:
        image = read_image_rgb(s.path)
        for name in CONFIGS:
            build_pipeline(name, split="train")(image)
    after = {s.path: (_md5(s.path), s.path.stat().st_mtime_ns) for s in SAMPLES}
    assert before == after


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed on {len(SAMPLES)} real DDR images")
    sys.exit(1 if failed else 0)
