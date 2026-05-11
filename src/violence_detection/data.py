from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_generators(
    data_dir: str | Path,
    image_size: int = 224,
    batch_train: int = 8,
    batch_val: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
    )

    target_size = (image_size, image_size)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_train,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=seed,
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_val,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    return train_generator, validation_generator


def compute_class_weight(classes) -> dict[int, float]:
    counts = np.bincount(classes)
    total = counts.sum()
    weights: dict[int, float] = {}
    for idx, count in enumerate(counts):
        weights[idx] = float(total) / (len(counts) * count) if count > 0 else 1.0
    return weights


def count_images(data_dir: str | Path) -> dict[str, int]:
    data_dir = Path(data_dir)
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result = {}
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        result[class_dir.name] = sum(1 for p in class_dir.rglob("*") if p.suffix.lower() in image_ext)
    return result
