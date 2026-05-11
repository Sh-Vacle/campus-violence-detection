from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .video_tools import extract_frames_from_video


def load_labels(class_indices_path: str | Path | None) -> dict[int, str]:
    if class_indices_path is None:
        return {0: "NonViolence", 1: "Violence"}
    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    return {int(index): name for name, index in class_indices.items()}


def preprocess_image(image_path: str | Path, image_size: int = 224) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model_path: str | Path, image_path: str | Path, class_indices_path: str | Path | None = None, image_size: int = 224) -> dict:
    model = tf.keras.models.load_model(model_path)
    labels = load_labels(class_indices_path)
    x = preprocess_image(image_path, image_size=image_size)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    pred_index = 1 if prob >= 0.5 else 0
    return {
        "image": str(image_path),
        "probability_violence": prob,
        "label": labels.get(pred_index, str(pred_index)),
    }


def predict_video(
    model_path: str | Path,
    video_path: str | Path,
    class_indices_path: str | Path | None = None,
    image_size: int = 224,
    fps: float = 1.0,
) -> dict:
    model = tf.keras.models.load_model(model_path)
    labels = load_labels(class_indices_path)

    with tempfile.TemporaryDirectory() as tmp:
        frame_dir = Path(tmp) / "frames"
        n_frames = extract_frames_from_video(video_path, frame_dir, fps=fps, overwrite=True)
        probs = []
        for frame in sorted(frame_dir.glob("*.jpg")):
            x = preprocess_image(frame, image_size=image_size)
            probs.append(float(model.predict(x, verbose=0).ravel()[0]))

    mean_prob = float(np.mean(probs)) if probs else 0.0
    max_prob = float(np.max(probs)) if probs else 0.0
    pred_index = 1 if mean_prob >= 0.5 else 0
    return {
        "video": str(video_path),
        "frames": n_frames,
        "mean_probability_violence": mean_prob,
        "max_probability_violence": max_prob,
        "label": labels.get(pred_index, str(pred_index)),
    }
