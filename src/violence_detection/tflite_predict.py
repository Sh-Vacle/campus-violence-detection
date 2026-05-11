from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # fallback for desktop TensorFlow
    from tensorflow.lite.python.interpreter import Interpreter


def load_labels(class_indices_path: str | Path | None) -> dict[int, str]:
    if class_indices_path is None:
        return {0: "NonViolence", 1: "Violence"}
    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    return {int(index): name for name, index in class_indices.items()}


def preprocess_image(image_path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_tflite(model_path: str | Path, image_path: str | Path, class_indices_path: str | Path | None = None) -> dict:
    labels = load_labels(class_indices_path)
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    image_size = int(input_details["shape"][1])

    x = preprocess_image(image_path, image_size=image_size).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()

    prob = float(interpreter.get_tensor(output_details["index"]).ravel()[0])
    pred_index = 1 if prob >= 0.5 else 0
    return {
        "image": str(image_path),
        "probability_violence": prob,
        "label": labels.get(pred_index, str(pred_index)),
    }
