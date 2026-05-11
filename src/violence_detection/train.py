from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model

from .data import compute_class_weight, make_generators
from .model import build_model, compile_model, unfreeze_last_layers
from .paths import ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def export_tflite_fp16(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output_path.write_bytes(converter.convert())


def append_history_rows(rows: list[dict], history, phase: str) -> None:
    if history is None:
        return

    epochs = len(history.history.get("loss", []))
    start_epoch = len(rows) + 1
    for i in range(epochs):
        row = {"epoch": start_epoch + i, "phase": phase}
        for key, values in history.history.items():
            value = values[i]
            row[key] = float(value) if isinstance(value, (float, int, np.floating)) else value
        rows.append(row)


def save_history_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = ["epoch", "phase"]
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_plots(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"skip plots: {exc}")
        return

    epochs = [row["epoch"] for row in rows]

    def plot_metric(metric: str, val_metric: str, filename: str, ylabel: str) -> None:
        has_metric = any(metric in row for row in rows)
        has_val_metric = any(val_metric in row for row in rows)
        if not has_metric and not has_val_metric:
            return

        plt.figure(figsize=(8, 5))
        if has_metric:
            plt.plot(epochs, [row.get(metric, np.nan) for row in rows], marker="o", label=metric)
        if has_val_metric:
            plt.plot(epochs, [row.get(val_metric, np.nan) for row in rows], marker="o", label=val_metric)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()

    plot_metric("loss", "val_loss", "loss_curve.png", "Loss")
    plot_metric("accuracy", "val_accuracy", "accuracy_curve.png", "Accuracy")
    plot_metric("auc", "val_auc", "auc_curve.png", "AUC")


def save_confusion_matrix_plot(cm: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"skip confusion matrix plot: {exc}")
        return

    labels = ["NonViolence", "Violence"]
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def evaluate_model(model: tf.keras.Model, generator, batch_size: int, out_dir: Path | None = None) -> dict:
    steps = max(1, math.ceil(generator.samples / batch_size))
    values = model.evaluate(generator, steps=steps, verbose=1)
    metric_names = model.metrics_names
    result = {name: float(value) for name, value in zip(metric_names, values)}

    generator.reset()
    y_prob = model.predict(generator, steps=steps, verbose=1).ravel()
    y_true = generator.classes[: len(y_prob)]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, y_prob)

    result.update(
        {
            "confusion_matrix": cm.tolist(),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_from_predictions": float(auc_metric.result().numpy()),
        }
    )

    if out_dir is not None:
        save_confusion_matrix_plot(cm, out_dir)

    return result


def run_training(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    log_dir = ensure_dir(out_dir / "logs")

    train_gen, val_gen = make_generators(
        data_dir=args.data,
        image_size=args.image_size,
        batch_train=args.batch_train,
        batch_val=args.batch_val,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    with open(out_dir / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)

    class_weight = compute_class_weight(train_gen.classes) if args.use_class_weight else None
    if class_weight:
        print("class_weight:", class_weight)

    steps_per_epoch = max(1, train_gen.samples // args.batch_train)
    validation_steps = max(1, val_gen.samples // args.batch_val)

    best_frozen = out_dir / "best_model_frozen.h5"
    best_finetuned = out_dir / "best_model_finetuned.h5"

    callbacks = [
        TensorBoard(log_dir=str(log_dir)),
        EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    model = build_model(image_size=args.image_size, dense_units=args.dense_units, dropout=args.dropout)
    compile_model(model, args.lr_frozen)
    model.summary()

    history_rows: list[dict] = []

    if args.frozen_epochs > 0:
        history_frozen = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=args.frozen_epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks + [ModelCheckpoint(str(best_frozen), monitor="val_loss", save_best_only=True, verbose=1)],
        )
        append_history_rows(history_rows, history_frozen, "frozen")

    if best_frozen.exists():
        model = load_model(best_frozen)

    if args.finetune_epochs > 0:
        unfrozen = unfreeze_last_layers(model, args.unfreeze_layers)
        print(f"unfrozen layers: {unfrozen}")
        compile_model(model, args.lr_finetune)
        history_finetune = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=args.finetune_epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks + [ModelCheckpoint(str(best_finetuned), monitor="val_loss", save_best_only=True, verbose=1)],
        )
        append_history_rows(history_rows, history_finetune, "finetune")

    save_history_csv(history_rows, out_dir / "history.csv")
    if not args.no_plots:
        save_training_plots(history_rows, out_dir)

    final_path = best_finetuned if best_finetuned.exists() else best_frozen
    if not final_path.exists():
        final_path = out_dir / "last_model.h5"
        model.save(final_path)

    model = load_model(final_path)
    metrics = evaluate_model(model, val_gen, args.batch_val, out_dir=out_dir if not args.no_plots else None)
    metrics["final_model"] = str(final_path)
    metrics["history_csv"] = str(out_dir / "history.csv")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.export_tflite:
        export_tflite_fp16(model, out_dir / "best_model_fp16.tflite")

    print("final model:", final_path)
    print("metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a campus violence detection model.")
    parser.add_argument("--data", required=True, help="抽帧后的数据目录，例如 data/frames")
    parser.add_argument("--out", default="runs/mobilenetv2", help="输出目录")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-train", type=int, default=8)
    parser.add_argument("--batch-val", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--dense-units", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--frozen-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    parser.add_argument("--lr-frozen", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="不导出训练曲线和混淆矩阵图片")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
