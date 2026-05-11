"""
04_train_model.py

作用：
    训练校园暴力二分类模型。

默认模型：
    MobileNetV2 + 自定义分类头。

默认输入：
    frames/NonViolence
    frames/Violence

默认输出：
    models/best_model_frozen.h5
    models/best_model_finetuned.h5
    models/best_model_fp16.tflite

运行示例：
    python src/04_train_model.py
    python src/04_train_model.py --batch-train 2 --batch-val 8
    python src/04_train_model.py --epochs-frozen 10 --epochs-finetune 8
    python src/04_train_model.py --no-finetune
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path

# 降低 TensorFlow 启动日志噪声。需要调试 CUDA 时可注释掉。
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_tensorflow() -> None:
    """
    配置 TensorFlow。

    重点：开启 GPU 显存按需增长，避免 3GB~4GB 可用显存的显卡在 Windows 上一启动就 OOM。
    """
    try:
        tf.config.optimizer.set_jit(False)
    except Exception as exc:
        print(f'[警告] 关闭 XLA JIT 失败：{exc}')

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('[警告] TensorFlow 没有检测到 GPU，将使用 CPU 训练。')
        return

    print('[信息] 检测到 GPU：')
    for gpu in gpus:
        print(f'    {gpu}')

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f'[信息] 已开启 memory growth：{gpu}')
        except Exception as exc:
            print(f'[警告] 设置 GPU memory growth 失败：{exc}')

    try:
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'[信息] 逻辑 GPU 数量：{len(logical_gpus)}')
    except Exception as exc:
        print(f'[警告] 获取逻辑 GPU 信息失败：{exc}')


def cleanup_memory() -> None:
    gc.collect()


def build_generators(data_dir: Path, img_size: tuple[int, int], batch_train: int, batch_val: int):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_train,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=SEED,
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_val,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED,
    )

    return train_generator, validation_generator


def compute_class_weight(classes: np.ndarray) -> dict[int, float]:
    counts = np.bincount(classes, minlength=2)
    total = counts.sum()

    return {
        0: float(total) / (2.0 * counts[0]) if counts[0] > 0 else 1.0,
        1: float(total) / (2.0 * counts[1]) if counts[1] > 0 else 1.0,
    }


def build_model(img_size: tuple[int, int], dense_units: int = 512) -> tf.keras.Model:
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=img_size + (3,),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)


def find_mobilenet_base(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenetv2'):
            return layer
    return None


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )


def evaluate_and_report(model: tf.keras.Model, validation_generator, batch_val: int) -> None:
    validation_generator.reset()
    validation_steps = max(1, math.ceil(validation_generator.samples / batch_val))

    results = model.evaluate(validation_generator, steps=validation_steps, verbose=1)
    print('[Eval]', dict(zip(model.metrics_names, results)))

    validation_generator.reset()
    y_prob = model.predict(validation_generator, steps=validation_steps, verbose=1).ravel()
    y_true = validation_generator.classes[:len(y_prob)]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
    print('\nConfusion Matrix:\n', cm)

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(y_true, y_prob)
        auc_value = float(auc_metric.result().numpy())

        print(f'Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  AUC={auc_value:.4f}')


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        print(f'TFLite FP16 模型已导出：{output_path}')
    except Exception as exc:
        print(f'TFLite 导出失败：{exc}')


def main() -> None:
    configure_tensorflow()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    root = project_root()

    parser = argparse.ArgumentParser(description='训练校园暴力识别模型')
    parser.add_argument('--data', default=str(root / 'frames'), help='frames 数据目录')
    parser.add_argument('--models-dir', default=str(root / 'models'), help='模型输出目录')
    parser.add_argument('--logs-dir', default=str(root / 'logs'), help='TensorBoard 日志目录')
    parser.add_argument('--img-size', type=int, default=224, help='输入图片尺寸')

    # 低显存稳定默认值。RTX 3060 Laptop 在 TensorFlow 2.10 下采用 2 / 8 作为低显存默认值。
    parser.add_argument('--batch-train', type=int, default=2, help='训练 batch size，显存不够就调小')
    parser.add_argument('--batch-val', type=int, default=8, help='验证 batch size，显存不够就调小')

    parser.add_argument('--epochs-frozen', type=int, default=10, help='冻结特征层训练轮数')
    parser.add_argument('--epochs-finetune', type=int, default=8, help='微调轮数')
    parser.add_argument('--unfreeze-last', type=int, default=20, help='微调时解冻 MobileNetV2 最后多少层')
    parser.add_argument('--dense-units', type=int, default=512, help='分类头 Dense 层神经元数量')
    parser.add_argument('--no-finetune', action='store_true', help='只训练冻结阶段，跳过微调阶段')
    parser.add_argument('--no-tflite', action='store_true', help='跳过 TFLite 导出')
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    models_dir = Path(args.models_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_size, args.img_size)

    print('[信息] 数据目录：', data_dir)
    print('[信息] 模型输出目录：', models_dir)
    print('[信息] 日志目录：', logs_dir)
    print('[信息] 图片尺寸：', img_size)
    print('[信息] batch_train：', args.batch_train)
    print('[信息] batch_val：', args.batch_val)

    train_generator, validation_generator = build_generators(
        data_dir=data_dir,
        img_size=img_size,
        batch_train=args.batch_train,
        batch_val=args.batch_val,
    )

    print('类别映射：', train_generator.class_indices)

    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise RuntimeError('训练集或验证集为空。需要先运行 02_extract_frames.py，并确认 frames 目录结构。')

    class_weight = compute_class_weight(train_generator.classes)
    print('Class weight:', class_weight)

    steps_per_epoch = max(1, train_generator.samples // args.batch_train)
    validation_steps = max(1, validation_generator.samples // args.batch_val)
    print('steps_per_epoch:', steps_per_epoch)
    print('validation_steps:', validation_steps)

    best_frozen = models_dir / 'best_model_frozen.h5'
    best_finetuned = models_dir / 'best_model_finetuned.h5'

    cleanup_memory()

    model = build_model(img_size=img_size, dense_units=args.dense_units)
    compile_model(model, learning_rate=1e-3)
    model.summary()

    callbacks_common = [
        TensorBoard(log_dir=str(logs_dir)),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    print('\n========== 阶段一：冻结特征层训练 ==========')
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs_frozen,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=callbacks_common + [
            ModelCheckpoint(str(best_frozen), monitor='val_loss', save_best_only=True, verbose=1)
        ],
    )

    if best_frozen.exists():
        model = load_model(best_frozen)
        print('已载入冻结阶段最佳模型：', best_frozen)

    cleanup_memory()

    if args.no_finetune:
        print('\n[信息] 已选择 --no-finetune，跳过阶段二微调。')
        final_path = best_frozen
    else:
        print('\n========== 阶段二：部分解冻微调 ==========')
        base_model = find_mobilenet_base(model)

        if base_model is not None:
            unfreeze_from = max(0, len(base_model.layers) - args.unfreeze_last)
            for i, layer in enumerate(base_model.layers):
                layer.trainable = i >= unfreeze_from and not isinstance(layer, layers.BatchNormalization)
            print(f'已解冻 MobileNetV2 最后 {len(base_model.layers) - unfreeze_from} 层，BatchNorm 保持冻结。')
        else:
            print('未找到 MobileNetV2 base_model，跳过解冻。')

        compile_model(model, learning_rate=1e-5)
        model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs_finetune,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks_common + [
                ModelCheckpoint(str(best_finetuned), monitor='val_loss', save_best_only=True, verbose=1)
            ],
        )

        final_path = best_finetuned if best_finetuned.exists() else best_frozen

    if not final_path.exists():
        raise RuntimeError('没有找到最终模型文件，训练可能没有成功保存模型。')

    model = load_model(final_path)
    print('最终载入模型：', final_path)

    cleanup_memory()
    evaluate_and_report(model=model, validation_generator=validation_generator, batch_val=args.batch_val)

    if args.no_tflite:
        print('[信息] 已选择 --no-tflite，跳过 TFLite 导出。')
    else:
        export_tflite(model=model, output_path=models_dir / 'best_model_fp16.tflite')


if __name__ == '__main__':
    main()
