"""
03_check_dataset.py

作用：检查 frames 数据集能否被 TensorFlow/Keras 正常读取。
运行示例：
    python src/03_check_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description='检查图像数据集')
    parser.add_argument('--data', default=str(root / 'frames'), help='frames 数据目录')
    parser.add_argument('--img-size', type=int, default=224, help='输入图片尺寸')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f'数据目录不存在：{data_dir}')

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42,
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42,
    )

    print(f'训练集图片数：{train_generator.samples}')
    print(f'验证集图片数：{val_generator.samples}')
    print(f'类别映射：{train_generator.class_indices}')

    if train_generator.samples == 0 or val_generator.samples == 0:
        print('警告：训练集或验证集为空。需要先运行抽帧或检查 frames 目录结构。')


if __name__ == '__main__':
    main()
