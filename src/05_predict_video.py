"""
05_predict_video.py

作用：
    使用训练好的 Keras 模型，对单个视频进行暴力/非暴力预测。

默认逻辑：
    1. 临时抽取视频帧；
    2. 对每一帧进行模型预测；
    3. 按指定策略汇总帧概率；
    4. score >= threshold 判定为 Violence，否则判定为 NonViolence。

运行示例：
    python src/05_predict_video.py --video ".\\dataset\\Violence\\V_100.mp4" --model ".\\models\\best_model_finetuned.h5"
    python src/05_predict_video.py --video ".\\dataset\\NonViolence\\NV_100.mp4" --model ".\\models\\best_model_finetuned.h5" --fps 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from ffmpeg_utils import find_ffmpeg, print_ffmpeg_help, run_subprocess


def configure_tensorflow() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def extract_temp_frames(video_path: Path, temp_dir: Path, ffmpeg_cmd: str, use_shell: bool, fps: float) -> list[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = temp_dir / 'frame_%06d.jpg'

    command = [
        ffmpeg_cmd,
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-q:v', '2',
        str(output_pattern),
    ]

    run_subprocess(command, use_shell=use_shell)

    frames = sorted(temp_dir.glob('frame_*.jpg'))
    if not frames:
        raise RuntimeError('视频抽帧失败，没有生成任何图片。')
    return frames


def load_frames_as_array(frame_paths: list[Path], img_size: tuple[int, int]) -> np.ndarray:
    images = []
    for frame_path in frame_paths:
        img = load_img(frame_path, target_size=img_size)
        arr = img_to_array(img) / 255.0
        images.append(arr)
    return np.asarray(images, dtype=np.float32)


def aggregate_probabilities(probabilities: np.ndarray, strategy: str, top_ratio: float) -> float:
    if probabilities.size == 0:
        return 0.0

    if strategy == 'avg':
        return float(np.mean(probabilities))

    if strategy == 'max':
        return float(np.max(probabilities))

    if strategy == 'topk':
        top_ratio = min(max(top_ratio, 0.0), 1.0)
        k = max(1, int(np.ceil(len(probabilities) * top_ratio)))
        top_values = np.sort(probabilities)[-k:]
        return float(np.mean(top_values))

    raise ValueError(f'未知策略：{strategy}')


def predict_video(
    model_path: Path,
    video_path: Path,
    fps: float,
    img_size: tuple[int, int],
    threshold: float,
    strategy: str,
    top_ratio: float,
    keep_frames: bool = False,
) -> None:
    ffmpeg_cmd, use_shell = find_ffmpeg()
    if ffmpeg_cmd is None:
        print_ffmpeg_help()
        return

    if not model_path.exists():
        raise FileNotFoundError(f'模型文件不存在：{model_path}')
    if not video_path.exists():
        raise FileNotFoundError(f'视频文件不存在：{video_path}')

    print(f'[信息] 使用 FFmpeg：{ffmpeg_cmd}')
    print(f'[信息] 视频文件：{video_path}')
    print(f'[信息] 模型文件：{model_path}')
    print(f'[信息] 抽帧 fps：{fps}')
    print(f'[信息] 汇总策略：{strategy}')
    print(f'[信息] 判定阈值：{threshold}')
    print()

    model = load_model(model_path)
    temp_root = Path(tempfile.mkdtemp(prefix='predict_frames_'))

    try:
        frames = extract_temp_frames(video_path, temp_root, ffmpeg_cmd, use_shell, fps)
        x = load_frames_as_array(frames, img_size)

        y_prob = model.predict(x, batch_size=8, verbose=1).ravel()
        avg_prob = float(np.mean(y_prob))
        max_prob = float(np.max(y_prob))
        min_prob = float(np.min(y_prob))
        score = aggregate_probabilities(y_prob, strategy=strategy, top_ratio=top_ratio)

        pred_label = 'Violence' if score >= threshold else 'NonViolence'

        print()
        print('========== 预测结果 ==========')
        print(f'视频：{video_path}')
        print(f'抽帧数量：{len(frames)}')
        print(f'平均暴力概率：{avg_prob:.4f}')
        print(f'最高暴力概率：{max_prob:.4f}')
        print(f'最低暴力概率：{min_prob:.4f}')
        print(f'最终判定分数：{score:.4f}')
        print(f'预测类别：{pred_label}')

    finally:
        if keep_frames:
            print(f'[信息] 临时帧保留在：{temp_root}')
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    configure_tensorflow()

    parser = argparse.ArgumentParser(description='单视频暴力识别预测')
    parser.add_argument('--video', required=True, help='待预测视频路径')
    parser.add_argument('--model', default='models/best_model_finetuned.h5', help='Keras .h5 模型路径')
    parser.add_argument('--fps', type=float, default=1.0, help='每秒抽几帧，默认 1')
    parser.add_argument('--img-size', type=int, default=224, help='模型输入图片尺寸')
    parser.add_argument('--threshold', type=float, default=0.5, help='暴力类别判定阈值')
    parser.add_argument('--strategy', choices=['avg', 'max', 'topk'], default='avg', help='视频级概率汇总策略')
    parser.add_argument('--top-ratio', type=float, default=0.3, help='strategy=topk 时使用概率最高的前多少比例帧')
    parser.add_argument('--keep-frames', action='store_true', help='保留临时抽帧结果，方便检查')
    args = parser.parse_args()

    predict_video(
        model_path=Path(args.model).resolve(),
        video_path=Path(args.video).resolve(),
        fps=args.fps,
        img_size=(args.img_size, args.img_size),
        threshold=args.threshold,
        strategy=args.strategy,
        top_ratio=args.top_ratio,
        keep_frames=args.keep_frames,
    )


if __name__ == '__main__':
    main()
