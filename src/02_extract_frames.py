"""
02_extract_frames.py

作用：
    从视频中抽帧，生成适合 Keras flow_from_directory 读取的图片目录。

默认输入：
    优先使用 fixed_videos；如果 fixed_videos 没有视频，会自动使用 dataset。

默认输出：
    frames/NonViolence
    frames/Violence

运行示例：
    python src/02_extract_frames.py
    python src/02_extract_frames.py --fps 1
    python src/02_extract_frames.py --input fixed_videos --output frames --fps 2
    python src/02_extract_frames.py --fps 1 --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ffmpeg_utils import find_ffmpeg, print_ffmpeg_help, run_subprocess

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
CATEGORIES = ('NonViolence', 'Violence')


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def has_videos(path: Path) -> bool:
    return path.exists() and any(
        p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        for p in path.rglob('*')
    )


def extract_frames(
    video_path: Path,
    output_dir: Path,
    ffmpeg_cmd: str,
    use_shell: bool,
    fps: float = 1.0,
    overwrite: bool = False,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob('frame_*.jpg'))
    if existing and not overwrite:
        print(f'[跳过] 已抽帧：{output_dir}')
        return True

    if overwrite:
        for img in existing:
            img.unlink(missing_ok=True)

    output_pattern = output_dir / 'frame_%06d.jpg'

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

    try:
        run_subprocess(command, use_shell=use_shell)
        count = len(list(output_dir.glob('frame_*.jpg')))

        if count > 0:
            print(f'[完成] {video_path.name} -> {count} 张图片')
            return True

        print(f'[失败] {video_path.name}：FFmpeg 执行后没有生成图片')
        return False

    except subprocess.CalledProcessError as exc:
        print(f'[失败] {video_path}：FFmpeg 执行失败，错误码 {exc.returncode}')
        return False
    except Exception as exc:
        print(f'[失败] {video_path}：{exc}')
        return False


def main() -> None:
    root = project_root()

    default_input = root / 'fixed_videos'
    if not has_videos(default_input):
        default_input = root / 'dataset'

    parser = argparse.ArgumentParser(description='视频抽帧')
    parser.add_argument('--input', default=str(default_input), help='视频输入根目录')
    parser.add_argument('--output', default=str(root / 'frames'), help='帧图片输出目录')
    parser.add_argument('--fps', type=float, default=1.0, help='每秒抽几帧，默认 1')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在帧图片')
    args = parser.parse_args()

    ffmpeg_cmd, use_shell = find_ffmpeg()
    if ffmpeg_cmd is None:
        print_ffmpeg_help()
        return

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    print(f'[信息] 使用 FFmpeg：{ffmpeg_cmd}')
    print(f'[信息] 输入目录：{input_root}')
    print(f'[信息] 输出目录：{output_root}')
    print(f'[信息] 抽帧 fps：{args.fps}')

    total = 0
    success = 0

    for category in CATEGORIES:
        category_dir = input_root / category
        if not category_dir.exists():
            print(f'[提示] 类别目录不存在，跳过：{category_dir}')
            continue

        for video_path in category_dir.rglob('*'):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                relative = video_path.relative_to(category_dir)
                video_stem_dir = output_root / category / relative.with_suffix('')
                total += 1
                if extract_frames(video_path, video_stem_dir, ffmpeg_cmd, use_shell, fps=args.fps, overwrite=args.overwrite):
                    success += 1

    print(f'抽帧完成：成功 {success}/{total} 个视频。')
    if total == 0:
        print('没有找到视频。检查 dataset 或 fixed_videos 目录结构。')


if __name__ == '__main__':
    main()
