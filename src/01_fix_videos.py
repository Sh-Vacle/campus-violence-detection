"""
01_fix_videos.py

作用：
    用 FFmpeg 重新封装/修复原始视频，减少后续抽帧时读不出来的问题。

默认输入：
    dataset/NonViolence
    dataset/Violence

默认输出：
    fixed_videos/NonViolence
    fixed_videos/Violence

运行示例：
    python src/01_fix_videos.py
    python src/01_fix_videos.py --input dataset --output fixed_videos --overwrite
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


def fix_video(input_path: Path, output_path: Path, ffmpeg_cmd: str, use_shell: bool, overwrite: bool = False) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f'[跳过] 已存在：{output_path}')
        return True

    command = [
        ffmpeg_cmd,
        '-hide_banner',
        '-loglevel', 'error',
        '-y' if overwrite else '-n',
        '-i', str(input_path),
        '-c', 'copy',
        '-map', '0',
        str(output_path),
    ]

    try:
        run_subprocess(command, use_shell=use_shell)
        print(f'[完成] {input_path.name} -> {output_path}')
        return True
    except subprocess.CalledProcessError as exc:
        print(f'[失败] {input_path}：FFmpeg 执行失败，错误码 {exc.returncode}')
        return False
    except Exception as exc:
        print(f'[失败] {input_path}：{exc}')
        return False


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description='修复/重新封装视频文件')
    parser.add_argument('--input', default=str(root / 'dataset'), help='原始视频根目录')
    parser.add_argument('--output', default=str(root / 'fixed_videos'), help='修复后视频输出目录')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出文件')
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
                output_path = output_root / category / relative
                total += 1
                if fix_video(video_path, output_path, ffmpeg_cmd, use_shell, overwrite=args.overwrite):
                    success += 1

    print(f'处理完成：成功 {success}/{total} 个视频。')
    if total == 0:
        print('没有找到视频。数据集目录要求：dataset/NonViolence 与 dataset/Violence。')


if __name__ == '__main__':
    main()
