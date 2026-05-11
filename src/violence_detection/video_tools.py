from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from .paths import ensure_dir, safe_stem

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_videos(root: str | Path):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                yield class_dir.name, video_path


def fix_video_file(src: str | Path, dst: str | Path, overwrite: bool = False) -> None:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)

    if dst.exists() and not overwrite:
        return

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-c",
        "copy",
        "-map",
        "0",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def fix_videos(input_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> int:
    videos = list(iter_videos(input_dir))
    fixed = 0
    for class_name, video_path in tqdm(videos, desc="fix videos"):
        dst = Path(output_dir) / class_name / video_path.name
        fix_video_file(video_path, dst, overwrite=overwrite)
        fixed += 1
    return fixed


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
    overwrite: bool = False,
) -> int:
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)

    if overwrite:
        for old_file in output_dir.glob("*.jpg"):
            old_file.unlink()
    elif any(output_dir.glob("*.jpg")):
        return len(list(output_dir.glob("*.jpg")))

    out_pattern = output_dir / "frame_%06d.jpg"
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(out_pattern), vf=f"fps={fps}", start_number=1)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"抽帧失败：{video_path}\n{stderr}") from exc

    return len(list(output_dir.glob("*.jpg")))


def extract_frames(input_dir: str | Path, output_dir: str | Path, fps: float = 1.0, overwrite: bool = False) -> int:
    videos = list(iter_videos(input_dir))
    total_frames = 0

    for class_name, video_path in tqdm(videos, desc="extract frames"):
        frame_dir = Path(output_dir) / class_name / safe_stem(video_path.name)
        total_frames += extract_frames_from_video(video_path, frame_dir, fps=fps, overwrite=overwrite)

    return total_frames


def copy_demo_tree(root: str | Path) -> None:
    """Create an empty data tree so new users know where files go."""
    root = Path(root)
    for part in ["raw/NonViolence", "raw/Violence", "fixed", "frames"]:
        ensure_dir(root / part)
