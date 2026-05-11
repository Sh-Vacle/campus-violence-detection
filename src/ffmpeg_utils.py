"""
ffmpeg_utils.py

项目内部 FFmpeg 工具函数。

目标：
    1. 不在代码里写死任何用户本地路径；
    2. 支持 conda 环境、系统 PATH、通过指定环境 python.exe 启动脚本的情况；
    3. 兼容 Windows 下 ffmpeg.bat / ffmpeg.cmd。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def candidate_roots_from_python() -> list[Path]:
    """
    根据当前 Python 解释器位置反推环境目录。

    Windows conda 常见结构：
        D:/miniconda3/envs/tf210_py39/python.exe

    Linux/macOS conda 常见结构：
        /path/to/envs/tf210_py39/bin/python
    """
    python_path = Path(sys.executable).resolve()
    roots = [python_path.parent, python_path.parent.parent]

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def ffmpeg_candidates() -> list[Path]:
    """生成可能的 FFmpeg 路径列表。"""
    candidates: list[Path] = []

    # 1. 系统 PATH
    for name in ("ffmpeg.exe", "ffmpeg"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    # 2. conda 当前环境
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_root = Path(conda_prefix)
        candidates.extend([
            conda_root / "Library" / "bin" / "ffmpeg.exe",
            conda_root / "Scripts" / "ffmpeg.exe",
            conda_root / "Scripts" / "ffmpeg.bat",
            conda_root / "bin" / "ffmpeg",
        ])

    # 3. 当前 python.exe 所在环境
    for root in candidate_roots_from_python():
        candidates.extend([
            root / "Library" / "bin" / "ffmpeg.exe",
            root / "Scripts" / "ffmpeg.exe",
            root / "Scripts" / "ffmpeg.bat",
            root / "bin" / "ffmpeg",
        ])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def needs_shell(path: Path) -> bool:
    """Windows 下执行 .bat/.cmd 通常需要 shell=True。"""
    return path.suffix.lower() in {".bat", ".cmd"}


def run_subprocess(command: list[str], use_shell: bool = False, quiet: bool = False) -> subprocess.CompletedProcess:
    """
    运行外部命令。

    如果 use_shell=True，则先用 subprocess.list2cmdline 处理带空格路径。
    """
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None

    if use_shell:
        command_line = subprocess.list2cmdline(command)
        return subprocess.run(command_line, check=True, shell=True, stdout=stdout, stderr=stderr)

    return subprocess.run(command, check=True, shell=False, stdout=stdout, stderr=stderr)


def test_ffmpeg(candidate: Path) -> tuple[bool, bool]:
    """
    测试某个 FFmpeg 候选路径能否运行。

    返回：
        ok: 是否可用
        use_shell: 是否需要 shell=True
    """
    if not candidate.exists():
        return False, False

    use_shell = needs_shell(candidate)
    try:
        run_subprocess([str(candidate), "-version"], use_shell=use_shell, quiet=True)
        return True, use_shell
    except Exception:
        return False, False


def find_ffmpeg() -> tuple[str | None, bool]:
    """
    自动查找可运行的 FFmpeg。

    返回：
        ffmpeg_cmd: 可执行文件路径或 None
        use_shell: 是否需要 shell=True
    """
    for candidate in ffmpeg_candidates():
        ok, use_shell = test_ffmpeg(candidate)
        if ok:
            return str(candidate), use_shell
    return None, False


def print_ffmpeg_help() -> None:
    """打印 FFmpeg 找不到时的排查信息。"""
    print("[错误] 没找到可运行的 FFmpeg。")
    print()
    print("FFmpeg 未安装或当前环境无法运行：")
    print("    ffmpeg -version")
    print()
    print("Conda environment command:")
    print("    conda activate tf210_py39")
    print("    conda install -c conda-forge ffmpeg -y")
    print()
    print("[诊断] 当前 Python：")
    print(f"    {sys.executable}")
    print("[诊断] 当前 CONDA_PREFIX：")
    print(f"    {os.environ.get('CONDA_PREFIX')}")
    print("[诊断] 已尝试查找位置：")
    for candidate in ffmpeg_candidates():
        print(f"    {candidate}")
