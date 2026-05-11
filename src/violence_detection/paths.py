from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
