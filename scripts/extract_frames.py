import argparse

from violence_detection.video_tools import extract_frames


def main():
    parser = argparse.ArgumentParser(description="Extract video frames by class folders.")
    parser.add_argument("--input", required=True, help="视频目录，例如 data/raw 或 data/fixed")
    parser.add_argument("--output", required=True, help="帧输出目录，例如 data/frames")
    parser.add_argument("--fps", type=float, default=1.0, help="每秒抽几帧")
    parser.add_argument("--overwrite", action="store_true", help="重新抽帧并覆盖旧图片")
    args = parser.parse_args()

    total = extract_frames(args.input, args.output, fps=args.fps, overwrite=args.overwrite)
    print(f"done, frames: {total}")


if __name__ == "__main__":
    main()
