import argparse

from violence_detection.video_tools import fix_videos


def main():
    parser = argparse.ArgumentParser(description="Repair videos with ffmpeg stream copy.")
    parser.add_argument("--input", required=True, help="原始视频目录，例如 data/raw")
    parser.add_argument("--output", required=True, help="修复后输出目录，例如 data/fixed")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()

    count = fix_videos(args.input, args.output, overwrite=args.overwrite)
    print(f"done, fixed videos: {count}")


if __name__ == "__main__":
    main()
