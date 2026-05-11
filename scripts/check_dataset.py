import argparse

from violence_detection.data import count_images


def main():
    parser = argparse.ArgumentParser(description="Count images in class folders.")
    parser.add_argument("--data", required=True, help="抽帧后的数据目录，例如 data/frames")
    args = parser.parse_args()

    counts = count_images(args.data)
    if not counts:
        print("没有找到类别目录。")
        return

    total = sum(counts.values())
    for name, n in counts.items():
        print(f"{name}: {n}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
