import argparse
import json

from violence_detection.predict import predict_video


def main():
    parser = argparse.ArgumentParser(description="Predict one video by sampled frames.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--class-indices", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    result = predict_video(args.model, args.video, args.class_indices, image_size=args.image_size, fps=args.fps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
