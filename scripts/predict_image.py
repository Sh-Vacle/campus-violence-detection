import argparse
import json

from violence_detection.predict import predict_image


def main():
    parser = argparse.ArgumentParser(description="Predict one image.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--class-indices", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    result = predict_image(args.model, args.image, args.class_indices, image_size=args.image_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
