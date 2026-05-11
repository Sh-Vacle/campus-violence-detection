import argparse
import json

from violence_detection.tflite_predict import predict_tflite


def main():
    parser = argparse.ArgumentParser(description="Predict one image with a TFLite model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--class-indices", default=None)
    args = parser.parse_args()

    result = predict_tflite(args.model, args.image, args.class_indices)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
