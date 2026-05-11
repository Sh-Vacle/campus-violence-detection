.PHONY: install check fix extract train predict-image predict-video tensorboard clean

install:
	pip install -r requirements.txt

check:
	python -m compileall src scripts tests
	python scripts/check_dataset.py --data data/frames

fix:
	python scripts/fix_videos.py --input data/raw --output data/fixed

extract:
	python scripts/extract_frames.py --input data/fixed --output data/frames --fps 1

train:
	python scripts/train.py --data data/frames --out runs/mobilenetv2 --export-tflite

predict-image:
	python scripts/predict_image.py --model runs/mobilenetv2/best_model_finetuned.h5 --class-indices runs/mobilenetv2/class_indices.json --image test.jpg

predict-video:
	python scripts/predict_video.py --model runs/mobilenetv2/best_model_finetuned.h5 --class-indices runs/mobilenetv2/class_indices.json --video test.mp4 --fps 1

tensorboard:
	tensorboard --logdir runs/mobilenetv2/logs

clean:
	rm -rf runs/* logs/* __pycache__ .pytest_cache
