# 🍊 Orange Pi Kunpeng Pro 部署记录

训练脚本可以导出 `best_model_fp16.tflite`。板端只做推理时，优先使用 TFLite，文件更小，也更方便放到 openEuler 环境中运行。

## 📤 1. 传输文件

在电脑端执行：

```bash
scp runs/mobilenetv2/best_model_fp16.tflite openEuler@板子IP:/home/openEuler/campus-violence/
scp runs/mobilenetv2/class_indices.json openEuler@板子IP:/home/openEuler/campus-violence/
```

## ⚙️ 2. 板端环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pillow numpy tflite-runtime
```

如果 `tflite-runtime` 安装失败，可以先在板端安装完整 TensorFlow，或将推理流程保留在电脑端运行。

## 🖼️ 3. 单张图片推理

将仓库代码同步到开发板后，执行：

```bash
python scripts/predict_tflite.py \
  --model best_model_fp16.tflite \
  --class-indices class_indices.json \
  --image test.jpg
```

## ⚠️ 4. 部署注意事项

- 板端视频推理前，应先用单张图片验证模型和环境；
- 摄像头接入、帧率、温度和供电需要单独测试；
- 如果比赛或课程只要求最终识别效果，应优先保证电脑端训练和验证指标稳定，再进行板端迁移。
