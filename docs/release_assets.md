<p align="center">
  <img src="../assets/logo.svg" width="96" alt="校园暴力识别 Logo">
</p>

<h1 align="center">GitHub Releases 模型资产说明</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Release-v1.0.0-181717?logo=github" alt="Release v1.0.0">
  <img src="https://img.shields.io/badge/Model-Keras%20%2B%20TFLite-FF6F00" alt="Keras and TFLite">
</p>

---

## Release 信息

```text
Tag: v1.0.0
Title: Campus Violence Recognition Models v1.0
Asset: campus_violence_models_v1.0.zip
```

## 资产包结构

```text
models/
├─ best_model_finetuned.h5
├─ best_model_fp16.tflite
└─ best_model_frozen.h5
```

## 模型文件说明

- `best_model_finetuned.h5`：最终微调后的 Keras 推理模型，用于单视频预测。
- `best_model_fp16.tflite`：FP16 TFLite 模型，用于部署实验。
- `best_model_frozen.h5`：冻结骨干网络阶段的最佳检查点，用于对比和回退。

## 使用方式

模型资产包解压到项目根目录后，目录结构为：

```text
models/best_model_finetuned.h5
models/best_model_fp16.tflite
models/best_model_frozen.h5
```

单视频推理命令：

```powershell
python src/05_predict_video.py --video ".\dataset\Violence\V_100.mp4" --model ".\models\best_model_finetuned.h5" --fps 1
```
