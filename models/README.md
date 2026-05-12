<p align="center">
  <img src="../assets/logo.svg" width="86" alt="校园暴力识别 Logo">
</p>

<h1 align="center">🧠 模型文件目录</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Keras-.h5-FF6F00" alt="Keras model">
  <img src="https://img.shields.io/badge/TFLite-FP16-2E8B57" alt="TFLite FP16">
  <img src="https://img.shields.io/badge/Release-v1.0.0-181717?logo=github" alt="GitHub Release">
  <img src="https://img.shields.io/badge/Git-Ignored-181717?logo=git" alt="Git Ignored">
</p>

---

## 📌 目录作用

`models/` 用于存放本地训练生成或从 GitHub Releases 下载的模型文件。模型文件体积较大，因此不进入 Git 仓库跟踪。

## 📁 常用模型文件

```text
models/
├─ best_model_frozen.h5
├─ best_model_finetuned.h5
└─ best_model_fp16.tflite
```

## 🧩 文件说明

| 文件 | 说明 |
|---|---|
| `best_model_finetuned.h5` | 最终微调后的 Keras 推理模型，由 `src/05_predict_video.py` 调用 |
| `best_model_fp16.tflite` | FP16 TFLite 部署模型，用于部署实验 |
| `best_model_frozen.h5` | 冻结骨干网络阶段的最佳模型，用于对比和回退 |

## 🚀 获取方式

模型文件通过 GitHub Releases 分发：

```text
campus_violence_models_v1.0.zip
```

解压后的目录结构：

```text
models/
├─ best_model_finetuned.h5
├─ best_model_fp16.tflite
└─ best_model_frozen.h5
```
