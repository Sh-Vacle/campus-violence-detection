# models

该目录用于存放本地训练生成或从 GitHub Releases 下载的模型文件。

常用模型文件：

```text
best_model_frozen.h5
best_model_finetuned.h5
best_model_fp16.tflite
```

文件说明：

- `best_model_finetuned.h5`：最终微调后的 Keras 推理模型，由 `src/05_predict_video.py` 调用。
- `best_model_fp16.tflite`：FP16 TFLite 部署模型，用于部署实验。
- `best_model_frozen.h5`：冻结骨干网络阶段的最佳模型，用于对比和回退。

模型文件通过 GitHub Releases 分发，不进入 Git 仓库跟踪。

Release Asset：

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
