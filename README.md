<p align="center">
  <img src="assets/logo.svg" width="128" alt="校园暴力识别 Logo">
</p>

<h1 align="center">校园暴力识别系统</h1>
<h3 align="center">Campus Violence Recognition</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.10">
  <img src="https://img.shields.io/badge/Backbone-MobileNetV2-2E8B57" alt="MobileNetV2">
  <img src="https://img.shields.io/badge/Task-Video%20Classification-6A5ACD" alt="Video Classification">
  <img src="https://img.shields.io/badge/FFmpeg-Required-007808?logo=ffmpeg&logoColor=white" alt="FFmpeg">
  <img src="https://img.shields.io/badge/Models-GitHub%20Releases-181717?logo=github" alt="GitHub Releases">
</p>

<p align="center">
  基于 <b>MobileNetV2</b> 的校园暴力 / 非暴力视频二分类项目<br>
  覆盖视频抽帧、迁移学习训练、模型评估、单视频推理与 TFLite 导出
</p>

---

## 目录

- [项目简介](#项目简介)
- [仓库结构](#仓库结构)
- [环境配置](#环境配置)
- [数据集目录](#数据集目录)
- [视频抽帧](#视频抽帧)
- [数据集检查](#数据集检查)
- [模型训练](#模型训练)
- [预训练模型与 GitHub Releases](#预训练模型与-github-releases)
- [单视频预测](#单视频预测)
- [实验结果](#实验结果)
- [Git 跟踪规则](#git-跟踪规则)

---

## 项目简介

本项目面向校园场景下的视频暴力行为识别任务，使用 **MobileNetV2** 作为特征提取骨干网络，并在其后接入全局平均池化、全连接层、Dropout、BatchNorm 与 Sigmoid 输出层，实现 **Violence / NonViolence** 二分类。

整体流程：

```text
原始视频 → FFmpeg 抽帧 → 图像二分类训练 → 单视频推理 → TFLite 模型导出
```

核心功能：

- FFmpeg 自动查找与跨平台调用；
- 原始视频按指定 FPS 抽取图片帧；
- 基于 Keras `flow_from_directory` 的图像数据读取；
- MobileNetV2 迁移学习训练与部分层微调；
- 输出 Accuracy、AUC、Precision、Recall、F1-score 与 Confusion Matrix；
- 导出 `.h5` Keras 模型与 FP16 `.tflite` 部署模型；
- 支持单个视频的 Violence / NonViolence 推理。

---

## 仓库结构

```text
校园暴力识别/
├─ assets/
│  └─ logo.svg                 # 项目标识
├─ src/
│  ├─ ffmpeg_utils.py          # FFmpeg 查找与子进程调用工具
│  ├─ 01_fix_videos.py         # 视频重新封装 / 修复
│  ├─ 02_extract_frames.py     # 视频抽帧
│  ├─ 03_check_dataset.py      # 数据集结构检查
│  ├─ 04_train_model.py        # 模型训练、评估、TFLite 导出
│  └─ 05_predict_video.py      # 单视频推理
├─ dataset/                    # 原始视频目录占位
├─ frames/                     # 抽帧图片目录占位
├─ fixed_videos/               # 修复后视频目录占位
├─ models/                     # 模型目录占位，模型文件通过 Releases 分发
├─ logs/                       # TensorBoard 日志目录占位
├─ results/                    # 训练指标与预测示例记录
├─ scripts/                    # Windows 辅助脚本
├─ docs/                       # 项目文档
├─ environment_tf210_py39.yml  # Conda 环境配置
├─ requirements_tf210.txt      # pip 依赖列表
├─ PROJECT_FILES_MANIFEST.txt  # 文件清单
├─ README.md
└─ .gitignore
```

---

## 环境配置

参考环境：

```text
Windows 10 / 11
Python 3.9
TensorFlow 2.10.x
FFmpeg
CUDA-capable GPU，可选
```

使用 Conda 创建环境：

```powershell
conda env create -f environment_tf210_py39.yml
conda activate tf210_py39
```

已有 Conda 环境的依赖更新命令：

```powershell
conda activate tf210_py39
conda install -c conda-forge ffmpeg scipy=1.9.3 -y
```

TensorFlow / GPU 检查命令：

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

FFmpeg 检查命令：

```powershell
ffmpeg -version
```

---

## 数据集目录

原始视频数据集不进入 Git 跟踪，原因是文件体积较大且存在数据集再分发限制。目录结构如下：

```text
dataset/
├─ NonViolence/
│  ├─ NV_1.mp4
│  ├─ NV_2.mp4
│  └─ ...
└─ Violence/
   ├─ V_1.mp4
   ├─ V_2.mp4
   └─ ...
```

默认类别映射：

```text
{'NonViolence': 0, 'Violence': 1}
```

数据集来源：Kaggle Real Life Violence Situations Dataset。

---

## 视频抽帧

标准抽帧命令：

```powershell
python src/02_extract_frames.py --fps 1
```

覆盖已有抽帧结果：

```powershell
python src/02_extract_frames.py --fps 1 --overwrite
```

默认输出目录：

```text
frames/NonViolence
frames/Violence
```

视频重新封装后抽帧：

```powershell
python src/01_fix_videos.py --overwrite
python src/02_extract_frames.py --input fixed_videos --output frames --fps 1 --overwrite
```

---

## 数据集检查

```powershell
python src/03_check_dataset.py
```

输出示例：

```text
Found 8418 images belonging to 2 classes.
Found 2103 images belonging to 2 classes.
训练集图片数：8418
验证集图片数：2103
类别映射：{'NonViolence': 0, 'Violence': 1}
```

---

## 模型训练

默认训练命令：

```powershell
python src/04_train_model.py
```

低显存模式：

```powershell
python src/04_train_model.py --batch-train 1 --batch-val 4
```

仅训练冻结骨干网络阶段：

```powershell
python src/04_train_model.py --no-finetune
```

默认输出文件：

```text
models/best_model_frozen.h5
models/best_model_finetuned.h5
models/best_model_fp16.tflite
```

主要推理模型：

```text
models/best_model_finetuned.h5
```

---

## 预训练模型与 GitHub Releases

训练后的模型文件不提交到 Git 仓库，通过 **GitHub Releases** 作为 Release Asset 分发。

Release 信息：

```text
Tag: v1.0.0
Title: Campus Violence Recognition Models v1.0
Asset: campus_violence_models_v1.0.zip
```

Release Asset 内容：

```text
models/
├─ best_model_finetuned.h5     # 最终微调后的 Keras 推理模型
├─ best_model_fp16.tflite      # FP16 TFLite 部署模型
└─ best_model_frozen.h5        # 冻结骨干网络阶段最佳模型
```

模型包解压到项目根目录后，`models/` 目录即可用于单视频推理。

---

## 单视频预测

暴力视频示例：

```powershell
python src/05_predict_video.py --video ".\dataset\Violence\V_100.mp4" --model ".\models\best_model_finetuned.h5" --fps 1
```

非暴力视频示例：

```powershell
python src/05_predict_video.py --video ".\dataset\NonViolence\NV_100.mp4" --model ".\models\best_model_finetuned.h5" --fps 1
```

输出示例：

```text
========== 预测结果 ==========
抽帧数量：5
平均暴力概率：0.8291
最高暴力概率：0.9180
最低暴力概率：0.7448
预测类别：Violence
```

---

## 实验结果

验证集规模：

```text
Train images: 8418
Validation images: 2103
```

最终评估指标：

```text
Accuracy: 0.9163
AUC: 0.9653
Precision: 0.9356
Recall: 0.8994
F1-score: 0.9171
```

混淆矩阵：

```text
[[953, 67],
 [109, 974]]
```

预测示例：

```text
V_100.mp4     avg_prob=0.8291    prediction=Violence
V_1000.mp4    avg_prob=0.8426    prediction=Violence
V_101.mp4     avg_prob=0.9385    prediction=Violence
NV_100.mp4    avg_prob=0.0940    prediction=NonViolence
NV_1000.mp4   avg_prob=0.0694    prediction=NonViolence
```

---

## Git 跟踪规则

`.gitignore` 保留代码、说明文档、结果记录和目录占位文件，同时排除数据集、抽帧结果、训练日志与模型大文件。

```gitignore
dataset/*
frames/*
fixed_videos/*
logs/*
models/*
```

`dataset/`、`frames/`、`fixed_videos/`、`models/` 和 `logs/` 目录通过 `.gitkeep` 与 README 文件保留目录结构。
