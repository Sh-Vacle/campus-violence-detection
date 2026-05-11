# 📘 实验报告：校园暴力视频识别

## 🎯 1. 实验目的

本实验实现一个基于视频抽帧的校园暴力识别流程。实验目标是完成从原始视频到模型训练、验证评估、离线预测和轻量化导出的完整闭环。

本实验不是实时监控系统，重点是验证图像分类模型在抽帧数据上的二分类效果。

## 🗂️ 2. 数据集

数据按两个类别组织：

```text
NonViolence  非暴力视频或帧
Violence     暴力视频或帧
```

原始视频放在 `data/raw`，经过 FFmpeg 修复后保存到 `data/fixed`，再按固定帧率抽帧到 `data/frames`。

训练前执行：

```bash
python scripts/check_dataset.py --data data/frames
```

将输出的类别数量记录在下表中：

| 类别 | 图片数量 |
| --- | ---: |
| NonViolence | 待填写 |
| Violence | 待填写 |
| Total | 待填写 |

## 🧹 3. 数据预处理

抽帧命令：

```bash
python scripts/extract_frames.py --input data/fixed --output data/frames --fps 1
```

训练阶段使用的图像预处理包括：

- resize 到 224×224；
- 像素值缩放到 `[0, 1]`；
- 训练集随机旋转、平移、缩放、剪切和水平翻转；
- 验证集只做缩放，不做增强。

## 🧠 4. 模型结构

模型以 MobileNetV2 为主干，去掉原始分类头后接入自定义二分类头：

```text
MobileNetV2 → GlobalAveragePooling2D → Dense → Dropout → BatchNorm → Sigmoid
```

训练分为冻结训练和微调两个阶段。

## ⚙️ 5. 训练配置

| 配置项 | 数值 |
| --- | --- |
| 输入尺寸 | 224×224 |
| batch size | 训练 8，验证 32 |
| 冻结训练轮数 | 10 |
| 微调轮数 | 8 |
| 冻结阶段学习率 | 0.001 |
| 微调阶段学习率 | 0.00001 |
| 验证集比例 | 0.2 |
| 随机种子 | 42 |

实际训练命令：

```bash
python scripts/train.py --data data/frames --out runs/mobilenetv2 --export-tflite
```

## 📊 6. 实验结果

训练完成后，从 `runs/mobilenetv2/metrics.json` 填写结果。

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 待填写 |
| Precision | 待填写 |
| Recall | 待填写 |
| F1 | 待填写 |
| AUC | 待填写 |

混淆矩阵图片：

```text
runs/mobilenetv2/confusion_matrix.png
```

训练曲线：

```text
runs/mobilenetv2/loss_curve.png
runs/mobilenetv2/accuracy_curve.png
runs/mobilenetv2/auc_curve.png
```

## 🔎 7. 结果分析

分析结果时重点关注：

1. 验证集准确率是否稳定；
2. 暴力类召回率是否足够；
3. 是否存在明显误报或漏报；
4. 微调阶段是否比冻结阶段有提升；
5. 数据量和类别分布是否影响结果。

## 🔧 8. 问题与改进

当前方法的主要限制：

- 单帧分类不能充分利用动作时序；
- 视频级标签用于帧训练时会引入噪声；
- 数据集场景单一时，模型容易过拟合；
- 真实部署前还需要做速度、误报率和隐私合规测试。

后续可改进方向：

- 使用多帧时序模型；
- 加入帧去重和质量过滤；
- 使用 Grad-CAM 做可解释性分析；
- 对不同场景、光照、角度做更细的测试。
