# 🧩 设计说明

## 🎞️ 1. 为什么先抽帧

当前版本把视频任务拆成图片二分类任务。这样做的优点是流程简单、训练速度快、对显存要求低，也便于导出 TFLite 模型。

局限也比较明确：模型只看单帧，无法直接利用动作的时间变化。如果后续继续提高效果，可以考虑 3D CNN、LSTM、TimeSformer 或“帧特征 + 时序模型”的方案。

## 🗂️ 2. 数据目录设计

项目默认使用三层数据目录：

```text
data/raw      原始视频
data/fixed    修复后视频
data/frames   抽帧图片
```

分层保存可以让每一步产物保持清楚：视频封装异常时只重跑修复，抽帧参数变化时只重跑抽帧，训练失败也不会影响原始数据。

## 🧠 3. 模型设计

模型使用 MobileNetV2 作为图像特征提取主干：

```text
Input(224×224×3)
  → MobileNetV2(include_top=False)
  → GlobalAveragePooling2D
  → Dense(512, relu)
  → Dropout(0.5)
  → BatchNormalization
  → Dense(1, sigmoid)
```

输出是一个 0 到 1 之间的概率，默认大于等于 0.5 判为 `Violence`。

## 🏋️ 4. 训练策略

训练分为两个阶段。

第一阶段冻结 MobileNetV2 主干，只训练后面的分类头。这个阶段使用较大学习率，让分类层先适应当前数据。

第二阶段解冻主干最后若干层，用更小学习率微调。BatchNorm 层保持冻结，减少小 batch 训练时统计量抖动带来的影响。

## 📊 5. 评估设计

训练结束后会导出：

- `metrics.json`：最终评估指标；
- `history.csv`：每轮训练记录；
- `loss_curve.png`、`accuracy_curve.png`、`auc_curve.png`：训练曲线；
- `confusion_matrix.png`：混淆矩阵；
- `class_indices.json`：类别和数字标签的对应关系。

报告中应同时关注 `accuracy`、`precision`、`recall` 和 `f1`。对于暴力识别任务，漏检和误报都需要单独分析。

## 📱 6. 为什么保留 H5 和 TFLite

`.h5` 适合在训练电脑上继续评估和预测；`.tflite` 更适合部署到开发板或轻量环境。当前导出的是 FP16 TFLite，文件更小，推理速度通常也更友好。

## 🔧 7. 后续改进方向

- 将视频预测从“帧概率平均”改成“Top-K 高风险帧平均”；
- 增加 Grad-CAM，检查模型关注区域；
- 加入多随机种子实验，减少单次结果偶然性；
- 加入帧去重，减少相邻帧高度重复带来的偏差；
- 训练时序模型，显式建模动作变化。
