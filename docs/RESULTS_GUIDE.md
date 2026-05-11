# 📊 结果文件说明

训练目录默认为 `runs/mobilenetv2`。

## 📄 metrics.json

最终评估指标，适合写进报告或项目说明。

重点字段：

- `accuracy`：验证集准确率；
- `precision`：预测为暴力的样本中，有多少是真的暴力；
- `recall`：真实暴力样本中，有多少被识别出来；
- `f1`：precision 和 recall 的综合指标；
- `auc_from_predictions`：基于预测概率重新计算的 AUC；
- `confusion_matrix`：混淆矩阵。

## 🧾 history.csv

每一轮训练的记录。该文件可以用 Excel 打开，也可以用于重新绘制训练曲线。

## 📈 曲线图

- `loss_curve.png`
- `accuracy_curve.png`
- `auc_curve.png`

曲线如果大幅震荡，优先检查 batch size、学习率和数据质量。

## 🧠 模型文件

- `best_model_frozen.h5`：冻结阶段验证损失最低的模型；
- `best_model_finetuned.h5`：微调阶段验证损失最低的模型；
- `best_model_fp16.tflite`：轻量化推理模型。
