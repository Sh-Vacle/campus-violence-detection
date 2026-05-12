<p align="center">
  <img src="../assets/logo.svg" width="86" alt="校园暴力识别 Logo">
</p>

<h1 align="center">📦 数据集目录</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Data-Video%20Dataset-6A5ACD" alt="Video Dataset">
  <img src="https://img.shields.io/badge/Classes-Violence%20%7C%20NonViolence-2E8B57" alt="Classes">
  <img src="https://img.shields.io/badge/Git-Ignored-181717?logo=git" alt="Git Ignored">
</p>

---

## 📌 目录作用

`dataset/` 用于存放原始视频数据集。该目录不进入 Git 跟踪，原因是视频文件体积较大，并且数据集通常存在再分发限制。

## 📁 推荐结构

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

## 🏷️ 类别说明

| 目录 | 类别标签 | 说明 |
|---|---:|---|
| `NonViolence/` | 0 | 非暴力视频 |
| `Violence/` | 1 | 暴力视频 |

默认类别映射：

```text
{'NonViolence': 0, 'Violence': 1}
```

## 🔗 数据集来源

数据集来源：Kaggle Real Life Violence Situations Dataset。

数据集文件不随仓库分发，复现实验时需要按上述目录结构自行放置原始视频。
