# 🗂️ 数据说明

## 🎬 原始视频

原始视频放在 `data/raw`，类别目录保持如下结构：

```text
data/raw/
├── NonViolence/
└── Violence/
```

支持的视频格式包括：`.mp4`、`.avi`、`.mov`、`.mkv`。

## 🎞️ 抽帧数据

训练代码读取图片数据，不直接读取视频。抽帧结果放在 `data/frames`：

```text
data/frames/
├── NonViolence/
│   └── video_name/
│       ├── frame_000001.jpg
│       └── ...
└── Violence/
    └── video_name/
        ├── frame_000001.jpg
        └── ...
```

## 📦 数据存放规则

`data/raw`、`data/fixed`、`data/frames` 默认被 `.gitignore` 忽略。公开仓库只保留代码、说明和示例格式，真实数据集可通过数据来源说明或下载链接单独提供。
