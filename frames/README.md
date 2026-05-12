<p align="center">
  <img src="../assets/logo.svg" width="86" alt="校园暴力识别 Logo">
</p>

<h1 align="center">🎞️ 抽帧图片目录</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Generated%20By-FFmpeg-007808?logo=ffmpeg&logoColor=white" alt="FFmpeg">
  <img src="https://img.shields.io/badge/Input-Video%20Frames-6A5ACD" alt="Video Frames">
  <img src="https://img.shields.io/badge/Git-Ignored-181717?logo=git" alt="Git Ignored">
</p>

---

## 📌 目录作用

`frames/` 用于存放由 `src/02_extract_frames.py` 生成的抽帧图片。该目录属于中间产物，可由原始视频重新生成，因此不进入 Git 仓库跟踪。

## 📁 推荐结构

```text
frames/
├─ NonViolence/
│  └─ <video_name>/
│     ├─ frame_000001.jpg
│     ├─ frame_000002.jpg
│     └─ ...
└─ Violence/
   └─ <video_name>/
      ├─ frame_000001.jpg
      ├─ frame_000002.jpg
      └─ ...
```

## 🚀 生成命令

```powershell
python src/02_extract_frames.py --fps 1
```

覆盖已有抽帧结果：

```powershell
python src/02_extract_frames.py --fps 1 --overwrite
```

## 🔎 检查命令

```powershell
python src/03_check_dataset.py
```
