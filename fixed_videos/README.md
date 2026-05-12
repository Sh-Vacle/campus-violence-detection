<p align="center">
  <img src="../assets/logo.svg" width="86" alt="校园暴力识别 Logo">
</p>

<h1 align="center">🛠️ 修复后视频目录</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Video-Remux-2E8B57" alt="Video Remux">
  <img src="https://img.shields.io/badge/Tool-FFmpeg-007808?logo=ffmpeg&logoColor=white" alt="FFmpeg">
  <img src="https://img.shields.io/badge/Git-Ignored-181717?logo=git" alt="Git Ignored">
</p>

---

## 📌 目录作用

`fixed_videos/` 用于存放由 `src/01_fix_videos.py` 重新封装后的视频文件。该目录属于中间产物，主要用于处理部分视频编码或封装不稳定的问题。

## 📁 推荐结构

```text
fixed_videos/
├─ NonViolence/
└─ Violence/
```

## 🚀 生成命令

```powershell
python src/01_fix_videos.py --overwrite
```

## 🎞️ 基于修复后视频抽帧

```powershell
python src/02_extract_frames.py --input fixed_videos --output frames --fps 1 --overwrite
```
