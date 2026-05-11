# frames

该目录用于存放由 `src/02_extract_frames.py` 生成的抽帧图片。

目录结构：

```text
frames/
├─ NonViolence/
│  └─ <video_name>/frame_000001.jpg
└─ Violence/
   └─ <video_name>/frame_000001.jpg
```

抽帧图片属于中间产物，可由原始视频重新生成，因此不进入 Git 仓库跟踪。
