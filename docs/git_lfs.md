# 📦 大文件管理

视频、模型权重和压缩包体积通常较大，不适合直接放进普通 Git 提交。本仓库已经在 `.gitattributes` 中为常见视频和模型格式预留 Git LFS 规则。

初始化 Git LFS：

```bash
git lfs install
```

按需跟踪大文件类型：

```bash
git lfs track "*.mp4"
git lfs track "*.h5"
git lfs track "*.tflite"
git add .gitattributes
```

更清晰的项目组织方式是：代码、配置和文档进入仓库；数据集、训练权重和中间产物通过单独的存储位置管理，并在文档中说明来源和处理流程。
