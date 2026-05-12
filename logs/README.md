<p align="center">
  <img src="../assets/logo.svg" width="86" alt="校园暴力识别 Logo">
</p>

<h1 align="center">📈 训练日志目录</h1>

<p align="center">
  <img src="https://img.shields.io/badge/TensorBoard-Logs-FF6F00?logo=tensorflow&logoColor=white" alt="TensorBoard Logs">
  <img src="https://img.shields.io/badge/Git-Ignored-181717?logo=git" alt="Git Ignored">
</p>

---

## 📌 目录作用

`logs/` 用于存放 TensorBoard 训练日志。训练日志可由训练过程重新生成，因此不进入 Git 仓库跟踪。

## 🚀 训练时自动生成

```powershell
python src/04_train_model.py
```

## 🔎 查看 TensorBoard

```powershell
tensorboard --logdir logs
```

浏览器访问：

```text
http://localhost:6006
```
