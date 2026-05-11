@echo off
chcp 65001 >nul

echo ==========================================
echo 校园暴力识别系统：训练流程
echo 1. 视频抽帧
echo 2. 数据集检查
echo 3. 模型训练
echo ==========================================
echo.
echo Conda 环境：tf210_py39
echo.

python src\02_extract_frames.py --fps 1
if errorlevel 1 goto error

python src\03_check_dataset.py
if errorlevel 1 goto error

python src\04_train_model.py
if errorlevel 1 goto error

echo.
echo 流程完成，模型文件输出到 models 目录。
pause
exit /b 0

:error
echo.
echo 流程中断，错误信息见上方输出。
pause
exit /b 1
