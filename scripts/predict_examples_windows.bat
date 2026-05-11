@echo off
chcp 65001 >nul

echo ==========================================
echo 校园暴力识别系统：单视频预测示例
echo ==========================================
echo.
echo Conda 环境：tf210_py39
echo.

python src\05_predict_video.py --video ".\dataset\Violence\V_100.mp4" --model ".\models\best_model_finetuned.h5" --fps 1
if errorlevel 1 goto error

python src\05_predict_video.py --video ".\dataset\Violence\V_1000.mp4" --model ".\models\best_model_finetuned.h5" --fps 1
if errorlevel 1 goto error

python src\05_predict_video.py --video ".\dataset\NonViolence\NV_100.mp4" --model ".\models\best_model_finetuned.h5" --fps 1
if errorlevel 1 goto error

echo.
echo 预测示例执行完成。
pause
exit /b 0

:error
echo.
echo 预测流程中断，错误信息见上方输出。
pause
exit /b 1
