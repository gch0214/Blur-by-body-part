@echo off
chcp 65001 >nul
rem 人体部位打码工具 - 启动器
rem 首次运行会在 %USERPROFILE%\blur_env 建一个独立的 Python 环境,不影响 Anaconda 和其他程序
set "ENV=%USERPROFILE%\blur_env"

if exist "%ENV%\Scripts\python.exe" goto run

echo 首次运行:正在创建独立环境并安装依赖,大约需要几分钟,请不要关闭本窗口...
echo.
python -m venv "%ENV%"
if errorlevel 1 goto fail
"%ENV%\Scripts\python.exe" -m pip install --upgrade pip
"%ENV%\Scripts\python.exe" -m pip install mediapipe
if errorlevel 1 goto fail
"%ENV%\Scripts\python.exe" -c "import mediapipe, cv2; print('依赖检查通过')"
if errorlevel 1 goto fail

:run
start "" "%ENV%\Scripts\pythonw.exe" "%~dp0body_blur.py" %*
exit /b 0

:fail
echo.
echo 安装失败。请把这个窗口的内容截图发给我。
echo 如需重新安装,先删除文件夹 %ENV% 再双击本文件。
pause
exit /b 1
