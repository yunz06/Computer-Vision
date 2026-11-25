@echo off
cd /d "%~dp0"
echo [1/6] Audit dataset ISIC...
D:\Python\.venv\Scripts\python.exe -m src.cli audit
if errorlevel 1 goto error
echo [2/6] Tao split tren toan bo du lieu...
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --seed 42
if errorlevel 1 goto error
echo [3/6] Chay baseline Otsu va K-means...
D:\Python\.venv\Scripts\python.exe -m src.cli baselines
if errorlevel 1 goto error
echo [4/6] Huan luyen U-Net...
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg
if errorlevel 1 goto error
echo [5/6] Huan luyen Faster R-CNN theo cau hinh...
D:\Python\.venv\Scripts\python.exe -m src.cli train-det
if errorlevel 1 goto error
echo [6/6] Danh gia tren test split...
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task seg
if errorlevel 1 goto error
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task det
if errorlevel 1 goto error
echo Da hoan tat pipeline huan luyen va danh gia.
echo Mo demo bang lenh: D:\Python\.venv\Scripts\python.exe -m src.cli demo
pause
exit /b 0
:error
echo Pipeline gap loi. Neu Detection het VRAM, chay lai rieng lenh:
echo D:\Python\.venv\Scripts\python.exe -m src.cli train-det --model fasterrcnn_mobilenet_v3_large_fpn
pause
exit /b 1
