@echo off
cd /d "%~dp0"
echo [1/8] Audit dataset ISIC...
D:\Python\.venv\Scripts\python.exe -m src.cli audit
if errorlevel 1 goto error
echo [2/8] Thong ke EDA...
D:\Python\.venv\Scripts\python.exe -m src.cli eda
if errorlevel 1 goto error
echo [3/8] Tao split tren toan bo du lieu...
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --seed 42
if errorlevel 1 goto error
echo [4/8] Chay baseline Otsu, K-means, Watershed, Center-box...
D:\Python\.venv\Scripts\python.exe -m src.cli baselines --split test
if errorlevel 1 goto error
echo [5/8] Huan luyen U-Net...
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg
if errorlevel 1 goto error
echo [6/8] Huan luyen Faster R-CNN theo cau hinh...
D:\Python\.venv\Scripts\python.exe -m src.cli train-det
if errorlevel 1 goto error
echo [7/8] Quet nguong tren Validation...
D:\Python\.venv\Scripts\python.exe -m src.cli select-val
if errorlevel 1 goto error
echo [8/8] Danh gia tren test split...
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task seg
if errorlevel 1 goto error
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task det
if errorlevel 1 goto error
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task unet_box
if errorlevel 1 goto error
echo Da hoan tat pipeline huan luyen va danh gia.
echo Mo demo bang lenh: D:\Python\.venv\Scripts\python.exe app.py
pause
exit /b 0
:error
echo Pipeline gap loi. Neu Detection het VRAM, chay lai rieng lenh:
echo D:\Python\.venv\Scripts\python.exe -m src.cli train-det --model fasterrcnn_mobilenet_v3_large_fpn --batch-size 1
pause
exit /b 1
