@echo off
cd /d "%~dp0"
echo [1/7] Kiem tra du lieu...
D:\Python\.venv\Scripts\python.exe -m src.cli audit
if errorlevel 1 goto error
echo [2/7] Phan tich EDA...
D:\Python\.venv\Scripts\python.exe -m src.cli eda --limit 100
if errorlevel 1 goto error
echo [3/7] Tao split mau 100 anh...
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --sample-size 100 --seed 42
if errorlevel 1 goto error
echo [4/7] Chay baseline...
D:\Python\.venv\Scripts\python.exe -m src.cli baselines --limit 100 --split test
if errorlevel 1 goto error
echo [5/7] Thu U-Net 1 epoch...
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg --epochs 1 --limit 100
if errorlevel 1 goto error
echo [6/7] Thu Faster R-CNN MobileNet 1 epoch, khong tai pretrained...
D:\Python\.venv\Scripts\python.exe -m src.cli train-det --epochs 1 --limit 100 --weights none --model fasterrcnn_mobilenet_v3_large_fpn
if errorlevel 1 goto error
echo [7/7] Chon nguong tren Validation va danh gia...
D:\Python\.venv\Scripts\python.exe -m src.cli select-val --limit 50
if errorlevel 1 goto error
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task seg
if errorlevel 1 goto error
D:\Python\.venv\Scripts\python.exe -m src.cli evaluate --task det
if errorlevel 1 goto error
echo Hoan tat pipeline mau. Day khong phai ket qua thuc nghiem cuoi cung.
pause
exit /b 0
:error
echo Pipeline gap loi. Hay chup man hinh CMD va gui log de xu ly.
pause
exit /b 1
