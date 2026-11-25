@echo off
cd /d "%~dp0"
echo [1/5] Kiem tra du lieu...
D:\Python\.venv\Scripts\python.exe -m src.cli audit
if errorlevel 1 goto error
echo [2/5] Tao split mau 100 anh...
D:\Python\.venv\Scripts\python.exe -m src.cli prepare --sample-size 100 --seed 42
if errorlevel 1 goto error
echo [3/5] Chay baseline...
D:\Python\.venv\Scripts\python.exe -m src.cli baselines --limit 100
if errorlevel 1 goto error
echo [4/5] Thu U-Net 1 epoch...
D:\Python\.venv\Scripts\python.exe -m src.cli train-seg --epochs 1 --limit 100
if errorlevel 1 goto error
echo [5/5] Thu Faster R-CNN MobileNet 1 epoch, khong tai pretrained...
D:\Python\.venv\Scripts\python.exe -m src.cli train-det --epochs 1 --limit 100 --weights none --model fasterrcnn_mobilenet_v3_large_fpn
if errorlevel 1 goto error
echo Hoan tat pipeline mau. Day khong phai ket qua thuc nghiem cuoi cung.
pause
exit /b 0
:error
echo Pipeline gap loi. Hay chup man hinh CMD va gui log de xu ly.
pause
exit /b 1
