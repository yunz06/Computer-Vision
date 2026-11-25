# Báo cáo kiểm thử source

## Kết quả đã kiểm tra

- Biên dịch toàn bộ `src`, `tests`, `app.py`: **PASS**.
- Unit & integration tests (`pytest -v`): **17 PASS** (0 failed, 0 skipped):
  - `tests/test_core.py`: `image_key`, `read_mask` ngưỡng `127`, `letterbox` & `unletterbox` (prob map linear, mask nearest, boxes), lọc box đa thành phần theo `min_area_ratio`, hậu xử lý `largest_component` + `hole_filling`, augmentation đồng bộ ảnh-mask, `binary_metrics` + `boundary_f1` + `segmentation_summary` (mean ± std, failure rate Jaccard < 0.65), `UNet` / `AttentionUNet` / `UNetResNet34` + 3 loss (`bce`, `dice`, `bce_dice`), `detection_ap` / `ap5095` / `P/R` / `top1_box_iou_summary` và cả 2 backbone Faster R-CNN.
  - `tests/test_dataset.py`: `audit` (tách riêng `bad_images` vs `bad_masks`, mask rỗng, pixel convention), `eda`, `prepare` (group split chống rò rỉ theo `lesion_id` + `manifest.csv` + `train/val/test.txt`), `SegmentationDataset` & `DetectionDataset` (có augmentation), baselines (`otsu_morphology`, `kmeans_lab`, `watershed`, `center_box`).
  - `tests/test_demo.py`: `integration` output chuẩn, render 4 panel + combined, nhánh rỗng / lỗi từng nhánh, validate ảnh đầu vào không hợp lệ, lưu file, dựng giao diện `gradio.Blocks`.
  - `tests/test_e2e_pipeline.py`: smoke test tích hợp toàn luồng (`audit` → `eda` → `prepare` → `baselines` → `train_seg` → `train_det` → `select_on_validation` → `eval_seg` → `eval_det` → `eval_unet_box` → `integration`) trên 12 cặp ảnh/mask tổng hợp với CUDA GPU.

## Môi trường kiểm thử thực tế

- Python `3.12.10` (`D:\Python\.venv\Scripts\python.exe`)
- PyTorch `2.15.0+cu134`, Torchvision `0.29.2+cu134`
- GPU: `NVIDIA GeForce RTX 2050` (`4.0 GB` VRAM)
- Gradio `5.50.0`

### Đo VRAM thực tế trên RTX 2050 (AMP bật)

- `UNet` 256×256, `batch_size=4`: **425.4 MB**
- `UNet` 320×320, `batch_size=4`: **649.3 MB**
- `Faster R-CNN MobileNetV3-Large-FPN` 512 px, `batch_size=2`: **442.6 MB**
- `Faster R-CNN ResNet50-FPN v2` 512 px, `batch_size=2`: **1125.3 MB**
- `Faster R-CNN ResNet50-FPN v2` 512 px, `batch_size=1`: **715.7 MB**

## Phạm vi chờ dữ liệu thật

- Thư mục `data/raw/masks/ISIC2018_Task1_Training_GroundTruth` đã có **2.594 mask PNG** (`{0, 255}`), `data/raw/images/` đang chờ tải xong ảnh `.jpg` để chạy thực nghiệm chính thức.
- Các chỉ số trên dữ liệu tổng hợp chỉ xác nhận phần mềm và pipeline chạy đúng, không thay thế kết quả trên ISIC 2018 thật.
