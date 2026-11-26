# Báo cáo kiểm tra project

## Kết quả rà soát

- Kiểm tra cấu trúc ZIP đầu vào: **PASS** (không phát hiện lỗi CRC hoặc lỗi giải nén).
- Biên dịch cú pháp Python cho `src/`, `tests/` và `app.py`: **PASS**.
- Bộ kiểm thử: **21 PASS, 0 FAIL** sau khi đồng bộ `tests/test_demo.py` với API demo hiện tại.
- CLI: kiểm tra lệnh trợ giúp và các nhóm lệnh: **PASS**.
- Healthcheck: PyTorch và Torchvision import thành công. Môi trường rà soát hiện tại chạy CPU, không có CUDA.
- Checkpoint: nạp thành công `unet_best.pt` và `fasterrcnn_best.pt` trên CPU.
- Demo tích hợp: chạy thử một ảnh mẫu, cả hai nhánh đều trả kết quả; tạo được mask, bounding box và trạng thái đối chiếu.
- Xuất báo cáo: kiểm thử tạo PNG, TXT và PDF thành công.

## Các điểm đã hiệu chỉnh

- CLI `prepare` hiện truyền đúng đường dẫn `--metadata` vào `prepare_splits`.
- CLI `prepare --no-phash` hiện thực sự tắt pHash; cấu hình mặc định vẫn được giữ nếu không truyền tùy chọn.
- Đồng bộ README với cấu hình thực tế: Faster R-CNN dùng ảnh 384 × 384, batch size mặc định 2.
- Bổ sung `reportlab` vào `requirements-demo.txt` để cài thư viện xuất PDF.
- PDF sử dụng font DejaVu Sans đi kèm Matplotlib khi có sẵn để hiển thị tiếng Việt.
- Đồng bộ kiểm thử demo với luồng `run_integration`, `format_status` và `save_result_files` hiện tại.

## Kết quả đánh giá đã lưu trong project

Các file `outputs/results/seg_test_metrics.json` và `outputs/results/det_test_metrics.json` ghi nhận kết quả trên Test Set gồm 388 mẫu. Các số liệu này được giữ nguyên, không huấn luyện lại trong lần đóng gói.

## Giới hạn xác minh

- Dataset ISIC đã được tách khỏi ZIP nên không chạy lại audit/EDA/prepare trên toàn bộ dataset trong lần rà soát này.
- Không chạy huấn luyện lại hoặc đánh giá lại toàn bộ Test Set.
- CUDA/GPU không khả dụng trong môi trường rà soát; chưa xác minh lại tốc độ và mức sử dụng VRAM trên RTX 2050.
- Demo được kiểm tra bằng checkpoint thật trên một ảnh mẫu và bằng dữ liệu tổng hợp trong bộ test; chưa kiểm tra giao diện trực tiếp trên máy Windows của người nộp.
- Ground-truth Detection được suy ra từ segmentation mask, không phải bounding box annotation độc lập do chuyên gia cung cấp.

## Trước khi nộp

1. Giải nén dataset ISIC 2018 Task 1 vào `data/raw/images/` và `data/raw/masks/` theo README.
2. Kiểm tra môi trường bằng `kiem_tra_moi_truong.cmd`.
3. Nếu muốn xác minh lại dữ liệu trên máy có dataset, chạy `run_audit.cmd`.
4. Chạy `run_tests.cmd` để kiểm tra source trên môi trường Windows/Python của project.
5. Chạy `app.py` để mở Gradio demo và thử xuất báo cáo PDF.

Không chạy smoke script mẫu trên thư mục dữ liệu/kết quả chính vì script kiểu này có thể tạo lại split hoặc checkpoint. Pipeline huấn luyện đầy đủ chỉ cần chạy khi muốn tái huấn luyện từ đầu.
