# Báo cáo kiểm tra source

## Kết quả kiểm tra trong lần rà soát này

- Bộ kiểm thử: **20 PASS, 0 FAIL**.
- Môi trường rà soát hiện tại: Python runtime của phiên kiểm tra, PyTorch CPU; CUDA không khả dụng.
- Đã chạy kiểm thử unit/integration trên dữ liệu tổng hợp.
- Đã kiểm tra cú pháp Python cho toàn bộ source và tests.

## Các điểm đã hiệu chỉnh

- Không fallback sang chia image-level khi có nhóm trùng nhưng chỉ có dưới 3 group; dừng với thông báo rõ ràng để tránh rò rỉ giữa train/validation/test.
- Detection mặc định chỉ dùng một bounding box từ thành phần trắng lớn nhất; đã thêm test cho mask có nhiều vùng trắng rời rạc.
- Không tự tạo bounding box giả 1x1 khi mask không còn vùng hợp lệ.
- Metric Detection ghép prediction với ground-truth chưa được ghép, tránh bỏ sót true positive khi nhiều box chồng lấn.
- Ngưỡng segmentation, hậu xử lý và score threshold được chọn từ `val_selection.json` sẽ được dùng cho đánh giá và demo nếu có; tham số truyền trực tiếp khi gọi evaluate vẫn được ưu tiên.
- Bổ sung test cho trường hợp group split không đủ nhóm và matching nhiều ground-truth.

## Giới hạn xác minh

- ZIP không chứa dataset ISIC; các kiểm thử hiện tại dùng dữ liệu tổng hợp.
- Chưa chạy audit, EDA, huấn luyện, đánh giá hoặc demo trên toàn bộ ảnh ISIC thật trong lần rà soát này.
- CUDA/GPU không khả dụng trong môi trường rà soát này, vì vậy chưa xác nhận lại huấn luyện trên RTX 2050.
- Các thông số VRAM từng ghi trong báo cáo cũ không được đo lại trong lần rà soát này và không được xem là cam kết hiệu năng.
- Ground-truth Detection được suy ra từ segmentation mask; đây không phải annotation bounding box độc lập do chuyên gia cung cấp.

## Trước khi nộp

1. Đặt đúng ảnh và mask ISIC 2018 vào `data/raw/images/` và `data/raw/masks/`.
2. Chạy `kiem_tra_moi_truong.cmd`, sau đó `run_audit.cmd`.
3. Kiểm tra `reports/audit.txt`, `reports/audit.json` và `reports/eda_report.md`.
4. Chạy `chay_thu_100_anh.cmd` để smoke test.
5. Chỉ khi smoke test thành công mới chạy `chay_toan_bo.cmd`.
6. Giữ nguyên Test split cho đánh giá cuối; không dùng kết quả Test để chọn tham số.
