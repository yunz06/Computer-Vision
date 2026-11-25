# Báo cáo kiểm thử source

## Kết quả đã kiểm tra

- Biên dịch toàn bộ `src` và `tests`: PASS.
- Unit tests: **11 PASS**.
- CLI help, audit thư mục dữ liệu rỗng và tạo split rỗng: PASS.
- Kiểm thử demo với mô hình giả lập: PASS; đã kiểm tra overlay, mask, bounding box, Box–Mask IoU và lưu file.
- Kiểm thử tích hợp bằng 12 cặp ảnh/mask tổng hợp: PASS cho audit, ghép cặp, split, Otsu + Morphology, K-means Lab, U-Net train/evaluate và Faster R-CNN MobileNet train/evaluate.
- Hai kiến trúc Faster R-CNN ResNet50-FPN v2 và MobileNetV3-FPN đều khởi tạo và inference được với 2 lớp trên ảnh 128×128.
- Kiểm thử tích hợp tại kích thước cấu hình: U-Net 256×256 và Faster R-CNN MobileNetV3-FPN 512×512, train 1 epoch và evaluate: PASS trên dữ liệu tổng hợp.

## Môi trường kiểm thử

- Python 3.13.5.
- PyTorch 2.10.0 CPU.
- Torchvision 0.25.0 CPU.
- Container không có GPU; không đo được VRAM hoặc tốc độ trên RTX 2050.

## Phạm vi chưa thể xác nhận

- Chưa chạy trên ảnh/mask ISIC 2018 thật vì dữ liệu đang được tải về máy người dùng và môi trường build không truy cập được ổ đĩa đó.
- Chưa xác nhận pretrained weights tải được trong môi trường offline.
- Chưa xác nhận thời gian huấn luyện, VRAM và chất lượng mô hình trên RTX 2050 4 GB.
- Các chỉ số trong smoke test tổng hợp chỉ chứng minh pipeline chạy được; **không phải kết quả nghiên cứu** và không được đưa vào báo cáo như kết quả ISIC.

## Cấu hình smoke test

- 12 cặp ảnh/mask tổng hợp tạm thời, kích thước 72×96.
- U-Net: 1 epoch, cả ảnh 64×64 và 256×256.
- Faster R-CNN MobileNetV3-FPN: 1 epoch, cả ảnh 128×128 và 512×512, không pretrained.
- Seed cố định 42.

## Kết luận

Source đã vượt qua kiểm thử phần mềm và smoke test tích hợp trên dữ liệu tổng hợp. Cần chạy audit và smoke test lại trên ISIC 2018 thật sau khi người dùng giải nén dữ liệu; chỉ sau đó mới có thể xác nhận hoàn chỉnh về mặt thực nghiệm.
