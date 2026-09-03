# YOLO11m + SAHI Vision Lab

Streamlit landing page and SAHI sliced-inference demo for the local Ultralytics checkpoint:

```text
models/yolo/best.pt
```

## Chạy ứng dụng

Từ thư mục gốc của dự án:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run web/app.py
```

Mặc định, Streamlit mở ứng dụng tại `http://localhost:8501`.

## Suy luận SAHI

Web dùng đúng cấu hình của notebook `visdrone-yolo11m-sahi.ipynb`:

- Tile `640x640`, overlap `20%`.
- SAHI chạy các tile và một lượt dự đoán chuẩn trên toàn ảnh (`perform_standard_pred=True`).
- Các dự đoán được hợp nhất bằng class-aware NMS với metric IoU.

Giao diện có ba thanh trượt:

- Confidence, mặc định `0.04`.
- Số bbox tối đa sau khi hợp nhất, mặc định `500`.
- Merge NMS IoU, mặc định `0.50`.

Mỗi tile được YOLO lọc riêng trước khi SAHI quy đổi bbox về tọa độ ảnh gốc.
Mỗi bbox cuối cùng hiển thị tên lớp và phần trăm confidence.
