# YOLO Vision Lab

Streamlit landing page and inference demo for the local Ultralytics checkpoint:

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

## Suy luận

Web chạy YOLO một lần trên toàn bộ ảnh. Giao diện có ba thanh trượt:

- Confidence, mặc định `0.04`.
- Số bbox tối đa, mặc định `300`.
- NMS IoU, mặc định `0.30`.

YOLO áp dụng confidence filter và NMS tích hợp trước khi trả kết quả. Mỗi bbox
hiển thị tên lớp và phần trăm confidence.
