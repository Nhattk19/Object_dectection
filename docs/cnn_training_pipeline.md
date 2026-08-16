# Pipeline Faster R-CNN: local → Kaggle

Branch làm việc: `cnn-faster-rcnn-pipeline`.

Mục tiêu của pipeline là kiểm tra dữ liệu và code ở local, sau đó fine-tune
Faster R-CNN ResNet-50-FPN trên Kaggle GPU. Không dùng kết quả smoke test để
báo cáo độ chính xác.

## 1. Chuẩn bị ở local

### Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-cnn.txt
```

### Kiểm tra code

```bash
pytest -q
```

### Tiền xử lý VisDrone

Dữ liệu gốc có thể lồng thêm một thư mục cùng tên; script tự tìm đúng split.

```bash
python src/preprocess_visdrone.py \
  --data-root data \
  --output-root data/processed/VisDrone \
  --splits train val test-dev \
  --image-mode hardlink
```

Một lần chạy tạo cả hai format:

```text
data/processed/VisDrone/
├── images/{train,val,test-dev}/
├── labels/{train,val,test-dev}/              # YOLO
├── annotations/instances_{split}.json        # COCO/Faster R-CNN
├── preprocessing_report.json
└── visdrone_yolo.yaml
```

Kiểm tra pipeline Faster R-CNN trên CPU/GPU local bằng hai batch:

```bash
python scripts/train_faster_rcnn.py \
  --data-root data/processed/VisDrone \
  --output-dir models/checkpoints \
  --experiment F0 \
  --smoke-test
```

Smoke test phải tạo được `models/checkpoints/f0/last.pth`, `history.csv` và
`summary.json`. Xóa hoặc bỏ qua checkpoint smoke khi chuyển sang full run;
không resume full training từ smoke checkpoint vì smoke không dùng pretrained
COCO weights.

## 2. Đưa code và dữ liệu lên Kaggle

Có hai cách:

### Cách A — repository + VisDrone gốc (khuyến nghị)

1. Push branch lên remote hoặc tạo một Kaggle Dataset chứa repository.
2. Add repository và dataset VisDrone gốc vào Notebook Input.
3. Notebook tự chuyển train/val sang COCO trong `/kaggle/working` và dùng
   symlink tới ảnh trong `/kaggle/input`, tránh copy thêm toàn bộ ảnh.

### Cách B — upload dữ liệu đã processed

Tạo Kaggle Dataset từ `data/processed/VisDrone`. Notebook tự tìm thư mục có
`annotations/instances_train.json`; nếu không tìm thấy, đặt trực tiếp biến
`PROCESSED_DATA_ROOT`.

Không commit ảnh, annotation sinh ra hoặc checkpoint vào Git.

## 3. Cấu hình Kaggle

Mở `models/faster_rcnn_visdrone.ipynb`, tạo Kaggle Notebook từ file này và đặt:

- Accelerator: GPU;
- Internet: On, để tải pretrained COCO weights lần đầu;
- add repository và VisDrone vào Input.

Trong cell cấu hình:

```python
EXPERIMENT = "F0"
SMOKE_TEST = True
EPOCHS = 25
BATCH_SIZE = 2
WORKERS = 2
SEED = 42
```

Nếu auto-discovery không tìm đúng đường dẫn:

```python
PROJECT_ROOT_OVERRIDE = "/kaggle/input/<repo-dataset>/Object_dectection"
RAW_DATA_ROOT = "/kaggle/input/<visdrone-dataset>"
# Hoặc:
PROCESSED_DATA_ROOT = "/kaggle/input/<processed-dataset>/VisDrone"
```

Chạy **Run All** với smoke test trước. Khi thành công:

1. đổi `SMOKE_TEST=False`;
2. để `RESUME_FROM=None`;
3. restart session;
4. chạy **Run All** để train F0 đủ 25 epoch.

Artifact được ghi vào:

```text
/kaggle/working/faster_rcnn_runs/f0/
├── best.pth
├── last.pth
├── history.csv
├── learning_curves.png
├── config.json
└── summary.json
```

Chọn **Save Version** để giữ Output sau khi session kết thúc.

## 4. Resume sau khi Kaggle ngắt session

Add Output của lần chạy trước làm Kaggle Input, rồi đặt:

```python
SMOKE_TEST = False
RESUME_FROM = "/kaggle/input/<previous-output>/f0/last.pth"
```

Checkpoint khôi phục model, optimizer, LR scheduler, AMP scaler, epoch, history
và best mAP. `EPOCHS` là tổng số epoch muốn đạt, không phải số epoch chạy thêm.

## 5. Chạy F1 và so sánh

Sau khi F0 hoàn tất, restart session và chạy độc lập:

```python
EXPERIMENT = "F1"
SMOKE_TEST = False
RESUME_FROM = None
```

F1 giữ nguyên split, seed, pretrained weights, optimizer, batch size và số
epoch; chỉ thêm augmentation. Không resume F1 từ checkpoint F0.

So sánh `summary.json` và `history.csv` của hai run theo:

- mAP50-95 (`map`);
- AP50 (`map_50`);
- AP75 (`map_75`);
- AP small (`map_small`);
- thời gian mỗi epoch và peak VRAM.

Chỉ dùng validation để chọn cấu hình. Test-dev được dùng sau khi đã chốt model.

## 6. Chạy F2 small anchors

Sau khi F1 không cải thiện mAP/AP_small so với F0, chạy F2 từ cùng pretrained
COCO initialization như F0 và chỉ thay anchor FPN thành `8,16,32,64,128`:

```python
EXPERIMENT = "F2"
RUN_SMOKE_FIRST = True
TRAIN_FULL = True
EPOCHS = 25
RESUME_FROM = None
```

F2 không dùng augmentation và không resume checkpoint F0/F1. Output được lưu
tại `/kaggle/working/faster_rcnn_runs/f2/`.

## 7. Chạy F3 crowded proposals

F2 tăng AP_small nhưng giảm mAP tổng và recall, nên F3 quay lại F0 và chỉ tăng
RPN proposal limits: train `4000→2000` và validation `2000→1000` trước/sau
NMS. Anchor và giới hạn COCO 100 detections/ảnh vẫn giữ như F0.

```python
EXPERIMENT = "F3"
RUN_SMOKE_FIRST = True
TRAIN_FULL = True
EPOCHS = 25
RESUME_FROM = None
```

F3 không resume F0/F1/F2. Output được lưu tại
`/kaggle/working/faster_rcnn_runs/f3/`.

## 8. Lệnh tương đương notebook

Nếu muốn chạy trực tiếp trong Kaggle terminal:

```bash
python scripts/train_faster_rcnn.py \
  --data-root /kaggle/working/visdrone_processed \
  --output-dir /kaggle/working/faster_rcnn_runs \
  --experiment F0 \
  --epochs 25 \
  --batch-size 2 \
  --workers 2
```

Resume từ checkpoint:

```bash
python scripts/train_faster_rcnn.py \
  --data-root /kaggle/working/visdrone_processed \
  --output-dir /kaggle/working/faster_rcnn_runs \
  --experiment F0 \
  --epochs 25 \
  --resume /kaggle/input/<previous-output>/f0/last.pth
```
