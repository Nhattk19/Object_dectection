# Plan CNN — Faster R-CNN ResNet-50-FPN trên VisDrone

Notebook Kaggle: `models/faster_rcnn_visdrone.ipynb`.

Hướng dẫn end-to-end local → Kaggle: `docs/cnn_training_pipeline.md`.

Training CLI dùng chung: `scripts/train_faster_rcnn.py`.

## Definition of done gần nhất

Pipeline được xem là sẵn sàng khi:

1. dataset adapter trả đúng image tensor `[C,H,W]`, box `xyxy`, label `1–10`;
2. visualization cho thấy box bám đúng object;
3. forward trả bốn loss hữu hạn;
4. backward smoke test chạy hết và lưu checkpoint;
5. F0 train/eval toàn bộ, có COCO `mAP50-95`, `AP50`, `AP75`, `AP_small`;
6. F1 giữ nguyên mọi cấu hình F0 và chỉ thêm augmentation;
7. có bảng so sánh F0/F1 trước khi tối ưu anchor hoặc tiling.

## Thứ tự chạy

### Bước 1 — Chuẩn bị COCO data

```bash
python src/preprocess_visdrone.py \
  --data-root data \
  --output-root data/processed/VisDrone \
  --splits train val test-dev
```

Faster R-CNN dùng:

```text
data/processed/VisDrone/
├── images/train
├── images/val
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

### Bước 2 — Smoke test local

```bash
python scripts/train_faster_rcnn.py \
  --data-root data/processed/VisDrone \
  --output-dir models/checkpoints \
  --experiment F0 \
  --smoke-test
```

Sau đó có thể chạy lại smoke test trong notebook Kaggle để xác nhận GPU và
đường dẫn Kaggle Input.

### Bước 3 — Smoke test notebook

Mở `models/faster_rcnn_visdrone.ipynb`:

```python
EXPERIMENT = "F0"
SMOKE_TEST = True
```

Chạy toàn bộ notebook. Đây chỉ là kiểm tra code, không báo cáo accuracy.

### Bước 4 — Train F0 baseline

Restart kernel, đổi:

```python
EXPERIMENT = "F0"
SMOKE_TEST = False
```

F0 gồm:

- Faster R-CNN ResNet-50-FPN pretrained COCO;
- model resize cạnh ngắn 800, cạnh dài tối đa 1333, giữ aspect ratio;
- anchor mặc định;
- tối đa 100 detection/ảnh theo baseline Torchvision;
- không random augmentation;
- SGD, batch size 2, 25 epochs;
- validation COCO sau mỗi epoch.

### Bước 5 — Train F1 augmentation

Restart kernel, chỉ đổi:

```python
EXPERIMENT = "F1"
SMOKE_TEST = False
```

F1 thêm controlled augmentation vào train. Split, seed, weights, resize,
optimizer, batch size và epochs phải giống F0. Validation không augmentation.

### Bước 6 — So sánh F0/F1

| Chỉ số | F0 | F1 | Chênh lệch |
|---|---:|---:|---:|
| mAP50-95 | | | |
| AP50 | | | |
| AP75 | | | |
| AP_small | | | |
| Train time | | | |
| Peak VRAM | | | |

Nếu F1 giảm, thử tắt lần lượt blur, vertical flip và giảm rotation từ `±10°`
xuống `±5°`. Không thêm Mosaic/tiling để “cứu” F1 trước khi tìm nguyên nhân.

## Các vòng sau

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| F0 | Baseline | Faster R-CNN cơ bản đạt gì? |
| F1 | F0 + augmentation | Augmentation có cải thiện generalization? |
| F2 | F0 + anchor 8–128 | Anchor nhỏ có tăng AP_small/recall? |
| F3 | F2 + tăng proposals/detections | Có giảm bỏ sót ảnh crowded? |
| F4 | F2/F3 + input 1024 | Resolution lớn có đáng chi phí GPU? |
| F5 | Tiling 640, overlap 20% | Tiling cải thiện small object bao nhiêu? |

F2 trở đi chỉ chạy sau khi F0 và F1 đã có kết quả. Vì F1 làm giảm mAP tổng và
AP_small trong lần chạy seed 42, F2 quay lại F0 rồi chỉ đổi anchor. Mỗi vòng
chỉ thay đổi một nhóm yếu tố để giữ ý nghĩa ablation.

## Local và Kaggle

Local phù hợp cho:

- unit test;
- đọc dataset và visualize;
- forward/backward một batch;
- smoke test.

Kaggle GPU phù hợp cho:

- F0/F1 toàn bộ dataset;
- input 1024/1280;
- tăng proposals;
- tiling và nhiều seed.

Checkpoint và history được lưu dưới `models/checkpoints/<experiment>/`; thư mục
này đã được `.gitignore`, nên phải tải từ Kaggle Output hoặc lưu thành Kaggle
Dataset nếu muốn giữ lại.

## Lưu ý quan trọng

- Label 0 luôn là background; VisDrone phải là `1–10`.
- Không resize ảnh thủ công thành hình vuông: model tự resize giữ tỷ lệ.
- Không augment validation/test.
- COCO AP mặc định giới hạn 100 detections/ảnh. Khi thử F3 với hơn 100
  detections, phải báo cáo thêm crowded recall ở ngưỡng tương ứng và vẫn giữ
  metric COCO chuẩn để so sánh F0/F1.
- Không dùng test-dev để lựa chọn hyperparameter.
