# BÁO CÁO KỸ THUẬT

## Huấn luyện và đánh giá Faster R-CNN trên bộ dữ liệu VisDrone2019-DET

**Nhánh phát triển:** `cnn-faster-rcnn-pipeline`

**Môi trường huấn luyện:** Kaggle Notebook, GPU NVIDIA Tesla T4

**Phạm vi thí nghiệm:** F0–F5

**Ngày cập nhật:** 19/08/2026

---

## Tóm tắt

Báo cáo trình bày pipeline huấn luyện Faster R-CNN ResNet-50-FPN cho bài toán
phát hiện 10 lớp đối tượng trên ảnh chụp từ drone của VisDrone2019-DET. Các
thí nghiệm F0–F5 lần lượt khảo sát baseline, data augmentation, anchors cho
vật thể nhỏ, tăng số Region Proposal Network (RPN) proposals, Faster R-CNN V2
và tăng độ phân giải đầu vào.

Dữ liệu được kiểm tra và chuyển từ định dạng VisDrone sang COCO trước khi
train. Tất cả thí nghiệm dùng chung 6.471 ảnh train và 548 ảnh validation; chỉ
F1 bật augmentation online. F0–F4 ban đầu chọn checkpoint bằng COCO mAP, sau
đó được chạy inference lại bằng evaluator tương thích VisDrone2018-DET-toolkit.
Từ F5, VisDrone AP được tính sau mỗi epoch và được dùng trực tiếp để chọn
`best.pth`.

Trong các checkpoint full-training đã đánh giá hợp lệ, F5 tốt nhất với
VisDrone AP 28,42%, AP50 51,94%, AP75 26,93%, AR@100 35,71% và AR@500 46,17%.
So với F4, F5 tăng 1,46 điểm AP và 2,60 điểm AR@500. Checkpoint tốt nhất của
F5 nằm ở epoch 5; không được thay bằng `last.pth` epoch 25. Kết quả official
của F0 hiện không hợp lệ do notebook đã lấy nhầm checkpoint smoke-test epoch
1; cần đánh giá lại checkpoint F0 full-training epoch 17.

## 1. Mục tiêu và phạm vi

Mục tiêu kỹ thuật của pipeline gồm:

1. xây dựng bộ dữ liệu COCO hợp lệ từ annotation VisDrone;
2. fine-tune Faster R-CNN bằng pretrained weights COCO;
3. đo tác động của từng thay đổi bằng chuỗi thí nghiệm F0–F5;
4. đánh giá trên toàn bộ validation set bằng quy trình tương thích toolkit
   VisDrone DET;
5. lưu đầy đủ config, lịch sử train và checkpoint để có thể resume hoặc tái
   lập thí nghiệm trên Kaggle.

Đây là bài toán object detection, không phải image classification. Mỗi đầu ra
gồm bounding box, nhãn lớp và confidence score.

## 2. Dữ liệu

### 2.1. Các lớp đối tượng

Pipeline giữ nguyên category ID 1–10 của VisDrone. Label 0 được Faster R-CNN
dành cho background.

| ID | Tên lớp | ID | Tên lớp |
|---:|---|---:|---|
| 1 | pedestrian | 6 | truck |
| 2 | people | 7 | tricycle |
| 3 | bicycle | 8 | awning-tricycle |
| 4 | car | 9 | bus |
| 5 | van | 10 | motor |

Category 0 là ignored region và category 11 là `others`; hai loại này không
được dùng làm nhãn train.

### 2.2. Thống kê sau preprocessing

Số liệu dưới đây lấy từ `data/processed/VisDrone/preprocessing_report.json`.

| Split | Ảnh | Annotation gốc | Box giữ lại | Dòng bị loại do score | Box không dương | Lỗi/mất dữ liệu |
|---|---:|---:|---:|---:|---:|---:|
| Train | 6.471 | 353.550 | 343.204 | 10.345 | 1 | 0 |
| Validation | 548 | 40.169 | 38.759 | 1.410 | 0 | 0 |
| Test-dev | 1.610 | 77.547 | 75.102 | 2.445 | 0 | 0 |

Không có ảnh không đọc được, annotation thiếu, ảnh thiếu hoặc dòng annotation
sai định dạng trong ba split đã xử lý.

## 3. Pipeline preprocessing

### 3.1. Preprocessing offline

Preprocessing được chạy một lần trước khi upload dữ liệu lên Kaggle và được
dùng chung cho F0–F5:

```bash
python src/preprocess_visdrone.py \
  --data-root data \
  --output-root data/processed/VisDrone \
  --splits train val test-dev \
  --image-mode hardlink
```

Quy trình xử lý:

1. Tìm đúng thư mục `VisDrone2019-DET-{split}`, kể cả trường hợp file giải nén
   tạo nhiều cấp thư mục trùng tên.
2. Ghép ảnh với annotation `.txt` theo stem của tên file và kiểm tra tính đọc
   được của ảnh.
3. Đọc mỗi annotation theo cấu trúc:

   ```text
   x, y, width, height, score, category, truncation, occlusion
   ```

4. Loại annotation có `score <= 0`, category ngoài 1–10, width/height không
   dương hoặc box nằm hoàn toàn ngoài ảnh.
5. Clip box cắt qua biên về phạm vi ảnh; sau clip box phải có width và height
   tối thiểu 1 px.
6. Xuất COCO JSON, giữ category ID 1–10, lưu bbox dạng pixel `xywh`, `area` và
   `iscrowd=0`.
7. Đồng thời xuất YOLO labels và YAML để phục vụ thí nghiệm khác. Faster R-CNN
   trong báo cáo này chỉ đọc COCO JSON.
8. Hardlink ảnh sang processed tree; nếu filesystem không hỗ trợ thì copy.
9. Ghi báo cáo kiểm tra vào `preprocessing_report.json`.

Cấu trúc dữ liệu đầu ra:

```text
data/processed/VisDrone/
├── images/
│   ├── train/
│   ├── val/
│   └── test-dev/
├── labels/
│   ├── train/
│   ├── val/
│   └── test-dev/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test-dev.json
├── preprocessing_report.json
└── visdrone_yolo.yaml
```

Preprocessing offline không resize, normalize hoặc augment ảnh. Pixel ảnh gốc
không bị thay đổi và không có processed dataset riêng cho từng F.

### 3.2. Tiền xử lý runtime

Mỗi mẫu đi qua DataLoader theo thứ tự:

1. mở ảnh bằng PIL và chuyển sang RGB;
2. đọc bbox COCO `xywh`, đổi sang `xyxy`;
3. tạo `labels`, `area`, `iscrowd` và `image_id`;
4. chạy augmentation nếu experiment là F1;
5. đổi ảnh từ `uint8 [0,255]` sang tensor `float32 [0,1]`;
6. resize giữ nguyên tỉ lệ bằng transform bên trong Faster R-CNN;
7. normalize theo pretrained ImageNet statistics.

| Tham số normalize | R | G | B |
|---|---:|---:|---:|
| Mean | 0,485 | 0,456 | 0,406 |
| Standard deviation | 0,229 | 0,224 | 0,225 |

F0–F4 dùng cạnh ngắn 800 px và giới hạn cạnh dài 1.333 px. F5 dùng 896 và
1.493 px. Validation và inference không dùng random augmentation.

### 3.3. Xử lý ignored regions

Ignored regions được loại khỏi nhãn train. Tuy nhiên khi đánh giá VisDrone,
pipeline đọc lại raw annotation của validation set và tạo mask ignored region.
Detection có từ 50% diện tích trở lên nằm trong vùng ignore được loại trước
khi matching. Vì vậy evaluator cần cả processed data và raw
`VisDrone2019-DET-val`.

## 4. Mô hình và cấu hình huấn luyện

### 4.1. Kiến trúc

Các mô hình dùng Faster R-CNN với backbone ResNet-50 và Feature Pyramid
Network (FPN). Pipeline gồm:

- backbone ResNet-50 trích xuất đặc trưng;
- FPN tạo đặc trưng đa tỉ lệ;
- RPN sinh object proposals;
- RoI Align trích xuất đặc trưng từng proposal;
- classification head dự đoán 11 nhãn gồm background và 10 lớp VisDrone;
- box regression head hiệu chỉnh bounding box.

F0–F3 dùng `fasterrcnn_resnet50_fpn`. F4–F5 dùng
`fasterrcnn_resnet50_fpn_v2`. Classification head pretrained COCO được thay
bằng `FastRCNNPredictor` 11 lớp.

### 4.2. Hyperparameters chung

| Nhóm | Thông số | Giá trị |
|---|---|---:|
| Khởi tạo | Pretrained weights | COCO DEFAULT |
| Số lớp | Background + VisDrone | 11 |
| Epoch | Tổng số epoch | 25 |
| Optimizer | Thuật toán | SGD |
| Optimizer | Learning rate | 0,0025 |
| Optimizer | Momentum | 0,9 |
| Optimizer | Weight decay | 0,0005 |
| Scheduler | Loại | MultiStepLR |
| Scheduler | Milestones | epoch 15 và 20 |
| Scheduler | Gamma | 0,1 |
| Reproducibility | Seed | 42 |
| DataLoader | Workers | 2 |
| Precision | Mixed precision | AMP |
| Validation loader | Batch size | 1 |
| Thiết bị | Accelerator | CUDA, Tesla T4 |

Learning rate theo lịch:

| Khoảng epoch | Learning rate |
|---|---:|
| 1–14 | 0,0025 |
| 15–19 | 0,00025 |
| 20–25 | 0,000025 |

Hàm loss khi train là tổng của bốn thành phần:

```text
L = L_classifier + L_box_reg + L_objectness + L_rpn_box_reg
```

Mỗi experiment mới khởi tạo độc lập từ pretrained COCO. Không resume F1 từ
F0, F2 từ F1 hoặc F4 từ F3. `last.pth` chỉ dùng để tiếp tục đúng experiment
đang bị gián đoạn.

### 4.3. Artifact huấn luyện

Mỗi run tạo:

```text
faster_rcnn_runs/<experiment>/
├── best.pth
├── last.pth
├── config.json
├── history.csv
├── learning_curves.png
└── summary.json
```

Checkpoint chứa model, optimizer, scheduler, AMP scaler, epoch, best metric,
history và config. Do đó có thể resume qua một Kaggle session khác nếu Add
Output cũ làm Input và trỏ `RESUME_FROM` tới `last.pth`.

## 5. Thiết kế thí nghiệm F0–F5

### 5.1. Ma trận cấu hình

| F | Model | Augmentation | Anchors | RPN proposals | Resize | Batch vật lý / tích lũy | Metric chọn best |
|---|---|---|---|---|---|---|---|
| F0 | V1 | Không | Mặc định | Mặc định | 800/1333 | 2 / 1 | COCO mAP |
| F1 | V1 | Có | Mặc định | Mặc định | 800/1333 | 2 / 1 | COCO mAP |
| F2 | V1 | Không | 8–128 px | Mặc định | 800/1333 | 2 / 1 | COCO mAP |
| F3 | V1 | Không | Mặc định | Crowded | 800/1333 | 2 / 1 | COCO mAP |
| F4 | V2 | Không | Mặc định | Crowded | 800/1333 | 1 / 2 | COCO mAP |
| F5 | V2 | Không | Mặc định | Crowded | 896/1493 | 1 / 2 | VisDrone AP |

Effective batch size của mọi full run bằng 2. Chỉ F1 có data augmentation.
Anchors, proposals, model version và resize là thay đổi cấu hình model, không
phải augmentation.

### 5.2. F0 — baseline

F0 dùng Faster R-CNN V1, anchors và RPN proposal limits mặc định, không
augmentation, resize 800/1333. Mục tiêu là tạo baseline để đánh giá từng thay
đổi sau đó.

Checkpoint full-training tốt nhất theo COCO mAP nằm ở epoch 17:

| mAP | AP50 | AP75 | AP-small | AP-medium | AP-large | AR@100 |
|---:|---:|---:|---:|---:|---:|---:|
| 22,42 | 38,74 | 22,48 | 14,06 | 32,29 | 41,53 | 32,05 |

AP-small 14,06% cho thấy vật thể nhỏ là bottleneck chính. F0 official hiện
phải đánh giá lại vì lần đánh giá trước dùng nhầm smoke-test checkpoint.

### 5.3. F1 — augmentation bbox-safe

F1 giữ nguyên F0 và chỉ thêm augmentation online trên train set:

| Phép biến đổi | Xác suất | Miền tham số |
|---|---:|---|
| Horizontal flip | 0,5 | Lật trái–phải |
| Vertical flip | 0,2 | Lật trên–dưới |
| Affine | 0,6 | Xoay ±10°, dịch ±5%, scale 0,9–1,1 |
| Brightness | 0,8 cùng photometric step | Hệ số 0,85–1,15 |
| Contrast | 0,8 cùng photometric step | Hệ số 0,85–1,15 |
| Gaussian blur | 0,1 | Radius 0,1–1,0 |

Các phép geometric biến đổi đồng thời ảnh và bốn góc bbox. Sau affine, bbox
được clip về biên ảnh; chỉ giữ box có width/height tối thiểu 2 px và còn ít
nhất 30% diện tích kỳ vọng sau scale.

F1 đạt best epoch 22. So với F0 theo cùng COCO evaluator, AP50 tăng 0,85 điểm
nhưng mAP giảm 0,43, AP75 giảm 0,65, AP-small giảm 0,12 và AR@100 giảm 0,92
điểm. Gói augmentation này không cải thiện tổng thể. Vertical flip, rotation
và blur có thể tạo phân bố khác ảnh drone thật hoặc làm giảm chất lượng box
rất nhỏ.

### 5.4. F2 — small anchors

F2 quay lại baseline F0, không dùng augmentation và thay anchor sizes của năm
tầng FPN:

```text
sizes         = ((8,), (16,), (32,), (64,), (128,))
aspect_ratios = ((0.5, 1.0, 2.0),) × 5
```

Mục tiêu là tăng độ khớp ban đầu giữa anchors với pedestrian và vehicle nhỏ.
F2 đạt best epoch 19. So với F0, AP-small tăng 0,46 điểm nhưng mAP giảm 0,74,
AP75 giảm 1,21 và AR@100 giảm 1,22 điểm. Small anchors giúp một phần cho vật
thể nhỏ nhưng gây trade-off bất lợi cho chất lượng tổng thể.

### 5.5. F3 — crowded RPN proposals

F3 quay lại F0, giữ default anchors và không augmentation. Thay đổi duy nhất
là tăng giới hạn RPN proposals:

| Giai đoạn | Pre-NMS top-N | Post-NMS top-N |
|---|---:|---:|
| Train | 4.000 | 2.000 |
| Validation/inference | 2.000 | 1.000 |

Mục tiêu là tránh loại candidate quá sớm trong ảnh crowded. F3 đạt best epoch
16. So với F0, mAP tăng 0,42, AP50 tăng 0,58, AP75 tăng 0,55 và AP-small tăng
0,76 điểm; AR@100 gần như không đổi, giảm 0,07 điểm. Đây là thay đổi có lợi và
khá đồng đều.

### 5.6. F4 — Faster R-CNN V2

F4 giữ default anchors, crowded proposals, không augmentation và resize
800/1333 của F3, sau đó đổi model từ V1 sang V2. Do V2 tốn bộ nhớ hơn, batch
vật lý giảm từ 2 xuống 1 và accumulation tăng lên 2, giữ effective batch size
bằng 2.

F4 đạt best epoch 16. Trên VisDrone evaluator, F4 tăng so với F3:

| AP | AP50 | AP75 | AR@10 | AR@100 | AR@500 |
|---:|---:|---:|---:|---:|---:|
| +0,72 | −0,35 | +1,50 | +0,40 | +1,19 | +0,88 |

Mức tăng lớn ở AP75 và recall cho thấy V2 cải thiện localization và khả năng
giữ true positive, thay vì chỉ tạo thêm detection dễ ở IoU 0,50.

### 5.7. F5 — tăng độ phân giải và đánh giá trực tiếp bằng VisDrone

F5 giữ cấu hình model của F4 và thay đổi:

1. resize từ 800/1333 lên 896/1493, tăng khoảng 25% số pixel;
2. giới hạn đầu ra validation là 500 detection/ảnh;
3. score threshold validation là 0,001;
4. chạy VisDrone evaluator sau mỗi epoch;
5. chọn `best.pth` theo `visdrone_ap` thay vì COCO mAP.

Cấu hình 1024/1707 ban đầu gây CUDA OOM trên Tesla T4 khi gặp ảnh có số lượng
ground truth/anchor lớn. Kích thước 896/1493 được chọn làm mức cân bằng giữa
chi tiết của vật thể nhỏ và VRAM. F5 dùng batch 1, accumulation 2 và cần raw
validation annotations để xử lý ignored regions.

F5 hoàn tất 25 epoch trong 9,36 giờ, peak VRAM 11,28 GB. `best.pth` được chọn
đúng theo VisDrone AP tại epoch 5:

| Epoch | AP | AP50 | AP75 | AR@1 | AR@10 | AR@100 | AR@500 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **5, best** | **28,42** | **51,94** | **26,93** | 0,95 | 6,53 | **35,71** | **46,17** |
| 25, last | 26,15 | 47,46 | 25,05 | 0,91 | 6,70 | 34,16 | 41,72 |

Train loss giảm đều từ 1,1165 xuống 0,4009, nhưng validation AP không tăng
đơn điệu. AP đạt đỉnh sớm ở epoch 5, có đỉnh phụ 28,28% tại epoch 17 rồi giảm
về 26,15% ở epoch 25. Điều này cho thấy tiếp tục tối ưu train loss không đồng
nghĩa tăng khả năng tổng quát hóa. Cơ chế chọn `best.pth` theo VisDrone AP đã
hoạt động đúng và là bắt buộc đối với F5.

Thư mục `f5/visdrone_predictions` được ghi đè sau mỗi lần validation nên bản
đang lưu tương ứng epoch 25, không phải best epoch 5. Để lưu bộ prediction
chính thức của F5, cần chạy `evaluate_visdrone_checkpoint.py` hoặc notebook
evaluation với `f5/best.pth` và một output directory riêng.

## 6. Giao thức đánh giá

### 6.1. COCO evaluator trong F0–F4

Trong quá trình train F0–F4, validation dùng pycocotools với tối đa 100
detections/ảnh. `best.pth` được chọn theo COCO mAP@[0.50:0.95]. Kết quả này
phù hợp để so sánh các thí nghiệm được train cùng pipeline, nhưng chưa phản
ánh đầy đủ đặc điểm crowded và ignored regions của VisDrone.

### 6.2. Evaluator tương thích VisDrone2018-DET-toolkit

Các checkpoint được chạy inference lại trên đủ 548 ảnh validation với:

| Tham số | Giá trị |
|---|---:|
| IoU thresholds | 0,50–0,95, bước 0,05 |
| Max detections | 1, 10, 100 và 500 |
| Score threshold inference | 0,001 |
| Số lớp | 10 |
| Số ảnh | 548 |
| Ignore overlap threshold | 50% diện tích detection |

Prediction được xuất thành một TXT cho mỗi ảnh theo định dạng VisDrone. AP
được tích phân trên precision–recall envelope, matching và class weighting
theo logic toolkit. Metric JSON có miền 0–1; báo cáo nhân 100 để biểu diễn
phần trăm.

Đây là Python port tương thích logic VisDrone2018-DET-toolkit v1.0.4, không
phải lần chạy MATLAB gốc. Prediction TXT được giữ lại để có thể kiểm chứng
chéo bằng repository chính thức.

## 7. Kết quả thực nghiệm

### 7.1. Kết quả COCO tại best checkpoint

| F | Best epoch | mAP | AP50 | AP75 | AP-small | AP-medium | AP-large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 17 | 22,42 | 38,74 | 22,48 | 14,06 | 32,29 | 41,53 | 32,05 |
| F1 | 22 | 21,99 | **39,60** | 21,83 | 13,94 | 31,29 | **44,20** | 31,14 |
| F2 | 19 | 21,67 | 38,45 | 21,27 | 14,52 | 30,16 | 40,44 | 30,84 |
| F3 | 16 | 22,84 | 39,32 | 23,03 | 14,82 | **32,43** | 42,00 | 31,99 |
| F4 | 16 | **23,44** | 39,03 | **24,11** | **16,30** | 32,19 | 38,97 | **33,15** |

Theo COCO evaluator, F4 dẫn đầu mAP, AP75, AP-small và AR@100.

### 7.2. Kết quả VisDrone official-compatible

| Hạng | F | Epoch | AP | AP50 | AP75 | AR@1 | AR@10 | AR@100 | AR@500 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **F5** | 5 | **28,42** | **51,94** | **26,93** | **0,95** | 6,53 | **35,71** | **46,17** |
| 2 | F4 | 16 | 26,96 | 48,60 | 26,00 | 0,80 | **6,56** | 34,72 | 43,57 |
| 3 | F3 | 16 | 26,24 | 48,95 | 24,50 | 0,78 | 6,17 | 33,52 | 42,69 |
| 4 | F2 | 19 | 25,80 | 49,86 | 23,21 | 0,64 | 5,66 | 32,62 | 43,36 |
| 5 | F1 | 22 | 24,11 | 46,54 | 22,29 | 0,52 | 5,55 | 31,89 | 38,85 |
| Loại | F0 smoke | 1 | 0,001 | 0,007 | 0,000 | 0,005 | 0,017 | 0,074 | 0,233 |

F0 bị loại khỏi xếp hạng. File được evaluator sử dụng là checkpoint epoch 1,
`smoke_test=true`, `pretrained=false`, không phải checkpoint full-training
epoch 17. Model smoke xuất đúng 500 detection/ảnh và có AP gần 0, phù hợp với
một model chưa train. Không được dùng hàng F0 smoke trong báo cáo kết quả cuối
cùng hoặc để tính improvement official.

### 7.3. Tài nguyên huấn luyện

| F | Thời gian epoch tại best checkpoint | Peak VRAM | Batch vật lý |
|---|---:|---:|---:|
| F0 | 18,1 phút | 10,50 GB | 2 |
| F1 | 19,2 phút | 10,45 GB | 2 |
| F2 | 18,3 phút | 10,50 GB | 2 |
| F3 | 18,5 phút | 10,50 GB | 2 |
| F4 | 19,9 phút | 9,10 GB | 1 |
| F5 | 22,5 phút | 11,28 GB | 1 |

F5 tăng khoảng 13% thời gian mỗi epoch và 2,18 GB peak VRAM so với F4 do độ
phân giải cao hơn. Toàn bộ 25 epoch mất khoảng 9,36 giờ.

## 8. Phân tích kết quả

### 8.1. Tác động của augmentation: F0 → F1

Đây là phép so sánh có kiểm soát vì hai run giữ nguyên model, anchors,
proposals, resize, optimizer, batch size và seed.

| Chênh lệch F1 − F0 | mAP | AP50 | AP75 | AP-small | AR@100 |
|---|---:|---:|---:|---:|---:|
| Điểm phần trăm | −0,43 | +0,85 | −0,65 | −0,12 | −0,92 |

Augmentation làm model dễ nhận ra object ở IoU 0,50 hơn nhưng giảm chất lượng
localization và recall tổng. Không nên giữ nguyên gói augmentation F1. Nếu thử
lại, nên loại vertical flip và blur trước, giảm rotation xuống ±5°, sau đó
thực hiện từng ablation riêng.

### 8.2. Tác động của small anchors: F0 → F2

| Chênh lệch F2 − F0 | mAP | AP50 | AP75 | AP-small | AR@100 |
|---|---:|---:|---:|---:|---:|
| Điểm phần trăm | −0,74 | −0,30 | −1,21 | +0,46 | −1,22 |

Small anchors đạt mục tiêu cục bộ là tăng AP-small nhưng làm giảm mAP, AP75 và
recall. Trên VisDrone evaluator, F2 có AP50 cao nhất 49,86% nhưng AP75 chỉ
23,21%, cho thấy phát hiện thô tốt hơn localization chính xác.

### 8.3. Tác động của crowded proposals: F0 → F3

| Chênh lệch F3 − F0 | mAP | AP50 | AP75 | AP-small | AR@100 |
|---|---:|---:|---:|---:|---:|
| Điểm phần trăm | +0,42 | +0,58 | +0,55 | +0,76 | −0,07 |

Tăng proposals cải thiện đồng thời mAP, AP50, AP75 và AP-small. AR@100 gần như
không đổi vì COCO evaluator vẫn giới hạn output ở 100 detection/ảnh. Đây là
thay đổi phù hợp với mật độ đối tượng cao của VisDrone.

### 8.4. Tác động của Faster R-CNN V2: F3 → F4

Đây là phép so sánh official-compatible rõ nhất. F4 giữ preprocessing,
augmentation, anchors, proposals và resize của F3; effective batch size vẫn
là 2.

| Chênh lệch F4 − F3 | AP | AP50 | AP75 | AR@1 | AR@10 | AR@100 | AR@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Điểm phần trăm | +0,72 | −0,35 | +1,50 | +0,02 | +0,40 | +1,19 | +0,88 |

AP50 giảm nhẹ nhưng AP tổng, AP75 và toàn bộ recall tăng. V2 cải thiện độ
chính xác bounding box và khả năng giữ true positive. Đây là thay đổi hiệu quả
nhất trong chuỗi thí nghiệm đã hoàn tất.

### 8.5. Tác động của tăng độ phân giải: F4 → F5

F5 giữ model V2, anchors, crowded proposals, augmentation, effective batch
size và seed của F4. Thay đổi chính là resize 800/1333 → 896/1493; đồng thời
F5 chọn best trực tiếp theo VisDrone AP. So sánh hai `best.pth` bằng cùng
VisDrone evaluator:

| Chênh lệch F5 − F4 | AP | AP50 | AP75 | AR@1 | AR@10 | AR@100 | AR@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Điểm phần trăm | **+1,46** | **+3,34** | **+0,92** | +0,16 | −0,03 | **+1,00** | **+2,60** |

AP tăng tương đối 5,40%. Cải thiện lớn ở AP50 và AR@500 xác nhận tăng độ phân
giải giúp tìm thêm vật thể trong ảnh crowded. AP75 cũng tăng 0,92 điểm, nên
lợi ích không chỉ đến từ các detection lỏng ở IoU thấp. AR@10 gần như không
đổi vì giới hạn 10 detection quá thấp so với mật độ đối tượng của VisDrone.

`last.pth` epoch 25 có AP thấp hơn `best.pth` 2,26 điểm và AR@500 thấp hơn 4,45
điểm. Khi inference hoặc báo cáo F5 phải dùng `best.pth` epoch 5.

### 8.6. Cải thiện tích lũy F1 → F5

F5 so với F1 tăng 4,31 điểm AP, 5,40 điểm AP50, 4,64 điểm AP75, 3,83 điểm
AR@100 và 7,32 điểm AR@500. AP tăng tương đối khoảng 17,9%.

Cải thiện tích lũy đến từ crowded proposals, Faster R-CNN V2 và độ phân giải
cao hơn. F1 augmentation và F2 small anchors không được giữ trong F5.

Không được xem F1→F2 hoặc F2→F3 là ablation nhân quả thuần: F2 vừa bỏ
augmentation vừa đổi anchors; F3 vừa bỏ small anchors vừa tăng proposals.

## 9. Đánh giá kỹ thuật và hạn chế

F5 `best.pth` epoch 5 là model tốt nhất hiện tại. Tuy nhiên:

1. AP50 51,94% nhưng AP75 chỉ 26,93%, chênh 25,01 điểm. Model tìm thấy nhiều
   object nhưng localization ở IoU cao còn hạn chế.
2. AR tăng từ 6,53% tại 10 detections lên 35,71% tại 100 và 46,17% tại 500.
   Giới hạn output nhỏ không phù hợp ảnh crowded.
3. AR@500 vẫn chỉ 46,17%, cho thấy false negative không chỉ do detection cap;
   vật thể nhỏ, che khuất và mật độ cao vẫn là bottleneck.
4. Chỉ có một seed 42 cho mỗi cấu hình. Chưa có độ lệch chuẩn qua nhiều seed,
   nên chênh lệch dưới khoảng 0,5 điểm cần được diễn giải thận trọng.
5. Evaluator hiện là Python port tương thích toolkit; cần chạy chéo prediction
   TXT bằng MATLAB toolkit nếu kết quả được dùng cho công bố chính thức.
6. Bảng F0 official chưa hợp lệ, vì vậy bảng F0–F5 cuối cùng còn thiếu baseline
   official đúng.

## 10. Khuyến nghị thí nghiệm tiếp theo

1. Dùng `models/checkpoints/f5/best.pth` epoch 5 làm model chính; không dùng
   `last.pth` epoch 25.
2. Đánh giá lại đúng `models/checkpoints/f0/results/faster_rcnn_runs/f0/best.pth`
   để hoàn chỉnh ablation table.
3. **Chưa bắt buộc train F6.** F5 đã vượt mọi ngưỡng chấp nhận và đủ làm kết
   quả cuối nếu mục tiêu là hoàn thiện báo cáo hiện tại.
4. Nếu cần tiếp tục tối ưu, chạy một thí nghiệm đánh giá rẻ hơn trước: tiled
   inference trên F5 `best.pth`, tile 640 px, overlap 20%, ánh xạ box về tọa độ
   toàn ảnh, class-wise NMS và giữ tối đa 500 box/ảnh. Chỉ đặt tên/train F6 nếu
   tiled inference tăng ít nhất 1,0 điểm AP hoặc 1,5 điểm AR@500.
5. Nếu tiled inference có lợi, F6 nên là **F5 + tiled train/inference**; giữ
   nguyên V2, default anchors, crowded proposals, seed, optimizer và evaluator
   để chỉ khảo sát một yếu tố là tiling.
6. Thử augmentation nhẹ theo từng bước: horizontal flip; sau đó photometric;
   cuối cùng affine ±5°. Không bật đồng thời tất cả khi chưa đo riêng tác động.
7. Bổ sung AP/AP50/AP75 theo từng class và phân tích lỗi theo kích thước,
   occlusion và truncation.
8. Chạy F5 với ít nhất ba seed để báo cáo mean ± standard
   deviation.

## 11. Tái lập thí nghiệm

### 11.1. Kiểm tra local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-cnn.txt
pytest -q
```

Smoke test:

```bash
python scripts/train_faster_rcnn.py \
  --data-root data/processed/VisDrone \
  --output-dir models/checkpoints \
  --experiment F0 \
  --smoke-test
```

Smoke test chỉ kiểm tra pipeline và không được dùng làm kết quả báo cáo.

### 11.2. Full training trên Kaggle

Notebook: `models/faster_rcnn_visdrone.ipynb`.

```bash
python scripts/train_faster_rcnn.py \
  --data-root /kaggle/input/<processed-dataset>/VisDrone \
  --output-dir /kaggle/working/faster_rcnn_runs \
  --experiment F4 \
  --epochs 25 \
  --batch-size 1 \
  --workers 2 \
  --seed 42
```

F5 cần thêm raw validation root:

```bash
python scripts/train_faster_rcnn.py \
  --data-root /kaggle/input/<processed-dataset>/VisDrone \
  --raw-val-root /kaggle/input/<raw-val>/VisDrone2019-DET-val \
  --output-dir /kaggle/working/faster_rcnn_runs \
  --experiment F5 \
  --epochs 25 \
  --batch-size 1 \
  --workers 2 \
  --seed 42 \
  --min-size 896 \
  --max-size 1493
```

### 11.3. Đánh giá checkpoint

Notebook: `models/evaluate_visdrone_checkpoints.ipynb`.

```bash
python scripts/evaluate_visdrone_checkpoint.py \
  --checkpoint /kaggle/input/<checkpoint>/best.pth \
  --data-root /kaggle/input/<processed-dataset>/VisDrone \
  --raw-val-root /kaggle/input/<raw-val>/VisDrone2019-DET-val \
  --output-dir /kaggle/working/visdrone_official_eval \
  --workers 2 \
  --score-threshold 0.001
```

## 12. Kết luận

Pipeline đã xây dựng được quy trình đầy đủ từ kiểm tra dữ liệu, chuyển đổi
annotation, augmentation bbox-safe, huấn luyện trên Kaggle, lưu checkpoint và
đánh giá tương thích VisDrone DET. Kết quả hiện tại cho thấy augmentation F1
và small anchors F2 không cải thiện mAP tổng; crowded proposals F3 đem lại cải
thiện nhỏ nhưng ổn định; Faster R-CNN V2 trong F4 cải thiện localization và
recall; tăng độ phân giải trong F5 tạo bước tăng lớn nhất trên VisDrone AP.

F5 `best.pth` epoch 5 là cấu hình được khuyến nghị với VisDrone AP 28,42%,
AP50 51,94%, AP75 26,93%, AR@100 35,71% và AR@500 46,17%. F5 đã đủ làm kết quả
cuối của pipeline hiện tại; F6 là tùy chọn nghiên cứu thêm, không phải yêu cầu
bắt buộc. Trước khi tốn chi phí train F6, nên kiểm tra giả thuyết tiling bằng
tiled inference trên checkpoint F5 hiện có.

## Tài liệu và mã nguồn liên quan

- PyTorch Torchvision Faster R-CNN:
  <https://docs.pytorch.org/vision/master/models/faster_rcnn.html>
- VisDrone2018-DET-toolkit:
  <https://github.com/VisDrone/VisDrone2018-DET-toolkit>
- Preprocessing: `src/preprocess_visdrone.py`, `src/visdrone_data.py`
- Augmentation: `src/augmentation.py`
- Model và COCO evaluation: `src/faster_rcnn.py`
- VisDrone evaluation: `src/visdrone_evaluation.py`
- Training entry point: `scripts/train_faster_rcnn.py`
- Evaluation entry point: `scripts/evaluate_visdrone_checkpoint.py`
