# Báo cáo đánh giá Faster R-CNN trên VisDrone DET

## 1. Phạm vi đánh giá

Báo cáo tổng hợp kết quả trong
`models/checkpoints/visdrone_official_eval` cho các thí nghiệm F0–F4. Các
prediction được đánh giá trên toàn bộ 548 ảnh validation bằng quy trình tương
thích `VisDrone2018-DET-toolkit`:

- IoU từ 0.50 đến 0.95, bước 0.05;
- tối đa 500 detections trên mỗi ảnh;
- score threshold 0.001;
- loại detections nằm trong ignored regions theo quy tắc VisDrone;
- AP được tích phân theo precision–recall envelope của toolkit;
- checkpoint đầu vào là `best.pth` của từng thí nghiệm.

Metric trong file JSON có miền 0–1. Các bảng dưới đây đổi sang phần trăm để dễ
đọc. Đây là kết quả của Python port tương thích logic toolkit; prediction TXT
đã được lưu để có thể kiểm chứng lại bằng bản MATLAB gốc tại
<https://github.com/VisDrone/VisDrone2018-DET-toolkit>.

## 2. Kiểm tra tính hợp lệ dữ liệu

Mỗi thí nghiệm có đủ 548 file prediction. F1–F4 sử dụng checkpoint full
training ở epoch 16–22 và có phân bố prediction hợp lý.

Riêng F0 **không hợp lệ để so sánh**:

- file được đánh giá là checkpoint epoch 1;
- config checkpoint xác nhận `smoke_test=true` và `pretrained=false`;
- AP chỉ đạt 0.0011%;
- model luôn xuất đúng 500 detections/ảnh, tổng cộng 274.000 detections;
- 191.013/274.000 predictions bị dồn vào class `car`, phù hợp hành vi của model
  gần như chưa train.

Checkpoint F0 đúng hiện có tại
`models/checkpoints/f0/results/faster_rcnn_runs/f0/best.pth`: epoch 17,
`smoke_test=false`, COCO mAP 0.2242. Cần dùng file này để đánh giá lại F0 trước
khi lập bảng F0–F4 cuối cùng.

## 3. Kết quả chính

| Hạng | Experiment | Epoch | AP | AP50 | AP75 | AR@1 | AR@10 | AR@100 | AR@500 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **F4** | 16 | **26.96** | 48.60 | **26.00** | **0.80** | **6.56** | **34.72** | **43.57** |
| 2 | F3 | 16 | 26.24 | 48.95 | 24.50 | 0.78 | 6.17 | 33.52 | 42.69 |
| 3 | F2 | 19 | 25.80 | **49.86** | 23.21 | 0.64 | 5.66 | 32.62 | 43.36 |
| 4 | F1 | 22 | 24.11 | 46.54 | 22.29 | 0.52 | 5.55 | 31.89 | 38.85 |
| — | F0 | 1 | 0.001 | 0.007 | 0.000 | 0.005 | 0.017 | 0.074 | 0.233 |

F0 được hiển thị để truy vết lỗi nhưng bị loại khỏi xếp hạng.

## 4. Phân tích ablation

### So sánh F1 và F2

F1 dùng augmentation với anchors mặc định, còn F2 tắt augmentation và thay
anchor FPN thành 8, 16, 32, 64, 128. Vì có hai yếu tố thay đổi đồng thời, đây
không phải ablation thuần để kết luận nhân quả chỉ cho small anchors. Về kết
quả quan sát được, F2 so với F1:

- AP tăng **1.69 điểm phần trăm**;
- AP50 tăng **3.32 điểm** và đạt mức cao nhất 49.86%;
- AP75 tăng 0.93 điểm;
- AR@500 tăng **4.51 điểm**.

Cấu hình F2 bắt được nhiều vật thể hơn, thể hiện rõ ở AP50 và recall. Tuy nhiên
mức cải thiện AP75 thấp hơn AP50, nghĩa là tăng khả năng phát hiện nhưng chưa
cải thiện tương ứng độ chính xác định vị hộp. Cần F0 đúng để tách riêng tác
động của small anchors khỏi việc bỏ augmentation.

### So sánh F2 và F3

F3 đồng thời quay về anchors mặc định và tăng số RPN proposals, nên phép so
sánh này cũng thay đổi hai yếu tố. So với F2:

- AP tăng 0.44 điểm;
- AP75 tăng **1.29 điểm**;
- AR@100 tăng 0.90 điểm;
- AP50 giảm 0.91 điểm;
- AR@500 giảm 0.67 điểm.

Cấu hình F3 cải thiện chất lượng tổng thể và định vị ở IoU cao, nhưng không tạo
recall cực đại tốt bằng F2. F3 phù hợp hơn khi AP@[.50:.95] là metric chính;
F2 phù hợp hơn nếu ưu tiên phát hiện thô ở IoU 0.50. Chưa thể quy toàn bộ chênh
lệch cho crowded proposals nếu thiếu kết quả F0 hợp lệ.

### F3 → F4: Faster R-CNN V2

F4 giữ crowded proposals của F3 và đổi sang Faster R-CNN ResNet-50-FPN V2.
So với F3:

- AP tăng **0.72 điểm**;
- AP75 tăng **1.50 điểm**;
- AR@100 tăng **1.19 điểm**;
- AR@500 tăng 0.88 điểm;
- AP50 giảm nhẹ 0.35 điểm.

V2 đem lại lợi ích rõ nhất ở localization chính xác và recall, không chỉ tăng
số lượng detection dễ tại IoU 0.50. Đây là thay đổi có lợi và nhất quán trên
các metric chính.

### F1 → F4: cải thiện tích lũy

F4 cải thiện so với F1:

- AP: +2.85 điểm, tương đương tăng tương đối khoảng 11.8%;
- AP50: +2.06 điểm;
- AP75: +3.72 điểm;
- AR@100: +2.83 điểm;
- AR@500: +4.72 điểm.

F4 tạo trung bình khoảng 400 detections/ảnh, ít hơn F1 khoảng 452
detections/ảnh nhưng đạt AP và recall cao hơn. Điều này cho thấy cải thiện đến
từ chất lượng xếp hạng/phân loại và định vị, không phải chỉ xuất nhiều hộp hơn.

## 5. Đánh giá chất lượng hiện tại

F4 là model tốt nhất hiện tại nếu tiêu chí chính là AP@[.50:.95]. Tuy nhiên hệ
thống vẫn còn ba hạn chế:

1. Khoảng cách AP50–AP75 của F4 là 22.60 điểm. Model nhận biết được nhiều vật
   thể nhưng bounding box vẫn chưa đủ chính xác ở IoU cao.
2. AR tăng mạnh từ 6.56% tại 10 detections lên 34.72% tại 100 và 43.57% tại
   500 detections. Dataset rất crowded; giới hạn số detection thấp làm bỏ sót
   phần lớn đối tượng.
3. AR@500 mới đạt 43.57%, nên bottleneck không chỉ là giới hạn output. Vật thể
   nhỏ, che khuất và mật độ cao vẫn là nguồn false negative chính.

## 6. Khuyến nghị

1. **Chọn F4 làm baseline tốt nhất hiện tại.** Dùng checkpoint F4 epoch 16 cho
   inference nếu chưa có kết quả F5.
2. **Đánh giá lại F0 đúng checkpoint epoch 17.** Không sử dụng số F0 hiện tại
   trong luận văn/báo cáo hoặc phép tính improvement.
3. **Tiếp tục F5 ở resize 896/1493.** F5 chỉ được xem là cải thiện nếu vượt các
   mốc F4: AP 26.96%, AP75 26.00% và AR@500 43.57%. AP50 không nên giảm quá 1
   điểm nếu AP tổng tăng.
4. **Giữ evaluator VisDrone từ F5 trở đi.** Chọn `best.pth` theo
   `visdrone_ap`, đồng thời lưu prediction TXT để tái kiểm chứng.
5. **Bổ sung metric theo từng class.** Aggregate AP chưa chỉ ra class nào đang
   kéo kết quả xuống; bước tiếp theo nên báo cáo AP/AP50 riêng cho pedestrian,
   people, car, van, truck, bus và motor.
6. Nếu F5 không tăng AP75/recall đáng kể, ưu tiên thử tiling có overlap thay vì
   tiếp tục tăng full-image resolution, vì ảnh cực crowded đã gây OOM ở
   1024/1707 trên Tesla T4.

## 7. Kết luận

Trong bốn checkpoint full-training hợp lệ, F4 đạt kết quả tốt nhất với AP
26.96%, AP75 26.00%, AR@100 34.72% và AR@500 43.57%. F2 dẫn đầu AP50 ở 49.86%
nhưng thua F4 về AP tổng, AP75 và recall. So sánh có kiểm soát F3→F4 cho thấy
Faster R-CNN V2 cải thiện rõ localization và recall. Các nhận định riêng về
augmentation, small anchors và crowded proposals vẫn chỉ là xu hướng vì F0
official-compatible hiện không hợp lệ; kết luận ablation cuối cùng phải chờ
đánh giá lại đúng checkpoint F0 epoch 17.
