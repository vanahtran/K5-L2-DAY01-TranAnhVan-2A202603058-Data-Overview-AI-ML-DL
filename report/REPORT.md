# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 Đây là nhãn dự đoán ở cấp độ toàn ảnh. Tác vụ trong JSON là image_classification; mỗi record gắn một lớp và một score với sample_id = traffic, không có tọa độ box hay polygon để chỉ ra vị trí một đối tượng cụ thể. Vì vậy, cab là kết quả mô hình dùng để mô tả/đại diện cho nội dung ảnh theo taxonomy, không phải khẳng định rằng trong ảnh chỉ có một chiếc cab.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
 Tên lớp và mã lớp được xác định bởi taxonomy đi kèm mô hình/checkpoint. JSON ghi taxonomy_name = ImageNet-1K; mô hình chọn lớp trong danh sách đã định nghĩa, không tự sáng tạo một nhãn tùy ý ở thời điểm dự đoán.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
 - class_id: mã số để hệ thống tham chiếu ổn định.
 - class_name: tên dễ đọc cho con người.
 - taxonomy_name: cho biết mã và tên đó thuộc bộ phân loại nào.
Chỉ ghi tên lớp sẽ thiếu ngữ cảnh khi các taxonomy khác nhau có thể dùng tên hoặc cách phân nhóm khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
 Guideline phải định nghĩa trước đơn vị cần phân loại và tiêu chí chọn nhãn chính. Ví dụ, nếu mục tiêu dự án là phân loại chủ đề chính của ảnh giao thông, annotator có thể chọn chủ đề chiếm ưu thế theo tiêu chí đã thống nhất (chẳng hạn loại phương tiện/hoạt động chính), không mặc định chọn vật ở chính giữa hoặc vật có diện tích lớn nhất nếu dự án chưa quy định như vậy. Khi hai chủ đề ngang nhau hoặc không thể áp dụng tiêu chí, annotator đánh dấu needs_review và chuyển reviewer/QC; không tự đoán.
6. Vì sao không được lấy `score` của model làm ground truth?
 score = 0.510915 là mức độ mô hình gán cho dự đoán cab, không phải xác nhận của con người về nhãn đúng. Model có thể tự tin nhưng vẫn nhầm; ground truth phải được tạo/duyệt bởi annotator theo guideline và reviewer/QC kiểm tra. Ảnh traffic có nhiều xe buýt, ô tô và người; nhãn top-1 cab không đủ để kết luận toàn bộ ảnh thực tế chỉ có cab. Đây là điểm cần reviewer xem xét theo mục tiêu phân loại của dự án, không tự động sửa prediction thành ground truth.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 Với định dạng xyxy = [x_min, y_min, x_max, y_max]:
- Góc trên-trái: (385.33, 69.24) – cách mép trái 385.33 pixel và mép trên 69.24 pixel.
- Góc dưới-phải: (498.92, 348.92).
- Chiều rộng: 498.92 − 385.33 = 113.59 pixel (JSON lưu 113.58, do làm tròn số thực).
- Chiều cao: 348.92 − 69.24 = 279.68 pixel.
Tọa độ tính từ góc trên-trái ảnh; trục x tăng sang phải, trục y tăng xuống dưới. Box là hình chữ nhật bao vật thể, không phải đường biên pixel chính xác.  
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 JSON được cung cấp chỉ ghi score_threshold = 0.35. Trong detection_predictions.json, mẫu kitchen có 11 prediction được lưu; trong segmentation_predictions.json, mẫu kitchen cũng có 11 instance được lưu ở threshold 0.35. Đây là số record hiện có trong file, không phải số lượng ground-truth object.

Không có output so sánh hoặc danh sách prediction thô cho threshold 0.20 và 0.60 trong các tệp đã gửi, nên không thể xác nhận số prediction ở hai ngưỡng đó. Không nên tự điền số. Về nguyên tắc, nếu giữ nguyên model và dữ liệu, hạ threshold thường giữ lại nhiều prediction hơn; nâng threshold thường loại bớt prediction điểm thấp. Muốn báo cáo con số chính xác cần chạy lại notebook ở từng ngưỡng và lưu bảng đếm tương ứng.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 Reviewer/QC cần rà soát số lượng candidate tăng thêm: xác minh nhãn, box có thực sự bao vật thể hay chỉ bao nền, box trùng/chồng lấn, vật thể bị bỏ sót và các trường hợp bị che/cắt mép. Chỉ xóa/sửa prediction khi có căn cứ từ ảnh và guideline; score thấp không tự động đồng nghĩa prediction sai, score cao cũng không tự động chứng minh prediction đúng.    
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
 Vẽ một bounding box cho từng đối tượng thuộc phạm vi taxonomy; box ôm sát phần đối tượng nhìn thấy được. Nếu đối tượng bị cắt bởi biên ảnh, box dừng tại biên ảnh và chỉ bao phần còn nhìn thấy; không suy diễn phần nằm ngoài ảnh. Không đưa bóng đổ hoặc vật thể lân cận vào box. Nếu không xác định được ranh giới do che khuất, đánh dấu cần review thay vì tự ước lượng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
 Trong detection_predictions.png và record kitchen có một người ở sát mép trái:
- class_name = person, score = 0.610931.
- bbox_xyxy = [0.12, 263.18, 61.55, 310.78].
Box bắt đầu gần x = 0, phù hợp với quan sát người chỉ xuất hiện một phần ở mép trái ảnh. Đây là ví dụ truncation có thể xử lý bằng guideline chung: chỉ gán nhãn phần nhìn thấy trong khung ảnh. Người đứng giữa ảnh (score = 0.912625) không bị cắt chân ở mép dưới theo box [385.33, 69.24, 498.92, 348.92]; do đó không dùng người này làm ví dụ cắt mép. 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 Bounding box chỉ mô tả vùng chữ nhật bao ngoài đối tượng. Polygon lưu các điểm đường biên để mô tả hình dạng và vùng mask sát đối tượng hơn, nhờ đó có thể phân biệt phần đối tượng với nền hoặc các vật thể gần kề. polygon_point_count = 348 là số điểm được lưu trong polygon của instance này; nó không phải số pixel ground truth và cũng không tự chứng minh mask chính xác.  
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
 instance_id = kitchen-001 là mã định danh cho một instance trong artifact/ảnh này. Nó khác với:
- class_id = 0: mã lớp person, có thể dùng chung cho nhiều người.
- Tracking ID: mã theo dõi một đối tượng xuyên qua nhiều frame của video.
Vì vậy, hai người đều thuộc lớp person vẫn cần hai instance riêng trong ảnh; không dùng instance_id thay cho class ID hoặc tracking ID.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
 Với mỗi đối tượng nhìn thấy, tạo một mask riêng và men theo đường biên quan sát được ở mức chi tiết mà guideline yêu cầu. Khi đối tượng bị che, chỉ gán vùng nhìn thấy; không vẽ xuyên qua vật che để đoán phần bị khuất. Không gộp hai cá thể cùng lớp thành một mask. Với vùng rỗng/lỗ bên trong, chỉ tạo vùng rỗng nếu định dạng và guideline của dự án hỗ trợ biểu diễn lỗ; không tự ý thêm “mask phụ” nếu schema không quy định.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 Trong `segmentation_prediction.png`, instance `kitchen-004` được dự đoán là `potted plant`, score `0.632235`, bbox `[0.05, 1.4, 63.45, 148.89]`. Vùng mask nằm ở góc trên-trái, phủ lên phần lá/đồ treo sát mép ảnh. Quan sát ảnh gốc `kitchen.jpg`, vùng này giống cụm lá treo nhưng không thấy rõ chậu cây trong khung hình. Vì vậy, tên lớp `potted plant` là **điểm mơ hồ cần review**, không thể kết luận chắc chắn là đúng hay sai chỉ từ score.

Hướng xử lý: annotator đánh dấu vùng và ghi lý do “không thấy rõ chậu / ranh giới đối tượng sát mép ảnh”; reviewer/QC đối chiếu taxonomy và guideline, sau đó xác nhận nhãn, đổi nhãn nếu có căn cứ, hoặc escalation cho Lab Coach nếu taxonomy không có lớp phù hợp. Không lấy prediction `potted plant` làm quy tắc gán nhãn.
  

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

| Tác vụ | Ground truth cần có | Quan sát từ prediction/ảnh | Annotator thực hiện | Reviewer/QC kiểm tra |
|---|---|---|---|---|
| Classification | Một nhãn cấp ảnh theo taxonomy và guideline | `traffic` top-1 là `cab`, score `0.510915`, trong khi ảnh có nhiều loại phương tiện | Chọn nhãn theo tiêu chí chủ đề chính đã thống nhất; trường hợp ngang nhau thì gắn cờ review | Kiểm tra tiêu chí chọn nhãn, taxonomy và tính nhất quán; không dùng score làm nhãn chuẩn |
| Detection | Mỗi vật thể là class + bounding box `xyxy` | Người ở mép trái `score 0.610931`, bbox bắt đầu gần x=0; có truncation | Vẽ box phần nhìn thấy, không đoán phần ngoài ảnh | Kiểm tra lớp, box, vật thể bị cắt/che, box dư hoặc thiếu |
| Instance segmentation | Mỗi instance là class + mask/polygon riêng | `kitchen-001` person có 348 điểm; `kitchen-004` potted plant ở góc trái là điểm mơ hồ | Vẽ mask riêng, bám biên nhìn thấy; đánh dấu vùng không chắc | Zoom kiểm tra biên, mask gộp/lấn nền, instance trùng và ca cần escalation |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
  Chỉ sử dụng dữ liệu được phép cho bài lab; không tải ảnh riêng, dữ liệu khách hàng, dữ liệu nội bộ/VinFast hoặc dữ liệu cá nhân không cần thiết lên repository công khai hay công cụ AI công cộng khi chưa được cấp quyền.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
 Nếu phát hiện dữ liệu sai phạm vi, nhạy cảm hoặc không rõ quyền sử dụng, dừng xử lý, không chia sẻ tiếp và báo Lab Coach/quản lý dự án theo kênh được chỉ định.


## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
