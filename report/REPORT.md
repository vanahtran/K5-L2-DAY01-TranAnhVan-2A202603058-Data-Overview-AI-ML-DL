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

1. Chép record hạng 1: `class_id`= 668 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 Record này nói về cả bức ảnh. Vì trường task ghi rõ là "image_classification" (phân loại toàn ảnh) và hoàn toàn không có các trường thông tin toạ độ như bounding box (bbox) hay điểm polygon để khoanh vùng vị trí của một vật thể cụ thể nào.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
 Danh sách này đi kèm với taxonomy/checkpoint.Khi nhìn vào JSON sẽ thấy trường taxonomy_name là "ImageNet-1K". Điều này có nghĩa là checkpoint model này đã được train trên bộ dữ liệu ImageNet-1K, nó sử dụng cố định danh sách 1000 lớp của bộ đó chứ không thể tự nghĩ ra tên lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
 Vì chỉ dùng class_name thì rất dễ sinh ra lỗi đồng âm khác nghĩa hoặc sai chính tả. taxonomy_name giúp xác định rõ hệ quy chiếu đang dùng, còn class_id là mã số định danh duy nhất giúp hệ thống máy tính xử lý chính xác, tự động mà không bị nhầm lẫn ngôn ngữ.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
 Guideline bắt buộc phải quy định rõ tiêu chí ưu tiên. Ví dụ: phải chọn đối tượng nằm ngay trung tâm ảnh, đối tượng chiếm diện tích lớn nhất, hoặc chọn theo chủ đề ưu tiên của dự án. Đồng thời, guideline cũng phải hướng dẫn cách xử lý ca khó (escalation báo cho reviewer/lab coach) nếu có nhiều vật tương đương nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
 Vì dưới ghi chú của hình ảnh có ghi: "độ tin cậy của mô hình (không phải chất lượng ground truth)" [1]. Score (ở đây là 0.510915) chỉ là mức độ tự tin của máy tính đối với phán đoán của nó. Máy hoàn toàn có thể tự tin rất cao nhưng dự đoán sai bét. Ground truth phải là "chân lý" do con người (annotator) xác nhận bằng mắt dựa trên guideline.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 Định dạng bbox_xyxy (x_min, y_min, x_max, y_max) biểu diễn tọa độ không gian của bounding box trên lưới pixel của bức ảnh. Áp dụng vào bản ghi trên: Góc trên-trái (x1, y1): Nằm cách mép trái của bức ảnh 385.33 pixel và cách mép trên của bức ảnh 69.24 pixel. Góc dưới-phải (x2, y2): Nằm cách mép trái của bức ảnh 498.92 pixel và cách mép trên của bức ảnh 348.92 pixel. Kích thước vật lý: Hộp bao quanh vật thể có chiều rộng đo được là 113.58 pixel và chiều cao là 279.68 pixel.   
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 Tác động của threshold (Theo yêu cầu Rubric): Khi đặt ngưỡng thấp (0.20), mô hình sẽ trả về số lượng prediction nhiều hơn, do chấp nhận cả những dự đoán có độ tin cậy thấp (dễ sinh ra False Positive). Ngược lại, khi đặt ngưỡng cao (0.60), mô hình trả về số lượng prediction ít hơn, chỉ giữ lại các hộp mà mô hình đánh giá với độ tự tin cao (giảm False Positive nhưng rủi ro tăng False Negative - bỏ sót vật thể).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 Khi hạ ngưỡng, lượng hộp dự đoán tăng lên kéo theo tỷ lệ nhiễu (False Positives) cao. Người kiểm tra chất lượng (Reviewer/QC) sẽ phải thực hiện các thao tác: kiểm tra trực quan một lượng lớn bounding box, xóa bỏ các hộp nhận diện sai (background hoặc nhận diện sai nhãn), và tinh chỉnh kích thước các hộp bị chồng lấn (overlapping) để đảm bảo chất lượng nhãn ground truth đạt chuẩn.    
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
 Quy tắc Bao sát (Tight Bounding Box): "Người gán nhãn (Annotator) phải vẽ bounding box sao cho các cạnh của hộp vừa chạm sát vào các điểm biên ngoài cùng của phần vật thể nhìn thấy được. Không được để khoảng trống (margin) dư thừa và tuyệt đối không vẽ hộp bao trùm lên phần bóng đổ (shadow) của vật thể trên các bề mặt."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
 Hiện tượng quan sát được: Trên ảnh detection_predictions.jpg, đối tượng "person" (người đứng quay lưng) đang bị cắt mất một phần chân bởi mép dưới của bức ảnh và một tay bị che bởi mép phải.   Quyết định: Viết thành Quy tắc trong Guideline thay vì Escalation.Giải thích (Theo tiêu chí Rubric): Tình trạng vật thể bị cắt bởi mép ảnh (truncation) hoặc che khuất (occlusion) là rất phổ biến trong dữ liệu thị giác máy tính. Thay vì để người gán nhãn (annotator) liên tục báo cáo (escalation) những ca này, dự án cần thống nhất quy tắc ngay từ đầu để xử lý hàng loạt. Ví dụ quy tắc: "Đối với vật thể bị cắt mép ảnh, chỉ vẽ bounding box bao quanh vùng diện tích hiển thị thực tế trên ảnh, không tự suy luận và vẽ lấn ra ngoài phần bị cắt." Việc này giúp tách bạch rõ ràng giữa quy tắc áp dụng chung và những trường hợp dị biệt thật sự cần escalation.   

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 So với bounding box (chỉ định vị không gian bằng một hình chữ nhật thô), đa giác (polygon) cho thêm thông tin chi tiết về đường viền biên (contour) và độ lồi lõm thực tế của vật thể ở cấp độ pixel. Việc này giúp mô hình tách bạch chính xác diện tích thực tế của vật thể ra khỏi phông nền (background) hoặc các vật thể lân cận bị lọt vào bên trong hộp chữ nhật (như đáp ứng yêu cầu "khác biệt với box" trong Rubric).   
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
 instance_id : đóng vai trò làm mã định danh duy nhất để đếm và phân biệt từng "cá thể" riêng lẻ tồn tại bên trong cùng một bức ảnh.Không phải class ID: Vì class ID chỉ đại diện cho chủng loại (nhóm). Nếu ảnh có 5 người thì cả 5 đều chung class_id = 0 nhưng phải có 5 instance_id khác biệt.Không phải tracking ID: Vì tracking ID được sinh ra để theo dõi quỹ đạo của cùng một cá thể xuyên suốt nhiều khung hình (frame) liên tiếp trong một video. Trong bài lab tĩnh này, instance_id chỉ có giá trị cục bộ trên một ảnh duy nhất.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
 Quy tắc: "Người gán nhãn (Annotator) phải vẽ đường bao đa giác (polygon) men sát theo đường biên ngoài cùng (pixel-perfect) của phần vật thể hiển thị thực tế trên ảnh. Đối với vật thể bị rỗng bên trong (như quai chảo/tay cầm đồ vật), phải vẽ thêm mask phụ để đục lỗ (khoét) phần phông nền lọt thỏm bên trong ra."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 Hiện tượng quan sát được: Trên ảnh "segmentation_prediction.jpg", khu vực cánh tay trái (từ góc nhìn của người xem) của đối tượng "person" (người) đang thao tác tại bàn bị che khuất và hòa lẫn vào các đồ vật trên mặt bàn (như bát, đĩa, bột). Ranh giới pixel giữa tay áo và đồ vật bên dưới rất mờ nhạt.   Quyết định xử lý: Xử lý bằng Guideline (Quy tắc hướng dẫn).Vì sao (Theo chuẩn Rubric): Sự kiện các đối tượng chạm nhau hoặc che lấn nhau (occlusion) là thuộc tính tự nhiên của dữ liệu thị giác. Việc quy định rõ cách xử lý ngay trong guideline (ví dụ: "Chỉ vẽ polygon bao quanh vùng diện tích hiển thị chắc chắn thuộc về vật thể, dừng đường nét tại điểm bắt đầu bị vật khác đè lên, tuyệt đối không suy luận phần hình khối bị che khuất phía sau") sẽ tạo ra một bộ quy tắc (rule) nhất quán, có khả năng mở rộng (scale) cho mọi annotator, tránh việc phải báo cáo (escalation) tràn lan làm nghẽn tiến độ dự án.   

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Là **một nhãn duy nhất** (gồm `class_id`, `class_name` từ Taxonomy) áp dụng cho toàn bộ bức ảnh. | Ảnh `traffic` có nhiều xe/người nhưng model dự đoán cao nhất là `cab` (score 0.51). Khó chọn nếu ảnh có nhiều vật thể chính. | Áp dụng Guideline (ví dụ: quy tắc ưu tiên vật thể to nhất/nằm giữa ảnh) để chọn đúng **một** nhãn. | Kiểm tra xem nhãn được chọn có tuân thủ đúng quy tắc ưu tiên trong Guideline hay không. |
| **Phát hiện vật thể** | Khung bao chữ nhật (**Bounding box** `bbox_xyxy`) bám sát vật thể, kèm nhãn `class_name`. | Trên ảnh `kitchen`, đối tượng người (`person`) bị cắt mất chân ở mép dưới ảnh. Dễ vẽ lấn ra ngoài. | Vẽ box bao sát (tight box) phần vật thể **thực sự nhìn thấy được**, không đoán mò vẽ lấn mép. | Rà soát các lỗi: box vẽ dư khoảng trống (margin), vẽ bao gồm cả phần bóng đổ, hoặc sai nhãn. |
| **Instance segmentation** | Đa giác (**Polygon** `polygon_xy`) ôm sát biên viền pixel, kèm mã định danh riêng `instance_id` cho từng cá thể. | Trên ảnh `kitchen`, cánh tay người (score 0.90) chạm đồ vật làm biên pixel bị mờ; các cái bát xếp chồng lên nhau. | Vẽ polygon men sát mép vật thể ở mức pixel, dừng nét tại điểm bị che khuất. Cấp `instance_id` riêng biệt. | Phóng to (zoom) để soi biên mask có lẹm vào nền, gộp nhầm 2 cá thể, hoặc vẽ bao hàm cả vùng bị che khuất không. |



## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
 Tuyệt đối không chia sẻ hình ảnh, dữ liệu khách hàng hoặc đưa dữ liệu nội bộ lên các công cụ AI công cộng (public) khi chưa được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
 Dừng ngay thao tác gán nhãn và báo cáo (escalation) cho Lab Coach hoặc Quản lý dự án (Project Manager).


## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
