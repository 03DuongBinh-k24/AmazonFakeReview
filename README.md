# AmazonFakeReview

**Demo trực tuyến:** [Mở ứng dụng trên Streamlit](https://amazonfakereview.streamlit.app/)

Ứng dụng hỗ trợ chuyên viên sàng lọc các đánh giá sản phẩm Amazon bằng tiếng Anh. Mô hình LightGBM dự đoán nhãn người viết (OR) hoặc máy sinh (CG) từ nội dung đánh giá, điểm sao và danh mục sản phẩm.

## Tính năng

- **Kiểm định đơn lẻ:** nhập nội dung, điểm sao và danh mục để xem nhãn dự đoán, xác suất CG, khuyến nghị xử lý và giải thích TreeSHAP.
- **Kiểm định hàng loạt:** tải tệp CSV hoặc XLSX có các cột `text`, `rating` và `category`. Mỗi lượt xử lý tối đa 5.000 dòng. Kết quả có thể xuất ra Excel.
- **Hướng dẫn vận hành:** xem phạm vi áp dụng và quy tắc phân luồng. Đánh giá dưới 20 từ hoặc có xác suất CG từ 0,40 đến 0,60 được đề xuất chuyển hậu kiểm.

## Chạy trên máy cá nhân

Dùng Python 3.11 để tương thích với môi trường lưu mô hình. Chạy từ thư mục gốc của repo:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Lần khởi động đầu tiên cần kết nối Internet để tải tài nguyên NLTK. Khi triển khai trên Streamlit Community Cloud, chọn `app.py` ở thư mục gốc làm điểm khởi chạy.

## Dữ liệu demo và diễn giải kết quả

Dùng `data/demo_batch_reviews.xlsx` để thử quy trình xử lý hàng loạt với 50 đánh giá. Tệp demo dùng để kiểm thử luồng xử lý và xuất báo cáo. Tab kiểm định đơn lẻ có sẵn bốn mẫu để minh họa dự đoán và cảnh báo hậu kiểm.

TreeSHAP cho thấy các đặc trưng đóng góp như thế nào vào dự đoán. Chuyên viên xem xét kết quả cùng ngữ cảnh trước khi quyết định xử lý.
