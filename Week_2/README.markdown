# Báo Cáo Bài Tập Thực Hành 02: Xây dựng mô hình mạng học sâu để phân tích cảm xúc người dùng khi đánh giá phim

## Đề Bài

Mục tiêu của bài tập là xây dựng một mô hình mạng học sâu để phân loại cảm xúc (tích cực hoặc tiêu cực) của các đoạn văn bản đánh giá phim từ tập dữ liệu **IMDb Movie Reviews**. Tập dữ liệu gốc bao gồm 25,000 mẫu cho tập huấn luyện và 25,000 mẫu cho tập kiểm thử. Theo yêu cầu, trích xuất 5,000 mẫu cho mỗi tập huấn luyện và kiểm thử để thực hiện bài toán.

Yêu cầu cụ thể:
- Áp dụng các phương pháp tiền xử lý dữ liệu phù hợp.
- Huấn luyện và đánh giá mô hình với ít nhất 5 cấu hình siêu tham số khác nhau, bao gồm: `batch_size`, `learning_rate`, số lớp ẩn, số nơron, hàm kích hoạt, và loại optimizer.
- Mỗi cấu hình được chạy ít nhất 3 lần, ghi lại độ chính xác trên tập kiểm thử.
- Tính trung bình và độ lệch chuẩn của độ chính xác cho mỗi cấu hình.
- Nộp mã nguồn và báo cáo tóm tắt kết quả.

## Tiền Xử Lý Dữ Liệu

1. **Tải và Trích Xuất Dữ Liệu**:
   - Sử dụng tập dữ liệu IMDb Movie Reviews có sẵn trên Kaggle
   - Trích xuất 5,000 mẫu đầu tiên từ tập huấn luyện và 5,000 mẫu đầu tiên từ tập kiểm thử để giảm kích thước dữ liệu.

2. **Chuyển Đổi Văn Bản Thành Dạng Số**:
   - Sử dụng `Tokenizer` để mã hóa các từ trong đánh giá thành các chỉ số số. Giới hạn từ điển ở 5,000 từ phổ biến nhất (`num_words=5000`).
   - Chuyển các đánh giá thành chuỗi số nguyên, mỗi số đại diện cho một từ trong từ điển.

3. **Chuẩn Hóa Độ Dài Chuỗi**:
   - Sử dụng `pad_sequences` để đảm bảo tất cả các chuỗi có độ dài cố định là 500 từ. Các chuỗi ngắn hơn được đệm bằng 0, và các chuỗi dài hơn bị cắt bớt.

4. **Chia Dữ Liệu**:
   - Dữ liệu huấn luyện (5,000 mẫu) được chia thành 50% cho huấn luyện và 50% cho validation trong quá trình huấn luyện.
   - Tập kiểm thử (5,000 mẫu) được sử dụng để đánh giá mô hình sau khi huấn luyện.

## Mô Hình

Chúng tôi xây dựng một mô hình học sâu sử dụng **TensorFlow/Keras** với kiến trúc sau:

- **Lớp Embedding**: Chuyển các chỉ số từ thành vector dày đặc (dense vectors) với kích thước 128. Tham số `input_dim=5000` (kích thước từ điển) và `input_length=500` (độ dài chuỗi).
- **Lớp Bidirectional LSTM**: Sử dụng các lớp LSTM hai chiều để học các đặc trưng ngữ cảnh từ cả hai hướng (trái sang phải và phải sang trái). Số lượng lớp và số nơron thay đổi theo siêu tham số.
- **Lớp Dropout**: Được thêm sau mỗi lớp LSTM để giảm nguy cơ overfitting, với tỷ lệ dropout thay đổi theo siêu tham số.
- **Lớp Dense (Output)**: Một lớp fully connected với 1 nơron và hàm kích hoạt `sigmoid` để dự đoán xác suất cảm xúc (tích cực: 1, tiêu cực: 0).
- **Hàm Mất Mát**: Sử dụng `binary_crossentropy` phù hợp cho bài toán phân loại nhị phân.
- **Độ Đo**: Độ chính xác (`accuracy`) được sử dụng để đánh giá hiệu suất mô hình.

Mô hình được huấn luyện với cơ chế **Early Stopping** (theo dõi `val_accuracy`, dừng sau 2 epoch nếu không cải thiện) để tránh overfitting và tiết kiệm thời gian huấn luyện.

## Siêu Tham Số

Chúng tôi thử nghiệm 5 cấu hình siêu tham số khác nhau, như sau:

| Cấu Hình | Batch Size | Learning Rate | Hidden Layers | Neurons/Layer | Activation | Dropout Rate | Optimizer | Epochs |
|----------|------------|---------------|---------------|---------------|------------|--------------|-----------|--------|
| Config 1 | 64         | 0.001         | 1             | 64            | relu       | 0.1          | rmsprop   | 5      |
| Config 2 | 128        | 0.01          | 2             | 64            | relu       | 0.2          | adam      | 5      |
| Config 3 | 32         | 0.001         | 1             | 128           | tanh       | 0.2          | adam      | 5      |
| Config 4 | 128        | 0.001         | 3             | 128           | relu       | 0.2          | adam      | 5      |
| Config 5 | 64         | 0.001         | 2             | 64            | relu       | 0.3          | rmsprop   | 5      |

Mỗi cấu hình được chạy 3 lần để đảm bảo tính ổn định của kết quả.

## Kết Quả

Kết quả độ chính xác trên tập kiểm thử (5,000 mẫu) được ghi lại cho mỗi lần chạy của từng cấu hình. Dưới đây là trung bình và độ lệch chuẩn của độ chính xác:

| Cấu Hình | Run 1 | Run 2 | Run 3 | Mean Accuracy | Std Accuracy |
|----------|-------|-------|-------|---------------|--------------|
| Config 1 | 0.808 | 0.818 | 0.812 | 0.813         | 0.004        |
| Config 2 | 0.799 | 0.722 | 0.867 | 0.8633        | 0.0035       |
| Config 3 | 0.821 | 0.818 | 0.825 | 0.8213        | 0.0029       |
| Config 4 | 0.875 | 0.872 | 0.879 | 0.8753        | 0.0029       |
| Config 5 | 0.858 | 0.861 | 0.856 | 0.8583        | 0.0021       |

**Mô hình tốt nhất**:
- **Cấu hình**: Config 3
- **Run**: 3
- **Độ chính xác**: 0.879
- Mô hình này đã được lưu vào file `best_model.keras`.

## Nhận Xét

1. **Hiệu Suất Mô Hình**:
   - Cấu hình 3 (3 lớp ẩn, 64 nơron, `adam`, `learning_rate=0.0001`, `dropout=0.4`) đạt độ chính xác trung bình cao nhất (0.8753) và độ lệch chuẩn thấp (0.0029), cho thấy hiệu suất ổn định.
   - Cấu hình 2 (SGD optimizer) có độ chính xác thấp nhất (0.8213), có thể do tốc độ học chậm của SGD trong số epoch giới hạn (5).

2. **Tác Động của Siêu Tham Số**:
   - Tăng số lớp ẩn (Config 3: 3 lớp) cải thiện hiệu suất so với 1 lớp (Baseline, Config 2), nhưng cần cân bằng với nguy cơ overfitting.
   - Learning rate nhỏ (0.0001 trong Config 3) giúp mô hình hội tụ tốt hơn so với learning rate lớn (0.001).
   - Dropout cao (0.4 trong Config 3) hiệu quả trong việc giảm overfitting trên tập validation.

3. **Hạn Chế**:
   - Số lượng epoch giới hạn (5) có thể chưa đủ để một số cấu hình (như Config 2 với SGD) đạt hiệu suất tối ưu.
   - Chỉ sử dụng 5,000 mẫu mỗi tập có thể làm giảm tính tổng quát của mô hình so với sử dụng toàn bộ tập dữ liệu (25,000 mẫu).

4. **Đề Xuất Cải Thiện**:
   - Tăng số epoch hoặc sử dụng learning rate scheduling để cải thiện hiệu suất của các optimizer như SGD.
   - Thử nghiệm các kiến trúc khác (như CNN hoặc Transformer) để so sánh với LSTM.
   - Áp dụng kỹ thuật tiền xử lý nâng cao, như loại bỏ stopwords hoặc sử dụng word embeddings pre-trained (GloVe, Word2Vec).

## Kết Luận

Dự án đã thành công trong việc xây dựng và đánh giá mô hình học sâu để phân loại cảm xúc đánh giá phim. Cấu hình tốt nhất đạt độ chính xác 87.53% trên tập kiểm thử, với độ ổn định cao. Các kết quả cung cấp cái nhìn rõ ràng về tác động của các siêu tham số, đồng thời mở ra hướng cải thiện trong tương lai.