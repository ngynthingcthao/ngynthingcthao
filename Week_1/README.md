# Bài tập thực hành 01 - Xây dựng mô hình mạng DNN từ đầu (from scratch)

## Bài 1
Xây dựng mô hình mạng DNN từ đầu (from scratch), không sử dụng các  thư viện deep learning có sẵn như TensorFlow, PyTorch để phân loại ảnh chữ số viết tay MNIST, với các yêu cầu sau: 
- Mạng chỉ bao gồm lớp đầu vào, lớp đầu ra và 1 lớp ẩn.  
- Huấn luyện và đánh giá trên ít nhất trên 5 bộ siêu tham số khác nhau. Các siêu 
tham số có thể điều chỉnh bao gồm: batch_size, learning_rate, số nơron lớp ẩn, 
hàm kích hoạt. Ví dụ: (32, 0.1, 16, Relu), (16, 0.01, 64, Sigmoid), … 
- Với mỗi bộ siêu tham số, sinh viên cần chạy thử nghiệm ít nhất 5 lần. Sau đó 
tính kết quả trung bình và độ lệch chuẩn của độ chính xác trên tập test.  
- Kết quả nộp bài bao gồm: mã nguồn và báo cáo tóm tắt kết quả (đề bài, mô 
hình, siêu tham số, kết quả, nhận xét).

## Mô hình và thuật toán
- **Mô hình**: DNN 3 lớp:
  - Lớp đầu vào: 784 nơron (do là ảnh 28x28).
  - Lớp ẩn: Số nơron thay đổi (16, 32, 64, 128), hàm kích hoạt (ReLU hoặc Sigmoid).
  - Lớp đầu ra: 10 nơron, hàm Softmax.
- **Thuật toán**:
  - Lan truyền xuôi (Forward Propagation): Tính đầu ra qua các lớp.
  - Lan truyền ngược (Backward Propagation): Cập nhật trọng số bằng Gradient Descent.
  - Hàm mất mát: Cross-Entropy Loss.

## Thực nghiệm
### Dữ liệu
- Dataset: MNIST (từ Kaggle).
- Tập huấn luyện: 60,000 ảnh.
- Tập kiểm tra: 10,000 ảnh.
- Tiền xử lý: Chuẩn hóa pixel về [0, 1], nhãn dạng one-hot encoding.

### Siêu tham số
Thử nghiệm 5 bộ siêu tham số:
1. (batch_size=32, learning_rate=0.1, hidden_size=16, activation=ReLU)
2. (batch_size=16, learning_rate=0.01, hidden_size=64, activation=Sigmoid)
3. (batch_size=64, learning_rate=0.05, hidden_size=32, activation=ReLU)
4. (batch_size=32, learning_rate=0.001, hidden_size=128, activation=Sigmoid)
5. (batch_size=16, learning_rate=0.1, hidden_size=32, activation=ReLU)

### Kết quả
| Batch Size | Learning Rate | Hidden Size | Activation | Mean Accuracy (%) | Std Accuracy (%)|
|------------|---------------|-------------|------------|-------------------|-----------------|
| 32         | 0.1           | 16          | ReLU       | 94.85             | 0.00278         |
| 16         | 0.01          | 64          | Sigmoid    | 92.30             | 0.00160         |
| 64         | 0.05          | 32          | ReLU       | 95.36             | 0.00159         |
| 32         | 0.001         | 128         | Sigmoid    | 46.23             | 0.00824         |
| 16         | 0.1           | 32          | ReLU       | 96.16             | 0.00493         |

- Số epoch: 10 cho mỗi lần huấn luyện.
- Độ chính xác: Tính trên tập test (10,000 ảnh).

## Nhận xét
- Với (32, 0.1, 16, 'relu') -> Accuracy cao nhưng hidden size hơi bé nên mạng có thể học chưa sâu -> là cấu hình tốt nhưng có thể chưa khai thác tối đa
- Với (16, 0.01, 64, 'sigmoid') -> Batch size nhỏ giúp học ổn định, Sigmoid + learning rate nhỏ -> học chậm hơn
- Với (64, 0.05, 32, 'relu') -> Accuracy cao, loss thấp nhất -> Là cấu hình cân bằng giữa batch size, learning rate, số nơron, và activation
- Với (32, 0.001, 128, 'sigmoid') -> Học quá chậm (learning rate = 0.001) + sigmoid -> mạng gần như không học được gì
- Với (16, 0.1, 32, 'relu') -> Accuracy cao nhất (96.16%)  tuy nhiên loss hơi cao -> có thể overfit nhẹ.
- **Hiệu quả**: Bộ siêu tham số (16, 0.1, 32, ReLU) cho độ chính xác trung bình cao nhất (96.16%) và độ lệch chuẩn thấp (0.00493), cho thấy mô hình ổn định.