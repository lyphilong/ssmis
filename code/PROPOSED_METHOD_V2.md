# Phương pháp đề xuất: Học bán giám sát kết hợp Consistency, Contrastive và Topological Progression Loss

## 1. Tổng quan

Framework của chúng tôi được xây dựng trên nền tảng của DyCON, một kiến trúc học bán giám sát (semi-supervised learning) mạnh mẽ, và được cải tiến bằng một cơ chế regularization mới có tên là **Topological Progression Loss**. Phương pháp này không chỉ tận dụng dữ liệu có nhãn và không nhãn thông qua học nhất quán (consistency) và học tương phản (contrastive), mà còn đảm bảo kết quả phân đoạn hợp lý về mặt cấu trúc trong không gian 3D.

Hàm mất mát tổng thể được định nghĩa như sau:

$$
\mathcal{L}_{\text{total}} = \lambda_s \mathcal{L}_{\text{sup}} + \lambda_u \mathcal{L}_{\text{cons}} + \lambda_{c} (\mathcal{L}_{\text{FeCL}} + \mathcal{L}_{\text{UnCL}}) + \lambda_t \mathcal{L}_{\text{prog}}
$$

Trong đó:
- **$\mathcal{L}_{\text{sup}}$**: Mất mát giám sát (Dice + Cross-Entropy) trên dữ liệu có nhãn.
- **$\mathcal{L}_{\text{cons}}$**: Mất mát nhất quán giữa dự đoán của student và teacher.
- **$\mathcal{L}_{\text{FeCL}} + \mathcal{L}_{\text{UnCL}}$**: Các mất mát tương phản ở cấp độ đặc trưng và có nhận biết bất định.
- **$\mathcal{L}_{\text{prog}}$**: **(Cải tiến chính)** Mất mát Tiến triển Topo, một dạng self-regularization khuyến khích sự mượt mà về cấu trúc giữa các lát cắt.
- $\lambda_s, \lambda_u, \lambda_c, \lambda_t$: Các trọng số để cân bằng các thành phần loss.

---

## 2. Các thành phần kế thừa từ DyCON

### 2.1. Mất mát Giám sát ($\mathcal{L}_{\text{sup}}$)
Trên một lượng nhỏ dữ liệu có nhãn, chúng tôi sử dụng hàm loss kết hợp giữa Dice và Cross-Entropy để tối ưu độ chính xác ở cấp độ pixel.

$$
\mathcal{L}_{\text{sup}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}
$$

### 2.2. Mất mát Nhất quán có nhận biết Bất định ($\mathcal{L}_{\text{cons}}$)
Dựa trên kiến trúc Mean-Teacher, teacher model được cập nhật bằng Trung bình Động Lũy thừa (EMA) từ student model. UnCLoss ($\mathcal{L}_{\text{UnCL}}$) được sử dụng để khuyến khích sự nhất quán giữa dự đoán của student và teacher trên dữ liệu không nhãn, đồng thời tự động giảm trọng số ở những vùng có độ bất định (entropy) cao.

### 2.3. Mất mát Tương phản ($\mathcal{L}_{\text{FeCL}}$)
FeCLoss được áp dụng ở cấp độ đặc trưng (feature-level), giúp học một không gian embedding có khả năng phân biệt cao. Nó kéo các patch (điểm ảnh trong feature map) cùng lớp lại gần nhau và đẩy các patch khác lớp ra xa.

---

## 3. Cải tiến chính: Topological Progression Loss ($\mathcal{L}_{\text{prog}}$)

### 3.1. Động lực
Các phương pháp topological loss truyền thống thường chỉ đánh giá cấu trúc trên một lát cắt 2D hoặc một khối 3D tĩnh. Điều này có thể bỏ qua một khía cạnh quan trọng trong dữ liệu y tế như MRI: **sự liên tục và tiến triển mượt mà của cấu trúc giải phẫu qua các lát cắt liên tiếp**. Một khối u không thể đột ngột xuất hiện hay biến mất giữa hai lát cắt liền kề.

### 3.2. Ý tưởng
Để giải quyết vấn đề này, chúng tôi đề xuất **Topological Progression Loss**, một cơ chế self-regularization mới. Thay vì so sánh student với teacher, loss này đánh giá chính sự "mượt mà" về mặt cấu trúc trong chuỗi dự đoán của student model. Nó mô hình hóa chuỗi các đặc trưng topo qua từng lát cắt và phạt những thay đổi đột ngột, không tự nhiên.

### 3.3. Quy trình tính toán
Quá trình tính toán $\mathcal{L}_{\text{prog}}$ cho một volume dự đoán 3D $S$ bao gồm ba bước:

1.  **Trích xuất chuỗi Persistence Diagram (PD):** Chúng tôi duyệt qua từng lát cắt 2D ($S_d$) của volume $S$ và tính toán 0-dimensional persistence diagram $PD(S_d)$ cho mỗi lát. PD ghi lại thông tin về sự "sinh ra" và "chết đi" của các thành phần liên thông (connected components).

2.  **Vector hóa PD thành Persistence Image (PI):** Do PD là một tập hợp điểm với kích thước thay đổi, chúng tôi chuyển đổi mỗi $PD(S_d)$ thành một **Persistence Image** $\text{PI}(S_d)$ có kích thước cố định. PI là một biểu diễn bền vững, có thể vi phân, được tạo ra bằng cách đặt một hàm Gaussian tại mỗi điểm trong PD và tổng hợp lại trên một lưới.

3.  **Tính Progression Loss:** Chúng tôi định nghĩa loss là tổng bình phương khoảng cách L2 giữa các PI của các lát cắt liên tiếp. Điều này khuyến khích chuỗi PI thay đổi một cách mượt mà.

    Công thức được định nghĩa như sau:
    $$
    \mathcal{L}_{\text{prog}}(S) = \frac{1}{D-1} \sum_{d=0}^{D-2} \| \text{PI}(S_d) - \text{PI}(S_{d+1}) \|_2^2
    $$
    Trong đó $D$ là tổng số lát cắt. Loss này được áp dụng trực tiếp lên dự đoán của student model.

---

## 4. Pipeline tổng thể và Đóng góp

Mô hình của chúng tôi được huấn luyện end-to-end. Trong mỗi vòng lặp, mô hình không chỉ học từ dữ liệu có nhãn và không nhãn mà còn tự regularize để đảm bảo kết quả phân đoạn cuối cùng vừa chính xác ở cấp độ pixel, vừa nhất quán về cấu trúc không gian 3D.

**Đóng góp chính của phương pháp này là:**
-   Đề xuất một hàm loss mới, **Topological Progression Loss**, để mô hình hóa và khuyến khích sự tiến triển mượt mà của cấu trúc topo qua các lát cắt MRI.
-   Tích hợp thành công loss này vào một framework học bán giám sát mạnh mẽ (DyCON), tạo ra một phương pháp mới có khả năng cho kết quả phân đoạn chính xác và hợp lý hơn về mặt giải phẫu. 