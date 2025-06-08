# Phương pháp đề xuất: DyCON kết hợp Topological Loss cho phân đoạn ảnh y tế bán giám sát

## Tổng quan

Chúng tôi đề xuất một framework phân đoạn ảnh y tế bán giám sát dựa trên kiến trúc DyCON, kết hợp giữa học nhất quán (consistency learning) có nhận biết bất định (uncertainty-aware) và học tương phản (contrastive learning). Ngoài ra, chúng tôi tích hợp thêm **topological loss** để tăng cường tính nhất quán về mặt cấu trúc hình học của kết quả phân đoạn.

Hàm mất mát tổng thể trong quá trình huấn luyện là tổng có trọng số của các thành phần:

$$
\mathcal{L}_{\text{total}} = \lambda_s \mathcal{L}_{\text{sup}} + \lambda_u \mathcal{L}_{\text{cons}} + \lambda_{c} (\mathcal{L}_{\text{FeCL}} + \mathcal{L}_{\text{UnCL}}) + \lambda_{t} \mathcal{L}_{\text{topo}}
$$

Trong đó:
- $\mathcal{L}_{\text{sup}}$: Mất mát giám sát (Dice + Cross-Entropy) trên dữ liệu có nhãn.
- $\mathcal{L}_{\text{cons}}$: Mất mát nhất quán giữa dự đoán của student và teacher trên dữ liệu không nhãn.
- $\mathcal{L}_{\text{FeCL}}$: Feature-level Contrastive Loss (tìm hard negative, có trọng số uncertainty).
- $\mathcal{L}_{\text{UnCL}}$: Uncertainty-aware Contrastive Loss cho dữ liệu không nhãn.
- $\mathcal{L}_{\text{topo}}$: Topological loss đảm bảo tính nhất quán về cấu trúc topo.
- $\lambda_s, \lambda_u, \lambda_c, \lambda_t$: Trọng số cho từng thành phần loss.

---

## 1. Mất mát giám sát (Supervised Loss)

Trên dữ liệu có nhãn, sử dụng kết hợp Dice loss và Cross-Entropy loss:

$$
\mathcal{L}_{\text{sup}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}
$$

---

## 2. Mất mát nhất quán (Consistency Loss)

Áp dụng mô hình Mean Teacher: student và teacher nhận cùng một ảnh (có augmentation/noise khác nhau). Mất mát này khuyến khích dự đoán của hai mô hình gần nhau trên dữ liệu không nhãn:

$$
\mathcal{L}_{\text{cons}} = \frac{1}{N_u} \sum_{i \in \text{unlabeled}} \| p^{(s)}_i - p^{(t)}_i \|^2
$$

Với $p^{(s)}_i$, $p^{(t)}_i$ là xác suất softmax của student và teacher.

---

## 3. Mất mát tương phản (Contrastive Loss)

### a. Feature-level Contrastive Loss (FeCL)
Khuyến khích embedding của cùng lớp gần nhau, khác lớp thì xa nhau, có hard negative mining và trọng số uncertainty:

$$
\mathcal{L}_{\text{FeCL}} = \text{ContrastiveLoss}(f^{(s)}, f^{(t)}, \text{mask}, \gamma, T)
$$

Với $f^{(s)}, f^{(t)}$ là embedding của student/teacher, $\gamma$ là hệ số tập trung, $T$ là nhiệt độ.

### b. Uncertainty-aware Contrastive Loss (UnCL)
Tính cho dữ liệu không nhãn, có trọng số dựa trên độ bất định:

$$
\mathcal{L}_{\text{UnCL}} = \sum_{i,j} w_{ij} \cdot \text{Contrastive}(f_i, f_j)
$$

Với $w_{ij}$ là trọng số dựa trên uncertainty.

---

## 4. Topological Loss

Để tăng cường tính đúng đắn về mặt cấu trúc hình học, ta thêm topological loss, đo sự khác biệt về đặc trưng topo (ví dụ Betti number, persistence diagram) giữa dự đoán của student và teacher:

$$
\mathcal{L}_{\text{topo}} = \frac{1}{N_l} \sum_{i \in \text{labeled}} d_{\text{topo}}(S^{(s)}_i, S^{(t)}_i)
$$

Với $d_{\text{topo}}$ là khoảng cách (ví dụ Wasserstein) giữa các đặc trưng topo của output student và teacher.

---

## 5. Dynamic Weighting

Các trọng số $\lambda_u, \lambda_t$ cho loss không giám sát và topo được tăng dần theo lịch sigmoid để ổn định quá trình huấn luyện ban đầu.

---

## Pipeline tổng thể

1. **Input**: Lấy batch gồm cả ảnh có nhãn và không nhãn.
2. **Augmentation**: Student và teacher nhận các phiên bản augment khác nhau.
3. **Forward**: Tính dự đoán và embedding cho cả hai mô hình.
4. **Loss**: Tính tất cả các thành phần loss như trên.
5. **EMA Update**: Cập nhật teacher bằng EMA của student.
6. **Tối ưu**: Lan truyền ngược và cập nhật student.

---

## Sơ đồ pipeline

```
Ảnh đầu vào ──► [Student Model] ──► Output phân đoạn ──┬─► Embedding ──► FeCL, UnCL
         │                │                            │
         │                │                            └─► Đặc trưng topo
         │                │
         │                └─────────────────────────────┐
         │                                              │
         └─► [Teacher Model (EMA)] ──► Output phân đoạn ─┴─► Đặc trưng topo
```
- **Consistency Loss**: giữa output student và teacher.
- **Contrastive Loss**: giữa embedding.
- **Topological Loss**: giữa đặc trưng topo của output student và teacher.

---

**Tài liệu tham khảo:**
- DyCON: Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation (arXiv:2504.04566)
- [Bổ sung paper về topo loss nếu cần] 