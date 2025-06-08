import numpy as np
import gudhi as gd
from pylab import *
import torch
import math
from ripser import ripser
import cripser as cr
import os
from gudhi.wasserstein import wasserstein_distance

# =============================
# Hàm getCriticalPoints_cr
# =============================
# Chức năng:
#   - Tính persistence diagram (biểu đồ tồn tại) từ bản đồ xác suất (likelihood map) của ảnh.
#   - Phân tách các đặc trưng topo thành tín hiệu (signal) và nhiễu (noise) dựa trên độ dài sống (persistence).
# Liên hệ paper:
#   - Đây là bước Signal-Noise Decomposition (Sec 3.3, Eq. 3).
# Ý nghĩa thực tiễn:
#   - Giúp xác định đâu là cấu trúc hình thái học quan trọng (signal) và đâu là chi tiết nhỏ, không quan trọng (noise).
#
# =============================
# Ví dụ input/output:
#   input: likelihood = np.array([[0.1, 0.2, 0.8], [0.2, 0.9, 0.7], [0.1, 0.3, 0.6]]), threshold = 0.3
#   output: (pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx)
#   - pd_lh: mảng các cặp (birth, death) cho từng đặc trưng topo
#   - bcp_lh, dcp_lh: toạ độ điểm sinh/chết
#   - valid_idx: chỉ số các đặc trưng là tín hiệu
#   - noisy_idx: chỉ số các đặc trưng là nhiễu
# =============================
def getCriticalPoints_cr(likelihood, threshold):
    # Đảo ngược likelihood để phù hợp với cripser (vì cripser coi giá trị nhỏ là foreground)
    lh = 1 - likelihood
    # Tính persistent homology (PH) cho ảnh, lấy các đặc trưng topo tới bậc 1
    pd = cr.computePH(lh, maxdim=1, location="birth")
    # Lọc ra các đặc trưng topo bậc 0 (connected components)
    pd_arr_lh = pd[pd[:, 0] == 0] # 0-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time và death time
    # Lấy toạ độ điểm sinh (birth) và điểm chết (death) trên ảnh
    bcp_lh = pd_arr_lh[:, 3:5]
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None
    # Nếu death time > 1.0 thì gán lại thành 1.0 (chuẩn hoá)
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    # Tính persistence (độ dài sống) cho mỗi đặc trưng topo
    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    # Phân tách: persistence > threshold là tín hiệu, ngược lại là nhiễu
    valid_idx = np.where(pd_pers > threshold)[0]   # Tín hiệu: persistence > threshold
    noisy_idx = np.where(pd_pers <= threshold)[0]  # Nhiễu: persistence <= threshold
    # Trả về các thông tin cần thiết cho các bước tiếp theo
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx

# =============================
# Hàm get_matchings
# =============================
# Chức năng:
#   - Tìm matching tối ưu giữa hai persistence diagram (student và teacher) bằng Wasserstein distance.
# Liên hệ paper:
#   - Đây là bước Signal Topology Consistency Loss (Sec 3.3, Eq. 1, 4).
# Ý nghĩa thực tiễn:
#   - Đảm bảo các cấu trúc topo quan trọng của student sẽ học theo teacher, giúp mô hình ổn định hơn.
#
# =============================
# Ví dụ input/output:
#   input: lh_stu = np.array([[0.1, 0.5], [0.2, 0.7]]), lh_tea = np.array([[0.15, 0.55]])
#   output: (dgm1_to_diagonal, off_diagonal_match)
#   - dgm1_to_diagonal: chỉ số các điểm của student không match được với teacher
#   - off_diagonal_match: các cặp chỉ số match giữa student và teacher
# =============================
def get_matchings(lh_stu, lh_tea):
    cost, matchings = wasserstein_distance(lh_stu, lh_tea, matching=True)
    # Các điểm chỉ match với đường chéo (không có cặp tương ứng)
    dgm1_to_diagonal = matchings[matchings[:,1] == -1, 0]
    dgm2_to_diagonal = matchings[matchings[:,0] == -1, 1]
    # Các cặp match thực sự giữa hai diagram
    off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)
    return dgm1_to_diagonal, off_diagonal_match

# =============================
# Hàm compute_dgm_force
# =============================
# Chức năng:
#   - Xác định các đặc trưng topo của student cần loại bỏ (nhiễu) và các cặp cần match với teacher (tín hiệu).
# Liên hệ paper:
#   - Kết hợp cả hai loss: Signal Consistency và Noise Removal.
# Ý nghĩa thực tiễn:
#   - Giúp student học đúng các cấu trúc quan trọng, loại bỏ các chi tiết không cần thiết.
#
# =============================
# Ví dụ input/output:
#   input: stu_lh_dgm = np.array([[0.1, 0.5], [0.2, 0.7]]), tea_lh_dgm = np.array([[0.15, 0.55]])
#   output: (idx_holes_to_remove, off_diagonal_match)
#   - idx_holes_to_remove: chỉ số các đặc trưng của student cần loại bỏ
#   - off_diagonal_match: các cặp chỉ số match giữa student và teacher
# =============================
def compute_dgm_force(stu_lh_dgm, tea_lh_dgm):
    if stu_lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match
    # Nếu teacher không có đặc trưng topo nào, loại bỏ hết của student
    if (tea_lh_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(tea_lh_dgm[:, 1] - tea_lh_dgm[:, 0])
        tea_n_holes = tea_pers.size
    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(stu_lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        # Nếu có, tính matching tối ưu
        idx_holes_to_remove, off_diagonal_match = get_matchings(stu_lh_dgm, tea_lh_dgm)
    return idx_holes_to_remove, off_diagonal_match

# =============================
# Hàm getTopoLoss
# =============================
# Chức năng:
#   - Tính tổng loss topo cho một ảnh hoặc patch, gồm cả loss nhất quán topo (tín hiệu) và loss loại bỏ nhiễu.
# Liên hệ paper:
#   - Tổng hợp hai loss: Signal Consistency Loss (Eq. 4, 5) và Noise Removal Loss (Eq. 6, 7), tổng loss (Eq. 8).
# Ý nghĩa thực tiễn:
#   - Giúp mô hình học được các đặc trưng hình thái học thực sự quan trọng, tăng tính ổn định và khả năng tổng quát của mô hình segmentation.
#
# =============================
# Ví dụ input/output:
#   input:
#     stu_tensor = torch.tensor([[0.1, 0.2, 0.8], [0.2, 0.9, 0.7], [0.1, 0.3, 0.6]]).cuda()
#     tea_tensor = torch.tensor([[0.15, 0.25, 0.75], [0.18, 0.88, 0.68], [0.12, 0.28, 0.65]]).cuda()
#     topo_size = 2, pd_threshold = 0.3
#   output: loss_topo (giá trị số thực, tổng loss topo cho ảnh)
# =============================
def getTopoLoss(stu_tensor, tea_tensor, topo_size=100, pd_threshold=0.7, loss_mode="mse"):
    """
    Tính toán tổng loss topo cho một ảnh hoặc patch.
    - Loss này gồm loss nhất quán topo (student match teacher) và loss loại bỏ nhiễu (đẩy các điểm nhiễu về đường chéo).
    - Tham khảo Eq. (8) trong paper.
    -
    Các bước chính:
    1. Chuyển đổi dữ liệu sang numpy để xử lý.
    2. Khởi tạo bản đồ trọng số và tham chiếu cho các điểm tới hạn.
    3. Duyệt từng patch nhỏ để giảm chi phí tính toán và tăng tính cục bộ.
    4. Tính persistence diagram cho từng patch (cả student và teacher).
    5. Phân tách tín hiệu/nhiễu cho từng patch.
    6. Tìm matching tối ưu giữa các điểm tín hiệu student và teacher.
    7. Tích luỹ loss:
       - Tín hiệu: Đẩy birth/death của student về đúng birth/death của teacher (giữ nhất quán topo).
       - Nhiễu: Đẩy các điểm nhiễu về đường chéo (birth=death), tức là loại bỏ các cấu trúc topo không quan trọng.
    8. Tính loss tổng: Dùng MSE giữa các điểm tới hạn của student và giá trị tham chiếu (từ teacher hoặc từ chính student nếu là nhiễu).
    """
    if stu_tensor.ndim != 2:
        print("incorrct dimension")
    # Chuyển tensor sang numpy để xử lý
    likelihood = stu_tensor.clone()
    gt = tea_tensor.clone()
    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()
    # Khởi tạo bản đồ trọng số và tham chiếu cho các điểm tới hạn
    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)
    # Duyệt qua từng patch nhỏ của ảnh
    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):
            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                         x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                         x:min(x + topo_size, gt.shape[1])]
            # Bỏ qua patch toàn 1 hoặc toàn 0 (không có cấu trúc topo)
            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue
            # --- Tính persistence diagram cho student và teacher ---
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx_lh, noisy_idx_lh = getCriticalPoints_cr(lh_patch, threshold=pd_threshold)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt, valid_idx_gt, noisy_idx_gt = getCriticalPoints_cr(gt_patch, threshold=pd_threshold)
            # Chỉ lấy các điểm tín hiệu để match
            pd_lh_for_matching = pd_lh[valid_idx_lh]
            pd_gt_for_matching = pd_gt[valid_idx_gt]
            # Nếu không có cặp hợp lệ thì bỏ qua
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue
            # --- Tính matching tối ưu và các chỉ số nhiễu ---
            idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh_for_matching, pd_gt_for_matching)
            idx_holes_to_remove = []
            off_diagonal_match = []
            # Map lại index về PD gốc
            if (len(idx_holes_to_remove_for_matching) > 0):
                for i in idx_holes_to_remove_for_matching:
                    index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
                    idx_holes_to_remove.append(index_pd_lh_removed)
            # Thêm các chỉ số nhiễu
            for k in noisy_idx_lh:
                idx_holes_to_remove.append(k)
            # Map các cặp match
            if len(off_diagonal_for_matching) > 0:
                for idx, (i, j) in enumerate(off_diagonal_for_matching):
                    index_pd_lh = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
                    index_pd_gt = np.where(np.all(pd_gt == pd_gt_for_matching[j], axis=1))[0][0]
                    off_diagonal_match.append((index_pd_lh, index_pd_gt))
            # --- Tích luỹ loss cho tín hiệu và nhiễu ---
            if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
                for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
                    # Nhất quán topo: match birth/death student với teacher
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1 # đẩy birth về birth của teacher
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = pd_gt[j][0]
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # đẩy death về death của teacher
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = pd_gt[j][1]
                for hole_indx in idx_holes_to_remove:
                    # Loại bỏ nhiễu: đẩy các điểm nhiễu về đường chéo (birth=death)
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # đẩy về đường chéo
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # đẩy về đường chéo
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0
    # Chuyển sang tensor để tính loss, đảm bảo cùng device với input
    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float, device=stu_tensor.device)
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float, device=stu_tensor.device)
    # --- Tổng loss topo: gồm loss nhất quán và loss loại bỏ nhiễu (Eq. 8) ---
    loss_topo = (((stu_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    return loss_topo

# =============================
# Hàm main: Chạy thử các bước với dữ liệu ví dụ nhỏ
# =============================
if __name__ == "__main__":
    import torch
    import numpy as np
    print("=== DEMO TOPO LOSS STEP BY STEP ===")
    # Ưu tiên dùng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Sử dụng device:", device)
    # Tạo dữ liệu ví dụ nhỏ
    likelihood = np.array([[0.1, 0.2, 0.8], [0.2, 0.9, 0.7], [0.1, 0.3, 0.6]])
    gt = np.array([[0.15, 0.25, 0.75], [0.18, 0.88, 0.68], [0.12, 0.28, 0.65]])
    threshold = 0.3
    print("Input likelihood:\n", likelihood)
    print("Input ground truth:\n", gt)
    # Step 1: Tính persistence diagram cho likelihood
    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx = getCriticalPoints_cr(likelihood, threshold)
    print("\n[Step 1] Persistence diagram (likelihood):")
    print("  pd_lh (birth, death):\n", pd_lh)
    print("  bcp_lh (birth coord):\n", bcp_lh)
    print("  dcp_lh (death coord):\n", dcp_lh)
    print("  valid_idx (signal):", valid_idx)
    print("  noisy_idx (noise):", noisy_idx)
    # Step 2: Tính persistence diagram cho ground truth
    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt, valid_idx_gt, noisy_idx_gt = getCriticalPoints_cr(gt, threshold)
    print("\n[Step 2] Persistence diagram (ground truth):")
    print("  pd_gt (birth, death):\n", pd_gt)
    print("  bcp_gt (birth coord):\n", bcp_gt)
    print("  dcp_gt (death coord):\n", dcp_gt)
    print("  valid_idx_gt (signal):", valid_idx_gt)
    print("  noisy_idx_gt (noise):", noisy_idx_gt)
    # Step 3: Matching tối ưu giữa các điểm tín hiệu
    pd_lh_for_matching = pd_lh[valid_idx]
    pd_gt_for_matching = pd_gt[valid_idx_gt]
    if len(pd_lh_for_matching) > 0 and len(pd_gt_for_matching) > 0:
        dgm1_to_diagonal, off_diagonal_match = get_matchings(pd_lh_for_matching, pd_gt_for_matching)
        print("\n[Step 3] Matching giữa signal student và teacher:")
        print("  dgm1_to_diagonal (student unmatched):", dgm1_to_diagonal)
        print("  off_diagonal_match (matched pairs):", off_diagonal_match)
    else:
        print("\n[Step 3] Không có điểm tín hiệu để match!")
    # Step 4: Tính loss topo tổng hợp (dùng torch tensor)
    stu_tensor = torch.tensor(likelihood, dtype=torch.float32, device=device)
    tea_tensor = torch.tensor(gt, dtype=torch.float32, device=device)
    loss_topo = getTopoLoss(stu_tensor, tea_tensor, topo_size=2, pd_threshold=threshold)
    print("\n[Step 4] Tổng loss topo tính được:", loss_topo.item())
    print("=== END DEMO ===")
