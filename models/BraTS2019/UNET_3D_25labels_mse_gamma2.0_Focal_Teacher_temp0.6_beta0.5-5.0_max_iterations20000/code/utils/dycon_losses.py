import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
    ratio = min_beta / max_beta
    exponent = epoch / total_epochs
    beta = max_beta * (ratio ** exponent)
    return beta
    
def gambling_softmax(logits):
    """
    Compute gambling softmax probabilities over the channel dimension.
    
    Args:
        logits (Tensor): Input tensor of shape (B, C, ...).
    
    Returns:
        Tensor: Softmax probabilities of the same shape.
    """
    exp_logits = torch.exp(logits)
    denom = torch.sum(exp_logits, dim=1, keepdim=True)
    return exp_logits / (denom + 1e-18)

def sigmoid_rampup(current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0):
    """
    Compute a dynamic threshold using a sigmoid ramp-up schedule.

    Args:
        current_epoch (int or float): The current training epoch.
        total_rampup_epochs (int or float): The number of epochs over which to ramp up the threshold.
        min_threshold (float): The initial threshold value, chosen based on the histogram's lower tail.
        max_threshold (float): The target threshold value after ramp-up.
        steepness (float, optional): Controls how quickly the threshold ramps up (default=5.0).

    Returns:
        float: The computed threshold for the current epoch.
    """
    if total_rampup_epochs == 0:
        return max_threshold
    current_epoch = max(0.0, min(float(current_epoch), total_rampup_epochs))
    phase = 1.0 - (current_epoch / total_rampup_epochs)
    ramp = math.exp(-steepness * (phase ** 2))
    return min_threshold + (max_threshold - min_threshold) * ramp


class UnCLoss(nn.Module):
    """
    UnCLoss implements an uncertainty-aware consistency loss that compares the prediction distributions 
    from a student and a teacher network. It is designed for semi-supervised learning scenarios, where 
    the teacher network provides guidance (e.g., via noise-added views) to the student network.
    
    Công thức đúng như paper DyCON (Eq. 1):
    
        L_UnCL = 1/N * sum_i [ L(p^s_i, p^t_i) / (exp(beta*H_s(p^s_i)) + exp(beta*H_t(p^t_i))) ]
                  + beta/N * sum_i [ H_s(p^s_i) + H_t(p^t_i) ]
    
    - p^s_i: softmax xác suất student tại voxel i (code: p_s)
    - p^t_i: softmax xác suất teacher tại voxel i (code: p_t)
    - H_s, H_t: entropy student/teacher tại voxel i (code: H_s, H_t)
    - L: loss (ở đây là MSE) giữa student và teacher (code: (p_s - p_t)^2)
    - beta: hệ số điều chỉnh mức độ ảnh hưởng của entropy
    - N: tổng số voxel (tự động trung bình hóa trong code)
    """
    def __init__(self):
        super(UnCLoss, self).__init__()

    def forward(self, s_logits, t_logits, beta):
        EPS = 1e-6

        # 1. Tính softmax xác suất cho student và teacher
        #   - p^s_i (paper) <-> p_s (code), shape: (B, C, H, W, D)
        #   - p^t_i (paper) <-> p_t (code), shape: (B, C, H, W, D)
        p_s = F.softmax(s_logits, dim=1)  # Student probability map
        p_t = F.softmax(t_logits, dim=1)  # Teacher probability map

        # 2. Tính entropy cho student và teacher
        #   - H_s(p^s_i) = -sum_c p^s_i * log(p^s_i), shape: (B, 1, H, W, D)
        #   - H_t(p^t_i) = -sum_c p^t_i * log(p^t_i), shape: (B, 1, H, W, D)
        p_s_log = torch.log(p_s + EPS)
        H_s = -torch.sum(p_s * p_s_log, dim=1, keepdim=True)  # Student entropy
        p_t_log = torch.log(p_t + EPS)
        H_t = -torch.sum(p_t * p_t_log, dim=1, keepdim=True)  # Teacher entropy
        # Ý nghĩa: entropy càng thấp, model càng chắc chắn; entropy cao -> không chắc chắn

        # 3. Exponentiate entropy với beta
        #   - exp(beta * H_s(p^s_i)) <-> exp_H_s (code), shape: (B, 1, H, W, D)
        #   - exp(beta * H_t(p^t_i)) <-> exp_H_t (code), shape: (B, 1, H, W, D)
        exp_H_s = torch.exp(beta * H_s)
        exp_H_t = torch.exp(beta * H_t)
        # Ý nghĩa: beta càng lớn, vùng không chắc chắn (entropy cao) càng bị giảm ảnh hưởng

        # 4. Tính loss chính: (p_s - p_t)^2 / (exp_H_s + exp_H_t)
        #   - Tương ứng với: L(p^s_i, p^t_i) / (exp(beta*H_s) + exp(beta*H_t))
        #   - L ở đây là MSE: (p_s - p_t)^2
        #   - Trọng số entropy giúp giảm ảnh hưởng vùng không chắc chắn (entropy cao)
        #   - Shape: (B, C, H, W, D)
        loss = (p_s - p_t)**2 / (exp_H_s + exp_H_t)

        # 5. Tổng hợp loss:
        #   - sum theo class (dim=1), shape còn lại: (B, H, W, D)
        #   - cộng regularization entropy (beta*(H_s+H_t)), shape: (B, 1, H, W, D)
        #   - rồi mean toàn bộ (tự động chia N như công thức paper)
        #   - Dòng đầu tiên công thức (1): loss.sum(dim=1)
        #   - Dòng thứ hai công thức (1): beta*(H_s + H_t)
        loss = torch.mean(loss.sum(dim=1) + beta * (H_s + H_t))
        # Ý nghĩa: loss này vừa khuyến khích student-teacher nhất quán ở vùng chắc chắn, vừa phạt vùng không chắc chắn

        # 6. Trung bình hóa toàn batch, không gian, v.v. (tự động chia N)
        return loss.mean()

class FeCLoss(nn.Module):
    """
    FeCLoss implements the Feature-level Contrastive Loss with dual focal weighting and hard negative mining,
    as described in DyCON (Eq. 3, 4, 5 in paper):

        L_FeCL = 1/|P(i)| sum_{k in P(i)} F^+_k * [ -log( exp(S_ik) / D(i) ) ]
        D(i) = exp(S_ik) + sum_{q in N(i)} F^-_q * [ exp(S_iq) + 1/K sum_{l=1}^K exp(S_il) ]
        F^+_k = (1 - S_ik)^gamma * exp(H_gs(p^s_h))   (positive focal weight, Eq.4)
        F^-_q = (S_iq)^gamma                        (negative focal weight, Eq.4)
        p^s_h = Gambling Softmax (Eq.5)

    - feat: patch embedding của student (code: feat, shape: (B, N, D))
    - mask: nhãn patch (code: mask, shape: (B, 1, N))
    - teacher_feat: embedding của teacher (phục vụ auxiliary loss)
    - gamma: tham số điều chỉnh độ tập trung focal
    - temperature: tham số nhiệt cho softmax tương tự InfoNCE
    - gambling_uncertainty: entropy từ Gambling Softmax (điều chỉnh trọng số positive)
    - use_focal: bật/tắt dual focal weights
    - rampup_epochs: số epoch để điều chỉnh ngưỡng hard positive/negative
    """
    def __init__(self, device, temperature=0.6, gamma=2.0, use_focal=False, rampup_epochs=2000, lambda_cross=1.0):
        super(FeCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross

    def forward(self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0):
        """
        Args:
            feat: (B, N, D) - Student patch embeddings (z^s in paper)
            mask: (B, 1, N) - Patch labels (same class: positive, diff class: negative)
            teacher_feat: (B, N, D) - Teacher patch embeddings (for auxiliary loss)
            gambling_uncertainty: (B, N) - Entropy from Gambling Softmax (for positive focal weight)
            epoch: current epoch (for dynamic threshold)
        Returns:
            Total loss (scalar): student loss + lambda_cross * teacher auxiliary loss
        """
        B, N, _ = feat.shape

        # 1. Tạo positive/negative mask giữa các patch
        #   - mem_mask: (B, N, N), 1 nếu cùng class (positive), 0 nếu khác class (negative)
        #   - mem_mask_neg: (B, N, N), 1 nếu khác class
        #   - mem_mask <-> 1_{y_i = y_k} trong paper
        print("[DEBUG] mask shape:", mask.shape)
        print("[DEBUG] mask (first 3x3 block, batch 0):\n", mask[0, :3, :3].detach().cpu().numpy())
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # Positive mask
        print("[DEBUG] mem_mask shape:", mem_mask.shape)
        print("[DEBUG] mem_mask (first 3x3 block, batch 0):\n", mem_mask[0, :3, :3].detach().cpu().numpy())
        mem_mask_neg = 1 - mem_mask  # Negative mask
        print("[DEBUG] mem_mask_neg (first 3x3 block, batch 0):\n", mem_mask_neg[0, :3, :3].detach().cpu().numpy())
        # 2. Tính similarity giữa các patch (student)
        #   - feat_logits[b, i, j] = S_ij trong paper (Eq.3,4)
        #   - S_ik: positive pair (mask[b, i, k] == 1)
        #   - S_iq: negative pair (mask_neg[b, i, q] == 1)
        #   - S_il: similarity với mọi patch l (dùng cho normalization)
        feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # S_ij
        if B > 0:
            print("[DEBUG] feat_logits (S_ij) shape:", feat_logits.shape)
            print("[DEBUG] S_ij (first 3x3 block, batch 0):\n", feat_logits[0, :3, :3].detach().cpu().numpy())
        identity = torch.eye(N, device=self.device)
        neg_identity = 1 - identity  # Loại self-similarity
        feat_logits = feat_logits * neg_identity # Loại self-similarity, loại bỏ đường chéo

        # 3. Chuẩn hóa logits để tránh overflow
        feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
        feat_logits = feat_logits - feat_logits_max.detach()

        # 4. Tính exp(similarity) cho InfoNCE denominator
        #   - exp_logits[b, i, j] = exp(S_ij)
        #   - exp(S_ik): positive, exp(S_iq): negative, exp(S_il): mọi patch
        exp_logits = torch.exp(feat_logits)  # (B, N, N) exp(S_ij)
        if B > 0:
            print("[DEBUG] exp_logits (exp(S_ij)) shape:", exp_logits.shape)
            print("[DEBUG] exp(S_ij) (first 3x3 block, batch 0):\n", exp_logits[0, :3, :3].detach().cpu().numpy())
        neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)  # (B, N) sum(exp(S_iq))
        if B > 0:
            print("[DEBUG] neg_sum (sum_q exp(S_iq)) shape:", neg_sum.shape)
            print("[DEBUG] neg_sum (first 3, batch 0):", neg_sum[0, :3].detach().cpu().numpy())

        # 5. Tính denominator D(i) 
        #   - D(i) = exp(S_ik) + sum_{q in N(i)} F^-_q * [ exp(S_iq) + 1/K sum_{l=1}^K exp(S_il) ]
        # Hiện tại chỉ có exp(S_ik) và sum_{q in N(i)} exp(S_iq) không có F^-_q là focal weight và 1/K sum_{l=1}^K exp(S_il) là trung bình similarity của mọi patch
        denominator = exp_logits + neg_sum.unsqueeze(dim=-1)  # D(i) = exp(S_ik) + sum_{q in N(i)} F^-_q * [ exp(S_iq) + 1/K sum_{l=1}^K exp(S_il) ]
        if B > 0:
            print("[DEBUG] denominator D(i) shape:", denominator.shape)
            print("[DEBUG] D(i) (first 3x3 block, batch 0):\n", denominator[0, :3, :3].detach().cpu().numpy())
        
        # 6. Tính softmax-like probability
        #   - division[b, i, j] = exp(S_ij) / D(i)
        division = exp_logits / (denominator + 1e-18)  # Softmax-like probability
        if B > 0:
            print("[DEBUG] division (exp(S_ij)/D(i)) shape:", division.shape)
            print("[DEBUG] division (first 3x3 block, batch 0):\n", division[0, :3, :3].detach().cpu().numpy())

        # 7. Tính loss matrix
        #   - loss_matrix[b, i, j] = -log(division[b, i, j])
        #   - loss_matrix = loss_matrix * mem_mask * neg_identity
        loss_matrix = -torch.log(division + 1e-18)

        # 8. Tính loss cho student
        # - loss_matrix là loss tại mỗi patch i với tất cả patch j
        # - mem_mask[b, i, j] = 1 nếu patch i và j cùng class (positive pair). -> chỉ tính loss tại các positive pair (exp(S_{ik}))
        # - neg_identity loại trừ self-similarity
        loss_matrix = loss_matrix * mem_mask * neg_identity # Loại bỏ loss tại patch i với chính nó (self-similarity)
        if B > 0:
            print("[DEBUG] loss_matrix shape:", loss_matrix.shape)
            print("[DEBUG] loss_matrix (first 3x3 block, batch 0):\n", loss_matrix[0, :3, :3].detach().cpu().numpy())

        loss_student = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
        loss_student = loss_student.mean()
        print("[DEBUG] loss_student:", loss_student.item())

        if self.use_focal:
            similarity = division  # S_ik trong paper
            focal_weights = torch.ones_like(similarity)
            # Ngưỡng động cho hard positive/negative (rampup theo epoch)
            pos_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=1.3, max_threshold=1.5)
            neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
            # Hard positive: positive pair có similarity thấp
            #   - F^+_k = (1 - S_ik)^gamma * exp(H_gs(p^s_h)) (Eq.4)
            hard_pos_mask = mem_mask.bool() & (similarity < pos_thresh)
            focal_weights[hard_pos_mask] = (1 - similarity[hard_pos_mask]).pow(self.gamma)
            # Hard negative: negative pair có similarity cao
            #   - F^-_q = (S_iq)^gamma (Eq.4)
            hard_neg_mask = mem_mask_neg.bool() & (similarity > neg_thresh)
            focal_weights[hard_neg_mask] = similarity[hard_neg_mask].pow(self.gamma)
            if B > 0:
                print("[DEBUG] focal_weights (first 3x3 block, batch 0):\n", focal_weights[0, :3, :3].detach().cpu().numpy())

            # Tính loss tại mỗi patch i với tất cả patch j
            loss_student = torch.sum(loss_matrix * focal_weights, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
            loss_student = loss_student.mean()
            print("[DEBUG] loss_student (with focal):", loss_student.item())

        # 8. Gambling Softmax entropy mask cho positive (Eq.5):
        #   - Nếu có gambling_uncertainty, dùng nó để điều chỉnh trọng số positive
        #   - p^s_h trong paper (Eq.5)
        if gambling_uncertainty is not None:
            loss_student_per_patch = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18) 
            loss_student = (loss_student_per_patch * gambling_uncertainty).mean()
            # Ý nghĩa: tập trung vào vùng biên không chắc chắn nhờ entropy

        # 9. Auxiliary Cross-Negative Loss (Teacher-Student):
        #   - Tăng khả năng phân biệt giữa student và teacher ở các negative pairs khó
        #   - cross_sim: S_iq giữa student và teacher
        loss_cross = 0.0
        if teacher_feat is not None:
            # Tính similarity giữa student và teacher embedding
            cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2))  # S_iq giữa student-teacher
            mem_mask_cross = torch.eq(mask, mask.transpose(1, 2)).float()
            mem_mask_cross_neg = 1 - mem_mask_cross  # Negative mask
            # Ngưỡng động cho hard negative
            cross_neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
            cross_hard_neg_mask = mem_mask_cross_neg.bool() & (cross_sim > cross_neg_thresh)
            # Phạt các negative pair khó (similarity cao)
            if cross_hard_neg_mask.sum() > 0:
                # -log(1 - S_iq) cho hard negative
                loss_cross_term = -torch.log(1 - cross_sim + 1e-18)
                loss_cross_term = loss_cross_term * cross_hard_neg_mask.float()
                loss_cross = torch.sum(loss_cross_term) / (torch.sum(cross_hard_neg_mask.float()) + 1e-18)
                print("[DEBUG] loss_cross (auxiliary):", loss_cross)
            else:
                loss_cross = 0.0

        # 10. Tổng hợp loss: student contrastive + auxiliary teacher loss
        total_loss = loss_student + self.lambda_cross * loss_cross
        print("[DEBUG] total_loss:", total_loss.item())
        return total_loss

if __name__ == "__main__":
    # Test the UnCLoss
    s_logits = torch.randn(8, 2, 16, 16, 16)
    t_logits = torch.randn(8, 2, 16, 16, 16)
    beta = 0.8
    uncl = UnCLoss()
    loss = uncl(s_logits, t_logits, beta)
    print(f"uncl_loss: {loss}")
    
    # Test the FeCLoss
    feat = torch.randn(8, 128, 128).cuda()
    mask = torch.randint(0, 2, (8, 1, 128)).cuda()
    decoded_logits = torch.randn(8, 128).cuda()
    
    fecl = FeCLoss(device='cuda:0', use_focal=True)
    loss = fecl(feat=feat, mask=mask, teacher_feat=None, gambling_uncertainty=decoded_logits)
    print(f"fecl_loss: {loss}")
