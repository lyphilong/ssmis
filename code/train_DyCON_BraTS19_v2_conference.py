import os
import sys
import shutil
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import wandb

import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from networks.net_factory_3d import net_factory_3d
from utils import ramps, metrics, losses, dycon_losses, test_3d_patch, monitor
from utils.topo_losses import getTopoLoss, getUncertaintyAwareTopoLoss
# NEW: Advanced topology losses for conference submission
from utils.advanced_topo_losses import create_advanced_topology_loss, UncertaintyAwareTopologyLossV2
from dataloaders.brats19 import BraTS2019, SagittalToAxial, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler 

import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ====== WANDB LOGIN HƯỚNG DẪN ======
# Nếu bạn chưa login wandb, hãy chạy lệnh: wandb login
# Hoặc để tự động login bằng dummy key (bạn cần sửa lại sau bằng key thật của bạn)
if not os.path.exists(os.path.expanduser("~/.netrc")) and not os.environ.get("WANDB_API_KEY"):
    # Dummy key, hãy thay bằng key thật của bạn sau!
    os.environ["WANDB_API_KEY"] = "340859a29629cb9f0783ca4213409e89c5eed05f"
    # Hoặc bạn có thể dùng: wandb.login(key="your_dummy_wandb_api_key_here")

# ================= DyCON Training Script for BraTS2019 =====================
# Các comment dưới đây giải thích mối liên hệ giữa code và kiến trúc DyCON trong paper (hình Figure 2)
# --------------------------------------------------------------------------
# 1. Preprocessing Input Volumes (Khối màu xanh nhạt bên trái hình):
#    - Đọc dữ liệu, áp dụng augmentation (RandomCrop, RandomRotFlip, ...)
#    - Chia batch thành labeled và unlabeled (TwoStreamBatchSampler)
# 2. Mean-Teacher với backbone 3D UNet (Khối giữa hình):
#    - Student Model: f^s_θ (model)
#    - Teacher Model: f^t_θ (ema_model, cập nhật bằng EMA)
#    - Cùng forward input qua student/teacher, lấy logits, features
# 3. Loss components (Khối phải hình):
#    - Supervised loss: L_Dice + L_CE (cho batch labeled)
#    - Unsupervised loss: L_UnCL, L_FeCL (cho batch unlabeled)
#    - Consistency loss giữa student/teacher
#    - Tổng hợp loss để backward
# 4. EMA update: Cập nhật teacher model từ student model
# 5. Logging, evaluation, save model
# --------------------------------------------------------------------------

# Argument parsing
parser = argparse.ArgumentParser(description="Training DyCON on BraTS2019 Dataset")

parser.add_argument('--root_dir', type=str, default="../data/BraTS2019", help='Path to BraTS-2019 dataset')
parser.add_argument('--patch_size', type=str, default=[112, 112, 80], help='Input image patch size')

parser.add_argument('--exp', type=str, default='BraTS2019', help='Experiment name')
parser.add_argument('--gpu_id', type=str, default=1, help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training (0 or 1)')

parser.add_argument('--model', type=str, choices=['unet_3D', 'vnet'], default='unet_3D', help='Model architecture')
parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--feature_scaler', type=int, default=2, help='Feature scaling factor for contrastive loss')

#parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum number of training iterations')
#New iterations 
parser.add_argument('--max_iterations', type=int, default=0, help='Maximum number of training iterations')

parser.add_argument('--batch_size', type=int, default=8, help='Total batch size per GPU')
parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')

parser.add_argument('--labelnum', type=int, default=8, help='Number of labeled samples per class')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher model')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Ramp-up duration for consistency weight')

# === DyCon-specific Parameters === #
parser.add_argument('--gamma', type=float, default=2.0, help='Focusing parameter for hard positives/negatives in FeCL (γ)')
parser.add_argument('--beta_min', type=float, default=0.5, help='Minimum value for entropy weighting (β)')
parser.add_argument('--beta_max', type=float, default=5.0, help='Maximum value for entropy weighting (β)')
parser.add_argument('--s_beta', type=float, default=None, help='If provided, use this static beta for UnCLoss instead of adaptive beta.')
parser.add_argument('--temp', type=float, default=0.6, help='Temperature for contrastive softmax scaling (optimal: 0.6)')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight for supervised loss')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight for unsupervised loss')
parser.add_argument('--use_focal', type=int, default=1, help='Whether to use focal weighting (1 for True, 0 for False)')
parser.add_argument('--use_teacher_loss', type=int, default=1, help='Use teacher-based auxiliary loss (1 for True, 0 for False)')

# === Topological Loss Parameters === #
parser.add_argument('--use_topo_loss', type=int, default=0, help='Use topological loss (1 for True, 0 for False)')
parser.add_argument('--topo_weight', type=float, default=0.01, help='Weight for topological loss')
parser.add_argument('--topo_size', type=int, default=32, help='Patch size for topological analysis')
parser.add_argument('--pd_threshold', type=float, default=0.3, help='Persistence diagram threshold for signal/noise separation')
parser.add_argument('--topo_rampup', type=float, default=500.0, help='Ramp-up duration for topological loss weight')

# === NEW: Uncertainty-Aware Topological Loss Parameters === #
parser.add_argument('--use_uncertainty_topo', type=int, default=0, help='Use uncertainty-aware topological loss (1 for True, 0 for False)')
parser.add_argument('--uncertainty_threshold', type=float, default=0.7, help='Threshold for confidence/uncertainty separation')
parser.add_argument('--topo_focus_mode', type=str, choices=['confident', 'uncertain'], default='confident', help='Focus on confident or uncertain regions')

# === CONFERENCE: Advanced Topological Loss Parameters === #
parser.add_argument('--use_advanced_topo', type=int, default=0, help='Use advanced topology loss for conference submission (1 for True, 0 for False)')
parser.add_argument('--lambda_info', type=float, default=0.1, help='Weight for information-theoretic topology loss')
parser.add_argument('--lambda_wasserstein', type=float, default=0.2, help='Weight for Wasserstein topology loss')
parser.add_argument('--lambda_bayesian', type=float, default=0.15, help='Weight for Bayesian topology loss')
parser.add_argument('--use_adaptive_weight', type=int, default=1, help='Use adaptive topology weighting (1 for True, 0 for False)')

args = parser.parse_args()

if args.s_beta is not None:
    beta_str = f"_beta{args.s_beta}"
else:
    beta_str = f"_beta{args.beta_min}-{args.beta_max}"

focal_str = "Focal" if bool(args.use_focal) else "NoFocal"
gamma_str = f"_gamma{args.gamma}" if bool(args.use_focal) else ""
teacher_str = "Teacher" if bool(args.use_teacher_loss) else "NoTeacher"
topo_str = f"_Topo{args.topo_weight}" if bool(args.use_topo_loss) else ""

# NEW: Advanced topology string for path
advanced_topo_str = ""
if bool(args.use_advanced_topo):
    advanced_topo_str = f"_AdvTopo_info{args.lambda_info}_wass{args.lambda_wasserstein}_bay{args.lambda_bayesian}"
    if bool(args.use_adaptive_weight):
        advanced_topo_str += "_adaptive"

snapshot_path = (
    f"../models/{args.exp}/{args.model.upper()}_{args.labelnum}labels_"
    f"{args.consistency_type}{gamma_str}_{focal_str}_{teacher_str}{topo_str}{advanced_topo_str}_temp{args.temp}"
    f"{beta_str}_max_iterations{args.max_iterations}"
)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) # Only GPU `args.gpu_id` is visible

batch_size = args.batch_size 
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = True 
    cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = args.num_classes = 2
patch_size = args.patch_size = (96, 96, 96) # (112, 112, 80)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_topo_weight(epoch):
    # Topological loss ramp-up similar to consistency ramp-up
    # Support both standard and uncertainty-aware topology loss
    use_any_topo = bool(args.use_topo_loss) or bool(args.use_uncertainty_topo) or bool(args.use_advanced_topo)
    if not use_any_topo:
        return 0.0
    
    # Adaptive weight: start smaller, ramp up gradually
    base_weight = args.topo_weight * 0.1  # Start with 10% of specified weight
    return base_weight * ramps.sigmoid_rampup(epoch, args.topo_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

from matplotlib import pyplot as plt

def plot_samples(image, mask, epoch):
    """Plot sample slices of the image/preds and mask"""
    # image: (C, H, W, D), mask: (H, W, D)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image[1][:, :, image.shape[-1]//2], cmap='gray') # access the class at index 1
    ax[1].imshow(mask[:, :, mask.shape[-1]//2], cmap='viridis')
    plt.savefig(f'../misc/train_preds/LA_sample_slice_{str(epoch)}.png')
    plt.close()

def plot_and_save_metrics(metrics_dict, save_dir):
    """
    Vẽ biểu đồ các độ đo (loss, dice, hd95, accuracy, ...) và lưu lại.
    metrics_dict: dict, key là tên metric, value là list các giá trị theo iter_num
    save_dir: thư mục để lưu các biểu đồ
    """
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, values in metrics_dict.items():
        plt.figure()
        plt.plot(values, label=metric_name)
        plt.xlabel('Iteration')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name}.png"))
        plt.close()

if __name__ == "__main__":
    # === 1. Khởi tạo logger, lưu code, chuẩn bị môi trường ===
    # (Không liên quan trực tiếp đến kiến trúc, chỉ để log và lưu lại code chạy)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # === Khởi tạo wandb ===
    wandb.init(
        project="DyCON-BraTS19",
        name=args.exp,
        config=vars(args)
    )

    # === 2. Tạo Student/Teacher Model (Khối Mean-Teacher, hình giữa) ===
    def create_model(ema=False):
        net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes, scaler=args.feature_scaler)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_() # Teacher model không cập nhật gradient
        return model

    model = create_model() # Student model: f^s_θ
    ema_model = create_model(ema=True) # Teacher model: f^t_θ (EMA)
    logging.info("Total params of model: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    # === 3. Chuẩn bị dữ liệu & Augmentation (Khối Preprocessing Input Volumes, hình trái) ===
    db_train = BraTS2019(base_dir=args.root_dir, 
                         split='train', 
                         transform=T.Compose([
                             SagittalToAxial(), # Chuyển hướng lát cắt
                             RandomCrop(patch_size), # Augmentation: crop ngẫu nhiên
                             RandomRotFlip(), # Augmentation: xoay/nghiêng ngẫu nhiên
                             ToTensor()
                        ]))
    
    # Chia chỉ số thành labeled và unlabeled (phục vụ semi-supervised)
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, db_train.__len__()))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        
    model.train()
    ema_model.train()

    # === 4. Khởi tạo optimizer, loss function ===
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Loss consistency giữa student/teacher (khối consistency trong hình)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} Itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # Loss đặc trưng của DyCON: UnCLoss, FeCLoss (khối xanh lá trong hình)
    uncl_criterion = dycon_losses.UnCLoss()
    fecl_criterion = dycon_losses.FeCLoss(device=torch.device("cuda"), temperature=args.temp, gamma=args.gamma, use_focal=bool(args.use_focal), rampup_epochs=1500)
    
    # NEW: Initialize advanced topology loss for conference submission
    advanced_topo_loss = None
    if bool(args.use_advanced_topo):
        try:
            config = {
                'lambda_info': args.lambda_info,
                'lambda_wasserstein': args.lambda_wasserstein,
                'lambda_bayesian': args.lambda_bayesian,
                'use_adaptive': bool(args.use_adaptive_weight)
            }
            advanced_topo_loss = create_advanced_topology_loss(device='cuda', config=config)
            logging.info(f"Advanced topology loss initialized with config: {config}")
        except Exception as e:
            logging.warning(f"Failed to initialize advanced topology loss: {str(e)}")
            advanced_topo_loss = None
    
    # === Khởi tạo dict lưu các metric để vẽ biểu đồ ===
    metrics_history = {
        'loss': [],
        'f_loss': [],
        'u_loss': [],
        'loss_ce': [],
        'loss_dice': [],
        'consistency_loss': [],
        'consistency_weight': [],
        'topo_loss': [],
        'topo_weight': [],
        'train_Dice': [],
        'train_HD95': [],
        'val_Dice': [],
        'val_Best_dice': [],
        'train_Accuracy': [],
        # NEW: Advanced topology metrics
        'advanced_topo_total': [],
        'advanced_topo_info': [],
        'advanced_topo_wasserstein': [],
        'advanced_topo_bayesian': [],
        'advanced_topo_adaptive_factor': [],
        'stu_uncertainty': [],
        'tea_uncertainty': [],
    }

    for epoch_num in iterator:
        # === 5. Adaptive beta cho UnCLoss (theo epoch) ===
        if args.s_beta is not None:
            beta = args.s_beta
        else:
            beta = dycon_losses.adaptive_beta(epoch=epoch_num, total_epochs=max_epoch, max_beta=args.beta_max, min_beta=args.beta_min)

        for i_batch, sampled_batch in enumerate(trainloader):
            # === 6. Lấy batch dữ liệu, augmentation noise cho teacher ===
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2) # Augmentation cho teacher
            ema_inputs = volume_batch + noise

            # Log input shapes to encoder
            with open(snapshot_path + "/log.txt", "a") as f:
                f.write(f"[Input to student encoder] volume_batch.shape: {volume_batch.shape}\n")
                f.write(f"[Input to teacher encoder] ema_inputs.shape: {ema_inputs.shape}\n")

            # === 7. Forward student/teacher model (khối giữa hình) ===
            _, stud_logits, stud_features = model(volume_batch) # Student: logits, features
            with torch.no_grad():
                _, ema_logits, ema_features = ema_model(ema_inputs) # Teacher: logits, features

            # Log output shapes from decoder
            with open(snapshot_path + "/log.txt", "a") as f:
                f.write(f"[Output from student decoder] stud_logits.shape: {stud_logits.shape}\n")
                f.write(f"[Output from teacher decoder] ema_logits.shape: {ema_logits.shape}\n")
                f.write(f"[Student features] stud_features.shape: {stud_features.shape}\n")
                f.write(f"[Teacher features] ema_features.shape: {ema_features.shape}\n")
           
            stud_probs = F.softmax(stud_logits, dim=1)
            ema_probs = F.softmax(ema_logits, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            # === 8. Tính các loss ===
            # --- Supervised loss (khối đỏ, L_Dice + L_CE) ---
            # Log input shapes to Dice/CE loss
            with open(snapshot_path + "/log.txt", "a") as f:
                f.write(f"[Input to Dice/CE] stud_logits[:labeled_bs].shape: {stud_logits[:labeled_bs].shape}, label_batch[:labeled_bs].shape: {label_batch[:labeled_bs].shape}\n")
                f.write(f"[Input to Dice] stud_probs[:labeled_bs, 1, :, :, :].shape: {stud_probs[:labeled_bs, 1, :, :, :].shape}, label_batch[:labeled_bs] == 1 shape: {(label_batch[:labeled_bs] == 1).shape}\n")
            loss_seg = F.cross_entropy(stud_logits[:labeled_bs], label_batch[:labeled_bs]) # CrossEntropy cho labeled
            loss_seg_dice = losses.dice_loss(stud_probs[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1) # Dice cho labeled
            
            # --- Chuẩn bị embedding cho contrastive loss (FeCL, UnCL) ---
            # stud_features có shape (B, C, D, H, W) với:
            #   B: batch size
            #   C: số lượng kênh đặc trưng (feature channels)
            #   D, H, W: kích thước không gian (depth, height, width)
            # Để tính contrastive loss, ta cần biểu diễn mỗi patch (vị trí không gian) thành 1 vector embedding.
            # Bước 1: .view(B, C, -1) => chuyển (B, C, D, H, W) thành (B, C, N), với N = D*H*W (tổng số patch)
            # Bước 2: .transpose(1, 2) => chuyển thành (B, N, C), mỗi patch là 1 vector C chiều
            # Bước 3: F.normalize(..., dim=-1) => chuẩn hóa vector embedding theo từng patch (chuẩn hóa theo chiều C)
            B, C, _, _, _ = stud_features.shape  # B: batch size, C: số kênh đặc trưng
            stud_embedding = stud_features.view(B, C, -1)         # (B, C, N)
            stud_embedding = torch.transpose(stud_embedding, 1, 2) # (B, N, C) 8, 1728, 256
            stud_embedding = F.normalize(stud_embedding, dim=-1)   # (B, N, C), mỗi patch là 1 vector đã chuẩn hóa 8, 1728, 256

            ema_embedding = ema_features.view(B, C, -1)
            ema_embedding = torch.transpose(ema_embedding, 1, 2)
            ema_embedding = F.normalize(ema_embedding, dim=-1)

            # --- Mask contrastive: tạo mask cho patch embedding (phục vụ FeCL) ---
            mask_con = F.avg_pool3d(label_batch.float(), kernel_size=args.feature_scaler*4, stride=args.feature_scaler*4)
            mask_con = (mask_con > 0.5).float()
            mask_con = mask_con.reshape(B, -1)
            mask_con = mask_con.unsqueeze(1) 

            # Log input shapes to LFeCL
            with open(snapshot_path + "/log.txt", "a") as f:
                f.write(f"[Input to LFeCL] stud_embedding.shape: {stud_embedding.shape}, mask_con.shape: {mask_con.shape}\n")
                if args.use_teacher_loss:
                    f.write(f"[Input to LFeCL] teacher_feat.shape: {ema_embedding.shape}\n")

            # === Wrapper class cho GradCAM với model có tuple output ===
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    # UNet3D returns (output_sdf, out_seg, features)
                    # GradCAM cần chỉ 1 output tensor, ta chọn out_seg (logits)
                    output = self.model(x)
                    if isinstance(output, tuple):
                        return output[1]  # out_seg (logits)
                    return output

            # === (NEW) Grad-CAM + wandb logging ===
            if iter_num % 200 == 0:
                try:
                    # Chọn sample đầu tiên trong batch
                    sample_input = volume_batch[0:1]  # (1, C, D, H, W)
                    sample_label = label_batch[0:1]   # (1, D, H, W)
                    
                    # Lấy lát giữa của volume
                    D = sample_input.shape[2]
                    mid_slice = D // 2
                    mask_slice = sample_label[0, :, :, mid_slice].cpu().numpy().astype(np.uint8)  # (H, W)
                    
                    # Chuẩn bị model wrapper cho GradCAM
                    base_model = model.module if hasattr(model, 'module') else model
                    model_for_cam = ModelWrapper(base_model)
                    
                    # Kiểm tra model output để xác định số classes
                    with torch.no_grad():
                        test_out = model_for_cam(sample_input)
                        num_channels = test_out.shape[1]
                        logging.info(f"Model output shape: {test_out.shape}, num_channels: {num_channels}")
                    
                    # Chỉ thực hiện GradCAM nếu có ít nhất 2 classes
                    if num_channels >= 2:
                        target_class = 1  # Class foreground
                        
                        # Grad-CAM setup với wrapper model
                        target_layer = base_model.center  # Lấy layer từ base model
                        cam = GradCAM(model=model_for_cam, target_layers=[target_layer])
                        
                        # Sử dụng 2D mask thay vì 3D để tương thích với GradCAM
                        # GradCAM sẽ tính toán cho toàn bộ 3D volume nhưng chỉ focus vào mask_slice
                        mask_2d = mask_slice  # (H, W)
                        targets = [SemanticSegmentationTarget(target_class, mask_2d)]
                        
                        # Tính GradCAM cho 3D input
                        grayscale_cam = cam(input_tensor=sample_input, targets=targets)[0]  # (D, H, W)
                        cam_slice = grayscale_cam[mid_slice]  # (H, W)
                        
                        # Chuẩn bị ảnh để visualization
                        img_np = sample_input[0, 0, mid_slice].detach().cpu().numpy()
                        img_slice_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                        img_rgb = np.stack([img_slice_norm]*3, axis=-1)
                        cam_overlay = show_cam_on_image(img_rgb, cam_slice, use_rgb=True)
                        
                        # Chuẩn bị probability maps để log
                        # Student probability map (class 1 - foreground)
                        stud_prob_slice = stud_probs[0, 1, :, :, mid_slice].detach().cpu().numpy()  # (H, W)
                        
                        # Teacher probability map (class 1 - foreground)  
                        tea_prob_slice = ema_probs[0, 1, :, :, mid_slice].detach().cpu().numpy()  # (H, W)
                        
                        # Difference map (student - teacher)
                        diff_map = stud_prob_slice - tea_prob_slice
                        
                        # Log lên wandb
                        wandb.log({
                            'GradCAM': wandb.Image(cam_overlay, caption=f"GradCAM iter {iter_num}"),
                            'MRI': wandb.Image(img_slice_norm, caption=f"MRI iter {iter_num}"),
                            'GT Mask': wandb.Image(mask_slice, caption=f"GT Mask iter {iter_num}"),
                            'Student Prob': wandb.Image(stud_prob_slice, caption=f"Student Probability Map iter {iter_num}"),
                            'Teacher Prob': wandb.Image(tea_prob_slice, caption=f"Teacher Probability Map iter {iter_num}"),
                            'Prob Difference': wandb.Image(diff_map, caption=f"Student - Teacher Difference iter {iter_num}")
                        })
                    else:
                        # Nếu chỉ có 1 class, chỉ log ảnh gốc và mask + probability maps
                        img_np = sample_input[0, 0, mid_slice].detach().cpu().numpy()
                        img_slice_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                        
                        # Vẫn log probability maps ngay cả khi không có GradCAM
                        if num_channels == 1:
                            # Nếu chỉ có 1 channel, coi như probability của class foreground
                            stud_prob_slice = torch.sigmoid(stud_logits[0, 0, :, :, mid_slice]).detach().cpu().numpy()
                            tea_prob_slice = torch.sigmoid(ema_logits[0, 0, :, :, mid_slice]).detach().cpu().numpy()
                        else:
                            # Nếu có nhiều channel nhưng < 2 (trường hợp lạ)
                            stud_prob_slice = stud_probs[0, -1, :, :, mid_slice].detach().cpu().numpy()
                            tea_prob_slice = ema_probs[0, -1, :, :, mid_slice].detach().cpu().numpy()
                        
                        diff_map = stud_prob_slice - tea_prob_slice
                        
                        wandb.log({
                            'MRI': wandb.Image(img_slice_norm, caption=f"MRI iter {iter_num}"),
                            'GT Mask': wandb.Image(mask_slice, caption=f"GT Mask iter {iter_num}"),
                            'Student Prob': wandb.Image(stud_prob_slice, caption=f"Student Probability Map iter {iter_num}"),
                            'Teacher Prob': wandb.Image(tea_prob_slice, caption=f"Teacher Probability Map iter {iter_num}"),
                            'Prob Difference': wandb.Image(diff_map, caption=f"Student - Teacher Difference iter {iter_num}")
                        })
                        logging.warning(f"Model only has {num_channels} output channels, skipping GradCAM")
                
                except Exception as e:
                    logging.warning(f"Error in GradCAM visualization at iteration {iter_num}: {str(e)}")
                    # Fallback: log ảnh gốc và probability maps nếu có thể
                    try:
                        sample_input = volume_batch[0:1]
                        sample_label = label_batch[0:1]
                        D = sample_input.shape[2]
                        mid_slice = D // 2
                        mask_slice = sample_label[0, :, :, mid_slice].cpu().numpy().astype(np.uint8)
                        
                        img_np = sample_input[0, 0, mid_slice].detach().cpu().numpy()
                        img_slice_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                        
                        # Cố gắng log probability maps
                        try:
                            stud_prob_slice = stud_probs[0, 1, :, :, mid_slice].detach().cpu().numpy()
                            tea_prob_slice = ema_probs[0, 1, :, :, mid_slice].detach().cpu().numpy()
                            diff_map = stud_prob_slice - tea_prob_slice
                            
                            wandb.log({
                                'MRI': wandb.Image(img_slice_norm, caption=f"MRI iter {iter_num}"),
                                'GT Mask': wandb.Image(mask_slice, caption=f"GT Mask iter {iter_num}"),
                                'Student Prob': wandb.Image(stud_prob_slice, caption=f"Student Probability Map iter {iter_num}"),
                                'Teacher Prob': wandb.Image(tea_prob_slice, caption=f"Teacher Probability Map iter {iter_num}"),
                                'Prob Difference': wandb.Image(diff_map, caption=f"Student - Teacher Difference iter {iter_num}")
                            })
                        except:
                            # Nếu không thể log probability maps, chỉ log ảnh cơ bản
                            wandb.log({
                                'MRI': wandb.Image(img_slice_norm, caption=f"MRI iter {iter_num}"),
                                'GT Mask': wandb.Image(mask_slice, caption=f"GT Mask iter {iter_num}")
                            })
                    except:
                        pass

            # --- FeCLoss: contrastive loss với hard negative mining (khối xanh lá, L_FeCL) ---
            teacher_feat = ema_embedding if args.use_teacher_loss else None
            f_loss = fecl_criterion(feat=stud_embedding,
                                    mask=mask_con, 
                                    teacher_feat=teacher_feat,
                                    gambling_uncertainty=None, # gambling_uncertainty
                                    epoch=epoch_num)
            # --- UnCLoss: contrastive loss cho unlabeled (khối xanh lá, L_UnCL) ---
            # Log input shapes to LUnCL
            with open(snapshot_path + "/log.txt", "a") as f:
                f.write(f"[Input to LUnCL] stud_logits.shape: {stud_logits.shape}, ema_logits.shape: {ema_logits.shape}\n")
            u_loss = uncl_criterion(stud_logits, ema_logits, beta)
            # --- Consistency loss giữa student/teacher (khối xanh dương nhạt) ---
            consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()
            
            # --- Topological loss: đảm bảo nhất quán về cấu trúc topo giữa student và teacher ---
            topo_weight = get_current_topo_weight(iter_num//150)
            topo_loss = torch.tensor(0.0).cuda()
            advanced_topo_losses = {}  # NEW: For advanced topology losses
            
            # Support standard, uncertainty-aware, and advanced topology loss
            use_any_topo = bool(args.use_topo_loss) or bool(args.use_uncertainty_topo) or bool(args.use_advanced_topo)
            
            # DEBUG: Log topology computation details
            if iter_num % 50 == 0:  # Log every 50 iterations
                logging.info(f"[TOPO DEBUG] Iter {iter_num}: use_topo_loss={args.use_topo_loss}, use_uncertainty_topo={args.use_uncertainty_topo}, use_advanced_topo={args.use_advanced_topo}, topo_weight={topo_weight:.6f}, labeled_bs={labeled_bs}")
            
            if use_any_topo and topo_weight > 0:
                try:
                    # Tính topological loss cho từng sample trong batch (chỉ labeled samples)
                    batch_topo_loss = 0.0
                    num_valid_samples = 0
                    
                    for b_idx in range(labeled_bs):
                        # Lấy probability map của student và teacher cho sample này
                        # Chỉ lấy class 1 (foreground) để phân tích topo
                        # Vì đây là bài toán segmentation với 2 class: 0 (background) và 1 (foreground)
                        # Foreground: là vùng quan tâm (ví dụ: khối u, cơ quan, tổn thương, ...), thường được gán label = 1.
                        # Background: là phần còn lại (không quan tâm), thường label = 0.
                        # Foreground mới là vùng chứa thông tin cấu trúc quan trọng (ví dụ: số lượng khối u, hình dạng, lỗ hổng, ...).
                        # Background thường là không gian trống, không có ý nghĩa về mặt topo (thường là 1 khối lớn, không có cấu trúc phức tạp).
                        # Nếu tính topo cho cả background, sẽ không phản ánh đúng chất lượng phân đoạn của vùng quan tâm. 
                        # Persistence diagram hoặc Betti number của foreground sẽ cho biết số thành phần liên thông, số lỗ, ... của vùng quan tâm.
                        # => Chúng ta chỉ lấy class foreground (label = 1) để phân tích topo vì đây là vùng quan tâm chính trong bài toán phân đoạn y tế. Các đặc trưng topo như số thành phần liên thông, số lỗ, ... của foreground phản ánh trực tiếp chất lượng phân đoạn. 
                        # => Ngược lại, background thường không mang ý nghĩa cấu trúc quan trọng, nên không cần tối ưu topo cho vùng này.
                        stu_prob_2d = stud_probs[b_idx, 1, :, :, stud_probs.shape[-1]//2]  # slice giữa
                        tea_prob_2d = ema_probs[b_idx, 1, :, :, ema_probs.shape[-1]//2]   # slice giữa
                        
                        # DEBUG: Log probability map stats
                        stu_variation = stu_prob_2d.max() - stu_prob_2d.min()
                        tea_variation = tea_prob_2d.max() - tea_prob_2d.min()
                        
                        if iter_num % 50 == 0 and b_idx == 0:  # Log first sample every 50 iterations
                            logging.info(f"[TOPO DEBUG] Sample {b_idx}: stu_prob range=[{stu_prob_2d.min():.4f}, {stu_prob_2d.max():.4f}], variation={stu_variation:.4f}")
                            logging.info(f"[TOPO DEBUG] Sample {b_idx}: tea_prob range=[{tea_prob_2d.min():.4f}, {tea_prob_2d.max():.4f}], variation={tea_variation:.4f}")
                        
                        # Chỉ tính topo loss nếu có đủ variation trong probability map
                        # => Nếu không có variation, sẽ không có cấu trúc topo để tối ưu, do probability map không có cấu trúc phức tạp (ví dụ: không có lỗ hổng, không có cấu trúc phức tạp, ...).
                        # Tránh tính topo loss cho các trường hợp vô nghĩa (toàn ảnh đen/trắng, hoặc dự đoán quá chắc chắn).
                        if stu_variation > 0.1 and tea_variation > 0.1:
                            # NEW: Choose between standard or uncertainty-aware topological loss
                            if bool(args.use_uncertainty_topo):
                                sample_topo_loss = getUncertaintyAwareTopoLoss(
                                    stu_tensor=stu_prob_2d,
                                    tea_tensor=tea_prob_2d,
                                    topo_size=args.topo_size,
                                    pd_threshold=args.pd_threshold,
                                    uncertainty_threshold=args.uncertainty_threshold,
                                    focus_mode=args.topo_focus_mode
                                )
                                if iter_num % 50 == 0 and b_idx == 0:
                                    logging.info(f"[TOPO DEBUG] UA Topo Loss (sample {b_idx}): {sample_topo_loss.item():.6f}")
                            else:
                                sample_topo_loss = getTopoLoss(
                                    stu_tensor=stu_prob_2d,
                                    tea_tensor=tea_prob_2d,
                                    topo_size=args.topo_size,
                                    pd_threshold=args.pd_threshold
                                )
                                if iter_num % 50 == 0 and b_idx == 0:
                                    logging.info(f"[TOPO DEBUG] Std Topo Loss (sample {b_idx}): {sample_topo_loss.item():.6f}")
                            
                            if not torch.isnan(sample_topo_loss) and not torch.isinf(sample_topo_loss):
                                batch_topo_loss += sample_topo_loss
                                num_valid_samples += 1
                        else:
                            if iter_num % 50 == 0 and b_idx == 0:
                                logging.info(f"[TOPO DEBUG] Sample {b_idx}: Skipped (insufficient variation)")
                            
                    
                    if num_valid_samples > 0:
                        topo_loss = batch_topo_loss / num_valid_samples
                        if iter_num % 50 == 0:
                            logging.info(f"[TOPO DEBUG] Final topo_loss: {topo_loss.item():.6f} (from {num_valid_samples} valid samples)")
                    else:
                        if iter_num % 50 == 0:
                            logging.info(f"[TOPO DEBUG] No valid samples for topo loss computation")
                        
                except Exception as e:
                    logging.warning(f"Error computing topological loss at iteration {iter_num}: {str(e)}")
                    topo_loss = torch.tensor(0.0).cuda()
                    
                # NEW: Advanced topology loss computation for conference submission
                if bool(args.use_advanced_topo) and advanced_topo_loss is not None:
                    try:
                        # Compute current training progress and dice score for adaptive weighting
                        epoch_progress = min(1.0, iter_num / max_iterations) if max_iterations > 0 else 0.0
                        
                        # Estimate current dice score from recent validation or use default
                        current_dice = best_performance if best_performance > 0 else 0.5
                        
                        # Process each sample in labeled batch
                        advanced_topo_batch_loss = 0.0
                        valid_advanced_samples = 0
                        
                        for b_idx in range(labeled_bs):
                            # Get probability maps for this sample (middle slice)
                            stu_prob_2d = stud_probs[b_idx, 1, :, :, stud_probs.shape[-1]//2]  # (H, W)
                            tea_prob_2d = ema_probs[b_idx, 1, :, :, ema_probs.shape[-1]//2]   # (H, W)
                            
                            # Skip if insufficient variation
                            stu_variation = stu_prob_2d.max() - stu_prob_2d.min()
                            tea_variation = tea_prob_2d.max() - tea_prob_2d.min()
                            
                            if stu_variation > 0.1 and tea_variation > 0.1:
                                # Compute advanced topology loss for this sample
                                sample_losses = advanced_topo_loss(
                                    stu_tensor=stu_prob_2d,
                                    tea_tensor=tea_prob_2d,
                                    epoch_progress=epoch_progress,
                                    current_dice=current_dice
                                )
                                
                                # Accumulate valid losses
                                if not torch.isnan(sample_losses['total_advanced_topology']) and not torch.isinf(sample_losses['total_advanced_topology']):
                                    advanced_topo_batch_loss += sample_losses['total_advanced_topology']
                                    valid_advanced_samples += 1
                                    
                                    # Store individual component losses (only from first sample for logging)
                                    if b_idx == 0 and iter_num % 50 == 0:
                                        for key, value in sample_losses.items():
                                            if key not in advanced_topo_losses:
                                                advanced_topo_losses[key] = value
                                            
                        # Average over valid samples
                        if valid_advanced_samples > 0:
                            advanced_topo_final = advanced_topo_batch_loss / valid_advanced_samples
                            # Add to total topology loss
                            topo_loss = topo_loss + advanced_topo_final
                            
                            if iter_num % 50 == 0:
                                logging.info(f"[ADVANCED TOPO] Final loss: {advanced_topo_final.item():.6f} (from {valid_advanced_samples} samples)")
                        else:
                            if iter_num % 50 == 0:
                                logging.info(f"[ADVANCED TOPO] No valid samples for advanced topology")
                                
                    except Exception as e:
                        logging.warning(f"Error computing advanced topological loss at iteration {iter_num}: {str(e)}")
            
            # Gather losses
            loss = args.l_weight * (loss_seg + loss_seg_dice) + consistency_weight * consistency_loss + args.u_weight * (f_loss + u_loss) + topo_weight * topo_loss
            # loss = args.l_weight * (loss_seg + loss_seg_dice) +  args.u_weight * (f_loss + u_loss) +  topo_loss

            # === 9. Backward, update model ===
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf found in loss at iteration {iter_num}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            # --- EMA update: cập nhật teacher model từ student model (khối EMA, hình giữa) ---
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
             
            # === 10. Logging các giá trị loss, metrics ===
            writer.add_scalar('info/loss', loss, iter_num)
            writer.add_scalar('info/f_loss', f_loss, iter_num)
            writer.add_scalar('info/u_loss', u_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_seg, iter_num)
            writer.add_scalar('info/loss_dice', loss_seg_dice, iter_num) 
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/topo_loss', topo_loss, iter_num)
            writer.add_scalar('info/topo_weight', topo_weight, iter_num)
            
            # NEW: Advanced topology tensorboard logging
            if advanced_topo_losses:
                writer.add_scalar('advanced_topo/total', advanced_topo_losses.get('total_advanced_topology', torch.tensor(0.0)).item(), iter_num)
                writer.add_scalar('advanced_topo/info_theory', advanced_topo_losses.get('info_topology', torch.tensor(0.0)).item(), iter_num)
                writer.add_scalar('advanced_topo/wasserstein', advanced_topo_losses.get('wasserstein_topology', torch.tensor(0.0)).item(), iter_num)
                writer.add_scalar('advanced_topo/bayesian', advanced_topo_losses.get('bayesian_topology', torch.tensor(0.0)).item(), iter_num)
                writer.add_scalar('advanced_topo/adaptive_factor', advanced_topo_losses.get('adaptive_factor', torch.tensor(1.0)).item(), iter_num)
                writer.add_scalar('uncertainty/student_mean', advanced_topo_losses.get('stu_uncertainty_mean', torch.tensor(0.0)).item(), iter_num)
                writer.add_scalar('uncertainty/teacher_mean', advanced_topo_losses.get('tea_uncertainty_mean', torch.tensor(0.0)).item(), iter_num)
            
            # === Lưu các giá trị vào metrics_history để vẽ biểu đồ sau này ===
            metrics_history['loss'].append(loss.item())
            metrics_history['f_loss'].append(f_loss.item())
            metrics_history['u_loss'].append(u_loss.item())
            metrics_history['loss_ce'].append(loss_seg.item())
            metrics_history['loss_dice'].append(loss_seg_dice.item())
            metrics_history['consistency_loss'].append(consistency_loss.item())
            metrics_history['consistency_weight'].append(consistency_weight)
            metrics_history['topo_loss'].append(topo_loss.item())
            metrics_history['topo_weight'].append(topo_weight)
            
            # NEW: Store advanced topology losses
            if advanced_topo_losses:
                metrics_history['advanced_topo_total'].append(advanced_topo_losses.get('total_advanced_topology', torch.tensor(0.0)).item())
                metrics_history['advanced_topo_info'].append(advanced_topo_losses.get('info_topology', torch.tensor(0.0)).item())
                metrics_history['advanced_topo_wasserstein'].append(advanced_topo_losses.get('wasserstein_topology', torch.tensor(0.0)).item())
                metrics_history['advanced_topo_bayesian'].append(advanced_topo_losses.get('bayesian_topology', torch.tensor(0.0)).item())
                metrics_history['advanced_topo_adaptive_factor'].append(advanced_topo_losses.get('adaptive_factor', torch.tensor(1.0)).item())
                metrics_history['stu_uncertainty'].append(advanced_topo_losses.get('stu_uncertainty_mean', torch.tensor(0.0)).item())
                metrics_history['tea_uncertainty'].append(advanced_topo_losses.get('tea_uncertainty_mean', torch.tensor(0.0)).item())
            else:
                # Fill with zeros if no advanced topology losses
                metrics_history['advanced_topo_total'].append(0.0)
                metrics_history['advanced_topo_info'].append(0.0)
                metrics_history['advanced_topo_wasserstein'].append(0.0)
                metrics_history['advanced_topo_bayesian'].append(0.0)
                metrics_history['advanced_topo_adaptive_factor'].append(0.0)
                metrics_history['stu_uncertainty'].append(0.0)
                metrics_history['tea_uncertainty'].append(0.0)

            del noise, stud_embedding, ema_logits, ema_features, ema_probs, mask_con

            # === 11. Đánh giá nhanh Dice, HD95, Accuracy trên batch hiện tại ===
            with torch.no_grad():
                outputs_bin = (stud_probs[:, 1, :, :, :] > 0.5).float()
                dice_score = metrics.compute_dice(outputs_bin, label_batch)
                H, W, D = stud_logits.shape[-3:]
                max_dist = np.linalg.norm([H, W, D])
                hausdorff_score = metrics.compute_hd95(outputs_bin, label_batch, max_dist)
                # Accuracy: số pixel đúng / tổng số pixel
                acc = (outputs_bin == (label_batch == 1)).float().mean().item()

            writer.add_scalar('train/Dice', dice_score.mean().item(), iter_num)
            writer.add_scalar('train/HD95', np.mean(hausdorff_score).item(), iter_num)
            writer.add_scalar('train/Accuracy', acc, iter_num)

            metrics_history['train_Dice'].append(dice_score.mean().item())
            metrics_history['train_HD95'].append(np.mean(hausdorff_score).item())
            metrics_history['train_Accuracy'].append(acc)
            
            # Enhanced topology loss logging với uncertainty info
            if use_any_topo and topo_weight > 0:
                if bool(args.use_advanced_topo) and advanced_topo_losses:
                    topo_type = "Adv-Topo"
                    info_loss = advanced_topo_losses.get('info_topology', torch.tensor(0.0)).item()
                    wass_loss = advanced_topo_losses.get('wasserstein_topology', torch.tensor(0.0)).item()
                    bay_loss = advanced_topo_losses.get('bayesian_topology', torch.tensor(0.0)).item()
                    adaptive = advanced_topo_losses.get('adaptive_factor', torch.tensor(1.0)).item()
                    topo_loss_str = f", {topo_type}: {topo_loss.item():.6f} (I:{info_loss:.4f},W:{wass_loss:.4f},B:{bay_loss:.4f},A:{adaptive:.3f})"
                elif bool(args.use_uncertainty_topo):
                    topo_type = f"UA-Topo({args.topo_focus_mode})"
                    topo_loss_str = f", {topo_type}: {topo_loss.item():.6f} (w={topo_weight:.6f})"
                else:
                    topo_type = "Std-Topo"
                    topo_loss_str = f", {topo_type}: {topo_loss.item():.6f} (w={topo_weight:.6f})"
            else:
                topo_loss_str = ""
            
            logging.info(
                'Iteration %d : Loss : %06f, Loss_CE: %06f, Loss_Dice: %06f, UnCLoss: %06f, FeCLoss: %06f%s, mean_dice: %06f, mean_hd95: %06f, acc: %06f' %
                (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(), u_loss.item(), f_loss.item(), topo_loss_str, dice_score.mean().item(), np.mean(hausdorff_score).item(), acc))

            # === Log metrics lên wandb ===
            wandb_metrics = {
                'loss': loss.item(),
                'f_loss': f_loss.item(),
                'u_loss': u_loss.item(),
                'loss_ce': loss_seg.item(),
                'loss_dice': loss_seg_dice.item(),
                'consistency_loss': consistency_loss.item(),
                'consistency_weight': consistency_weight,
                'topo_loss': topo_loss.item(),
                'topo_weight': topo_weight,
                'train_Dice': dice_score.mean().item(),
                'train_HD95': np.mean(hausdorff_score).item(),
                'train_Accuracy': acc,
                'iter': iter_num
            }
            
            # Enhanced wandb logging cho uncertainty-aware topology
            if use_any_topo and topo_weight > 0:
                if bool(args.use_advanced_topo) and advanced_topo_losses:
                    wandb_metrics.update({
                        'advanced_topo_total': advanced_topo_losses.get('total_advanced_topology', torch.tensor(0.0)).item(),
                        'advanced_topo_info': advanced_topo_losses.get('info_topology', torch.tensor(0.0)).item(),
                        'advanced_topo_wasserstein': advanced_topo_losses.get('wasserstein_topology', torch.tensor(0.0)).item(),
                        'advanced_topo_bayesian': advanced_topo_losses.get('bayesian_topology', torch.tensor(0.0)).item(),
                        'advanced_topo_adaptive_factor': advanced_topo_losses.get('adaptive_factor', torch.tensor(1.0)).item(),
                        'stu_uncertainty_mean': advanced_topo_losses.get('stu_uncertainty_mean', torch.tensor(0.0)).item(),
                        'tea_uncertainty_mean': advanced_topo_losses.get('tea_uncertainty_mean', torch.tensor(0.0)).item(),
                        'topo_type': 'advanced_conference',
                        'lambda_info': args.lambda_info,
                        'lambda_wasserstein': args.lambda_wasserstein,
                        'lambda_bayesian': args.lambda_bayesian
                    })
                elif bool(args.use_uncertainty_topo):
                    wandb_metrics.update({
                        'UA_topo_loss': topo_loss.item(),
                        'UA_topo_focus_mode': args.topo_focus_mode,
                        'UA_uncertainty_threshold': args.uncertainty_threshold,
                        'topo_type': 'uncertainty_aware'
                    })
                else:
                    wandb_metrics.update({
                        'std_topo_loss': topo_loss.item(),
                        'topo_type': 'standard'
                    })
                    
                # Additional topology-specific metrics
                wandb_metrics.update({
                    'topo_size': args.topo_size,
                    'pd_threshold': args.pd_threshold,
                    'weighted_topo_loss': topo_loss.item() * topo_weight
                })
            
            wandb.log(wandb_metrics)

            # === 12. Đánh giá toàn bộ validation set định kỳ, lưu model tốt nhất ===
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_3d_patch.var_all_case_BraTS19(model, args.root_dir, num_classes=args.num_classes, patch_size=patch_size, stride_xy=64, stride_z=64)
                if avg_metric > best_performance:
                    best_performance = round(avg_metric, 4)

                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format( iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/Dice', avg_metric, iter_num)
                writer.add_scalar('info/Best_dice', best_performance, iter_num)
                metrics_history['val_Dice'].append(avg_metric)
                metrics_history['val_Best_dice'].append(best_performance)
                logging.info('Iteration %d : Dice: %03f Best_dice: %03f' % (iter_num, avg_metric, best_performance))
                model.train()

            # === 13. Lưu checkpoint định kỳ ===
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    print("Training Finished!")

    # === Vẽ và lưu biểu đồ các metric sau khi train xong ===
    plot_and_save_metrics(metrics_history, os.path.join(snapshot_path, "metrics_plots"))

    # === Kết thúc wandb run ===
    wandb.finish()
