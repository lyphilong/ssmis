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
from utils import ramps, metrics, losses, dycon_losses, test_3d_patch
from utils.topological_loss_3d import TopologicalLoss3D # Sử dụng loss 3D mới
from dataloaders.brats19 import BraTS2019, SagittalToAxial, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler 

# Argument parsing
parser = argparse.ArgumentParser(description="Training DyCON with 3D Topological Loss on BraTS2019")

# --- Các tham số cơ bản ---
parser.add_argument('--root_dir', type=str, default="../data/BraTS2019", help='Path to BraTS-2019 dataset')
parser.add_argument('--patch_size', type=str, default=[112, 112, 80], help='Input image patch size')
parser.add_argument('--exp', type=str, default='BraTS2019_Topo3D', help='Experiment name')
parser.add_argument('--gpu_id', type=str, default=1, help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training (0 or 1)')

# --- Tham số Model ---
parser.add_argument('--model', type=str, choices=['unet_3D', 'vnet'], default='unet_3D', help='Model architecture')
parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--feature_scaler', type=int, default=2, help='Feature scaling factor for contrastive loss')

# --- Tham số Training ---
parser.add_argument('--max_iterations', type=int, default=30000, help='Maximum number of training iterations')
parser.add_argument('--batch_size', type=int, default=4, help='Total batch size per GPU')
parser.add_argument('--labeled_bs', type=int, default=2, help='Labeled batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')

# --- Tham số Semi-Supervised & DyCON ---
parser.add_argument('--labelnum', type=int, default=16, help='Number of labeled samples')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher model')
parser.add_argument('--consistency', type=float, default=0.1, help='Consistency weight')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Ramp-up duration for consistency weight')
parser.add_argument('--gamma', type=float, default=2.0, help='Focusing parameter for FeCL (γ)')
parser.add_argument('--temp', type=float, default=0.6, help='Temperature for contrastive softmax scaling')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight for supervised loss')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight for unsupervised loss')

# --- Tham số Topological Loss 3D ---
parser.add_argument('--use_topo_loss', type=int, default=1, help='Use 3D topological loss (1 for True, 0 for False)')
parser.add_argument('--topo_weight', type=float, default=0.1, help='Weight for topological loss')
parser.add_argument('--pd_threshold', type=float, default=0.1, help='Persistence diagram threshold for topo loss')
parser.add_argument('--topo_rampup', type=float, default=200.0, help='Ramp-up duration for topological loss weight')

args = parser.parse_args()

def train(args):
    # ---- 1. Thiết lập môi trường ----
    snapshot_path = f"../models/{args.exp}/{args.model.upper()}_{args.labelnum}labels_topo{args.topo_weight}"
    os.makedirs(snapshot_path, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # ---- 2. Khởi tạo Model, Loss, Optimizer ----
    def create_model(ema=False):
        net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=args.num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Loss functions
    ce_loss = F.cross_entropy
    dice_loss = losses.dice_loss
    consistency_criterion = losses.softmax_mse_loss
    
    topo_criterion_3d = TopologicalLoss3D(
        homology_dimensions=(0, 1), 
        pd_threshold=args.pd_threshold
    ).cuda()

    # ---- 3. Chuẩn bị Dataloader ----
    db_train = BraTS2019(base_dir=args.root_dir, 
                         split='train', 
                         transform=T.Compose([
                             SagittalToAxial(),
                             RandomCrop(args.patch_size),
                             RandomRotFlip(),
                             ToTensor()
                        ]))
    
    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    # ---- 4. Khởi tạo WandB, Logging ----
    wandb.init(project="DyCON-BraTS19-Topo3D", name=args.exp, config=vars(args))
    logging.basicConfig(filename=f"{snapshot_path}/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ---- 5. Vòng lặp Training ----
    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    best_performance = 0.0

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            labeled_bs = args.labeled_bs

            # ---- Forward pass ----
            outputs, _, _ = model(volume_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            
            with torch.no_grad():
                ema_outputs, _, _ = ema_model(volume_batch)
                ema_outputs_soft = F.softmax(ema_outputs, dim=1)

            # ---- Tính các thành phần Loss ----
            # Supervised Loss (trên batch có nhãn)
            loss_sup_ce = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
            loss_sup_dice = dice_loss(outputs_soft[:labeled_bs, 1, ...], (label_batch[:labeled_bs] == 1))
            loss_sup = 0.5 * (loss_sup_ce + loss_sup_dice)
            
            # Unsupervised Consistency Loss (trên batch không nhãn)
            consistency_weight = args.consistency * ramps.sigmoid_rampup(iter_num, args.consistency_rampup)
            consistency_loss = consistency_criterion(outputs_soft[labeled_bs:], ema_outputs_soft[labeled_bs:])

            # Topological Loss (trên batch có nhãn)
            topo_loss = torch.tensor(0.0).cuda()
            topo_weight = args.topo_weight * ramps.sigmoid_rampup(iter_num, args.topo_rampup)
            if args.use_topo_loss and topo_weight > 0:
                batch_topo_loss = 0.0
                for i in range(labeled_bs):
                    pred_map = outputs_soft[i, 1]
                    true_mask = (label_batch[i] == 1)
                    try:
                        sample_loss = topo_criterion_3d(pred_map, true_mask.float())
                        if not torch.isnan(sample_loss): batch_topo_loss += sample_loss
                    except Exception as e:
                        logging.warning(f"TopoLoss Error at iter {iter_num}: {e}")
                if labeled_bs > 0:
                    topo_loss = batch_topo_loss / labeled_bs
            
            # Loss tổng
            loss = loss_sup + consistency_weight * consistency_loss + topo_weight * topo_loss

            # ---- Backward và tối ưu ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Cập nhật EMA model ----
            alpha = min(1 - 1 / (iter_num + 1), args.ema_decay)
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
            
            # ---- Logging ----
            iter_num += 1
            wandb.log({
                'loss/total_loss': loss.item(),
                'loss/sup_loss': loss_sup.item(),
                'loss/consistency_loss': consistency_loss.item(),
                'loss/topo_loss': topo_loss.item(),
                'param/consistency_weight': consistency_weight,
                'param/topo_weight': topo_weight,
                'param/lr': optimizer.param_groups[0]['lr']
            })

            if iter_num % 200 == 0:
                # --- Đánh giá trên validation set ---
                model.eval()
                avg_metric = test_3d_patch.var_all_case_BraTS19(model, args.root_dir, num_classes=args.num_classes, patch_size=args.patch_size, stride_xy=64, stride_z=64)
                wandb.log({'val/dice': avg_metric, 'iter': iter_num})
                logging.info(f"Validation Dice: {avg_metric}")

                if avg_metric > best_performance:
                    best_performance = avg_metric
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info(f"Saved best model at iteration {iter_num}")
                
                model.train()
                
            if iter_num >= args.max_iterations:
                break
        
        if iter_num >= args.max_iterations:
            iterator.close()
            break
            
    # ---- Kết thúc Training ----
    wandb.finish()
    logging.info("Training Finished!")

if __name__ == "__main__":
    # --- Thiết lập args và chạy train ---
    args.patch_size = tuple(int(i) for i in args.patch_size)
    train(args) 