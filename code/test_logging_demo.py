#!/usr/bin/env python3
"""
Demo script to showcase enhanced logging for Uncertainty-Aware Topological Loss
"""

import torch
import numpy as np
from utils.topo_losses import getTopoLoss, getUncertaintyAwareTopoLoss

def demo_enhanced_logging():
    """
    Demo để show cách log được enhanced cho uncertainty-aware topo loss
    """
    print("=== ENHANCED LOGGING DEMO FOR UNCERTAINTY-AWARE TOPO LOSS ===")
    
    # Tạo synthetic data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Student và teacher probability maps
    H, W = 64, 64
    
    # Student map có uncertainty cao ở boundary
    student_map = torch.zeros(H, W, device=device)
    center_x, center_y = H//2, W//2
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if 10 < dist < 20:  # Ring structure với noise
                student_map[i, j] = 0.6 + 0.3 * torch.randn(1).to(device)  # Uncertain
            elif dist <= 10:
                student_map[i, j] = 0.1 + 0.05 * torch.randn(1).to(device)  # Confident
            else:
                student_map[i, j] = 0.05 + 0.02 * torch.randn(1).to(device)  # Confident
    
    student_map = torch.clamp(student_map, 0, 1)
    
    # Teacher map (more confident)
    teacher_map = torch.zeros(H, W, device=device)
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if 12 < dist < 18:
                teacher_map[i, j] = 0.95
            elif dist <= 12:
                teacher_map[i, j] = 0.05
            else:
                teacher_map[i, j] = 0.02
    
    # Test parameters
    topo_size = 16
    pd_threshold = 0.3
    uncertainty_threshold = 0.7
    topo_weight = 0.01
    
    print(f"\nTest parameters:")
    print(f"  topo_size: {topo_size}")
    print(f"  pd_threshold: {pd_threshold}")
    print(f"  uncertainty_threshold: {uncertainty_threshold}")
    print(f"  topo_weight: {topo_weight}")
    
    # ================================================================
    # TEST 1: Standard Topology Loss
    # ================================================================
    print(f"\n--- CASE 1: STANDARD TOPOLOGY LOSS ---")
    
    std_topo_loss = getTopoLoss(
        stu_tensor=student_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold
    )
    
    # Simulate iteration 100
    iter_num = 100
    loss = torch.tensor(5.234567).to(device)
    loss_ce = torch.tensor(0.876543).to(device)
    loss_dice = torch.tensor(0.987654).to(device)
    u_loss = torch.tensor(3.456789).to(device)
    f_loss = torch.tensor(2.345678).to(device)
    mean_dice = 0.543210
    mean_hd95 = 89.123456
    acc = 0.678901
    
    # Standard topology logging
    use_any_topo = True
    use_uncertainty_topo = False
    topo_loss = std_topo_loss
    
    if use_any_topo and topo_weight > 0:
        if use_uncertainty_topo:
            topo_type = f"UA-Topo(confident)"
        else:
            topo_type = "Std-Topo"
        topo_loss_str = f", {topo_type}: {topo_loss.item():.6f} (w={topo_weight:.3f})"
    else:
        topo_loss_str = ""
    
    log_message = (
        f'Iteration {iter_num} : Loss : {loss.item():.6f}, Loss_CE: {loss_ce.item():.6f}, '
        f'Loss_Dice: {loss_dice.item():.6f}, UnCLoss: {u_loss.item():.6f}, '
        f'FeCLoss: {f_loss.item():.6f}{topo_loss_str}, mean_dice: {mean_dice:.6f}, '
        f'mean_hd95: {mean_hd95:.6f}, acc: {acc:.6f}'
    )
    
    print("Console Output:")
    print(log_message)
    
    # Wandb metrics
    wandb_metrics = {
        'loss': loss.item(),
        'f_loss': f_loss.item(),
        'u_loss': u_loss.item(),
        'loss_ce': loss_ce.item(),
        'loss_dice': loss_dice.item(),
        'topo_loss': topo_loss.item(),
        'topo_weight': topo_weight,
        'train_Dice': mean_dice,
        'train_HD95': mean_hd95,
        'train_Accuracy': acc,
        'iter': iter_num
    }
    
    if use_any_topo and topo_weight > 0:
        wandb_metrics.update({
            'std_topo_loss': topo_loss.item(),
            'topo_type': 'standard',
            'topo_size': topo_size,
            'pd_threshold': pd_threshold,
            'weighted_topo_loss': topo_loss.item() * topo_weight
        })
    
    print("Wandb Metrics:")
    for key, value in wandb_metrics.items():
        print(f"  {key}: {value}")
    
    # ================================================================
    # TEST 2: Uncertainty-Aware Topology Loss (Confident Focus)
    # ================================================================
    print(f"\n--- CASE 2: UNCERTAINTY-AWARE TOPOLOGY LOSS (CONFIDENT) ---")
    
    ua_topo_loss_conf = getUncertaintyAwareTopoLoss(
        stu_tensor=student_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold,
        uncertainty_threshold=uncertainty_threshold,
        focus_mode="confident"
    )
    
    # UA topology logging
    use_uncertainty_topo = True
    topo_loss = ua_topo_loss_conf
    topo_focus_mode = "confident"
    
    if use_any_topo and topo_weight > 0:
        if use_uncertainty_topo:
            topo_type = f"UA-Topo({topo_focus_mode})"
        else:
            topo_type = "Std-Topo"
        topo_loss_str = f", {topo_type}: {topo_loss.item():.6f} (w={topo_weight:.3f})"
    else:
        topo_loss_str = ""
    
    log_message = (
        f'Iteration {iter_num} : Loss : {loss.item():.6f}, Loss_CE: {loss_ce.item():.6f}, '
        f'Loss_Dice: {loss_dice.item():.6f}, UnCLoss: {u_loss.item():.6f}, '
        f'FeCLoss: {f_loss.item():.6f}{topo_loss_str}, mean_dice: {mean_dice:.6f}, '
        f'mean_hd95: {mean_hd95:.6f}, acc: {acc:.6f}'
    )
    
    print("Console Output:")
    print(log_message)
    
    # Enhanced wandb metrics for UA topo
    wandb_metrics = {
        'loss': loss.item(),
        'f_loss': f_loss.item(),
        'u_loss': u_loss.item(),
        'loss_ce': loss_ce.item(),
        'loss_dice': loss_dice.item(),
        'topo_loss': topo_loss.item(),
        'topo_weight': topo_weight,
        'train_Dice': mean_dice,
        'train_HD95': mean_hd95,
        'train_Accuracy': acc,
        'iter': iter_num
    }
    
    if use_any_topo and topo_weight > 0:
        wandb_metrics.update({
            'UA_topo_loss': topo_loss.item(),
            'UA_topo_focus_mode': topo_focus_mode,
            'UA_uncertainty_threshold': uncertainty_threshold,
            'topo_type': 'uncertainty_aware',
            'topo_size': topo_size,
            'pd_threshold': pd_threshold,
            'weighted_topo_loss': topo_loss.item() * topo_weight
        })
    
    print("Enhanced Wandb Metrics:")
    for key, value in wandb_metrics.items():
        print(f"  {key}: {value}")
    
    # ================================================================
    # TEST 3: Uncertainty-Aware Topology Loss (Uncertain Focus)
    # ================================================================
    print(f"\n--- CASE 3: UNCERTAINTY-AWARE TOPOLOGY LOSS (UNCERTAIN) ---")
    
    ua_topo_loss_uncer = getUncertaintyAwareTopoLoss(
        stu_tensor=student_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold,
        uncertainty_threshold=uncertainty_threshold,
        focus_mode="uncertain"
    )
    
    # UA topology logging (uncertain focus)
    topo_loss = ua_topo_loss_uncer
    topo_focus_mode = "uncertain"
    
    if use_any_topo and topo_weight > 0:
        topo_type = f"UA-Topo({topo_focus_mode})"
        topo_loss_str = f", {topo_type}: {topo_loss.item():.6f} (w={topo_weight:.3f})"
    
    log_message = (
        f'Iteration {iter_num} : Loss : {loss.item():.6f}, Loss_CE: {loss_ce.item():.6f}, '
        f'Loss_Dice: {loss_dice.item():.6f}, UnCLoss: {u_loss.item():.6f}, '
        f'FeCLoss: {f_loss.item():.6f}{topo_loss_str}, mean_dice: {mean_dice:.6f}, '
        f'mean_hd95: {mean_hd95:.6f}, acc: {acc:.6f}'
    )
    
    print("Console Output:")
    print(log_message)
    
    # ================================================================
    # COMPARISON SUMMARY
    # ================================================================
    print(f"\n=== TOPOLOGY LOSS COMPARISON ===")
    print(f"Standard Topo Loss:           {std_topo_loss.item():.6f}")
    print(f"UA Topo Loss (confident):     {ua_topo_loss_conf.item():.6f}")
    print(f"UA Topo Loss (uncertain):     {ua_topo_loss_uncer.item():.6f}")
    
    print(f"\n=== LOGGING ENHANCEMENTS ===")
    print("✅ Enhanced console output với topo type và weight")
    print("✅ Detailed wandb metrics cho uncertainty-aware features")
    print("✅ Support cho cả standard và uncertainty-aware topology")
    print("✅ Clear distinction giữa focus modes")
    print("✅ Additional metrics: weighted_topo_loss, topo_type, etc.")

if __name__ == "__main__":
    demo_enhanced_logging() 