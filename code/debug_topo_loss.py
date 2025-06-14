#!/usr/bin/env python3
"""
Debug script to understand why topology loss = 0
"""

import torch
import numpy as np
from utils.topo_losses import getTopoLoss, getUncertaintyAwareTopoLoss
from utils import ramps

def debug_sigmoid_rampup():
    """
    Test sigmoid rampup function
    """
    print("=== DEBUG SIGMOID RAMPUP ===")
    
    # Test parameters
    topo_weight = 0.03
    topo_rampup = 500.0
    
    for epoch in [0, 1, 2, 5, 10, 50, 100, 200, 500, 1000]:
        rampup_factor = ramps.sigmoid_rampup(epoch, topo_rampup)
        final_weight = topo_weight * rampup_factor
        print(f"Epoch {epoch:4d}: rampup_factor={rampup_factor:.6f}, final_weight={final_weight:.6f}")

def debug_probability_maps():
    """
    Test với different probability map scenarios
    """
    print("\n=== DEBUG PROBABILITY MAPS ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 96, 96
    
    # Scenario 1: Very early training (low confidence)
    print("\n--- Scenario 1: Early training (low confidence) ---")
    early_student = torch.ones(H, W, device=device) * 0.1  # Mostly background prediction
    early_teacher = torch.ones(H, W, device=device) * 0.08
    
    stu_var = early_student.max() - early_student.min()
    tea_var = early_teacher.max() - early_teacher.min()
    print(f"Student variation: {stu_var:.6f}")
    print(f"Teacher variation: {tea_var:.6f}")
    print(f"Passes variation check (>0.1): {stu_var > 0.1 and tea_var > 0.1}")
    
    # Scenario 2: Training with some structure
    print("\n--- Scenario 2: Training with structure ---")
    structured_student = torch.zeros(H, W, device=device)
    structured_teacher = torch.zeros(H, W, device=device)
    
    # Add some structure
    center_x, center_y = H//2, W//2
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < 20:
                structured_student[i, j] = 0.3 + 0.2 * torch.randn(1).to(device)
                structured_teacher[i, j] = 0.4 + 0.1 * torch.randn(1).to(device)
            else:
                structured_student[i, j] = 0.1 + 0.05 * torch.randn(1).to(device)
                structured_teacher[i, j] = 0.08 + 0.02 * torch.randn(1).to(device)
    
    structured_student = torch.clamp(structured_student, 0, 1)
    structured_teacher = torch.clamp(structured_teacher, 0, 1)
    
    stu_var = structured_student.max() - structured_student.min()
    tea_var = structured_teacher.max() - structured_teacher.min()
    print(f"Student variation: {stu_var:.6f}")
    print(f"Teacher variation: {tea_var:.6f}")
    print(f"Passes variation check (>0.1): {stu_var > 0.1 and tea_var > 0.1}")
    
    if stu_var > 0.1 and tea_var > 0.1:
        # Test topology loss computation
        print("\n--- Testing Topology Loss Computation ---")
        
        try:
            std_loss = getTopoLoss(
                stu_tensor=structured_student,
                tea_tensor=structured_teacher,
                topo_size=16,
                pd_threshold=0.3
            )
            print(f"Standard Topo Loss: {std_loss.item():.6f}")
        except Exception as e:
            print(f"Standard Topo Loss ERROR: {e}")
        
        try:
            ua_loss = getUncertaintyAwareTopoLoss(
                stu_tensor=structured_student,
                tea_tensor=structured_teacher,
                topo_size=16,
                pd_threshold=0.3,
                uncertainty_threshold=0.7,
                focus_mode="confident"
            )
            print(f"UA Topo Loss (confident): {ua_loss.item():.6f}")
        except Exception as e:
            print(f"UA Topo Loss ERROR: {e}")

def debug_training_config():
    """
    Debug training configuration
    """
    print("\n=== DEBUG TRAINING CONFIG ===")
    
    # Simulate args
    class Args:
        use_topo_loss = 1
        use_uncertainty_topo = 1
        topo_weight = 0.03
        topo_rampup = 500.0
        topo_size = 32
        pd_threshold = 0.3
        uncertainty_threshold = 0.7
        topo_focus_mode = "confident"
    
    args = Args()
    
    print(f"use_topo_loss: {args.use_topo_loss}")
    print(f"use_uncertainty_topo: {args.use_uncertainty_topo}")
    print(f"topo_weight: {args.topo_weight}")
    print(f"topo_rampup: {args.topo_rampup}")
    
    # Test ramp-up
    for iter_num in [0, 10, 50, 100, 200, 500, 1000, 2000]:
        epoch = iter_num // 150  # Same as in training code
        
        # Simulate get_current_topo_weight function
        use_any_topo = bool(args.use_topo_loss) or bool(args.use_uncertainty_topo)
        if not use_any_topo:
            topo_weight = 0.0
        else:
            topo_weight = args.topo_weight * ramps.sigmoid_rampup(epoch, args.topo_rampup)
        
        print(f"Iter {iter_num:4d} (epoch {epoch:3d}): topo_weight={topo_weight:.6f}")

def debug_main():
    """
    Main debug function
    """
    print("=== DEBUGGING TOPOLOGY LOSS = 0 ISSUE ===")
    
    debug_sigmoid_rampup()
    debug_probability_maps()
    debug_training_config()
    
    print("\n=== POTENTIAL CAUSES OF TOPO LOSS = 0 ===")
    print("1. ❌ topo_weight = 0 due to ramp-up not started")
    print("2. ❌ Probability maps have insufficient variation (<0.1)")
    print("3. ❌ No valid samples pass the variation check")
    print("4. ❌ Error in topology loss computation")
    print("5. ❌ Logic bug in training code")
    
    print("\n=== SOLUTIONS ===")
    print("✅ Check debug logs during training")
    print("✅ Reduce topo_rampup for faster start")
    print("✅ Reduce variation threshold (0.1 -> 0.05)")
    print("✅ Check probability map statistics")
    print("✅ Verify args configuration")

if __name__ == "__main__":
    debug_main() 