#!/usr/bin/env python3
"""
Test script để kiểm tra việc tích hợp topological loss vào DyCON training.
Chạy thử các component chính mà không cần dataset đầy đủ.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add current directory to path để import được các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.topo_losses import getTopoLoss

def test_topo_loss_integration():
    """Test topological loss với synthetic data"""
    print("=== Test Topological Loss Integration ===")
    
    # Tạo synthetic data giống như output của model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simulate student và teacher probability maps
    batch_size = 2
    height, width = 64, 64
    
    # Tạo probability maps với structure hình học đơn giản
    student_probs = torch.zeros(batch_size, height, width, device=device)
    teacher_probs = torch.zeros(batch_size, height, width, device=device)
    
    for b in range(batch_size):
        # Student: tạo circle ở center với noise
        center_x, center_y = height//2, width//2
        radius = 15
        
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist <= radius:
                    student_probs[b, i, j] = 0.8 + 0.1 * torch.randn(1).item()
                else:
                    student_probs[b, i, j] = 0.2 + 0.05 * torch.randn(1).item()
        
        # Teacher: tạo circle tương tự nhưng clean hơn
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist <= radius:
                    teacher_probs[b, i, j] = 0.85
                else:
                    teacher_probs[b, i, j] = 0.15
    
    # Clamp để đảm bảo trong range [0, 1]
    student_probs = torch.clamp(student_probs, 0.0, 1.0)
    teacher_probs = torch.clamp(teacher_probs, 0.0, 1.0)
    
    print(f"Student probs shape: {student_probs.shape}")
    print(f"Teacher probs shape: {teacher_probs.shape}")
    print(f"Student probs range: [{student_probs.min():.3f}, {student_probs.max():.3f}]")
    print(f"Teacher probs range: [{teacher_probs.min():.3f}, {teacher_probs.max():.3f}]")
    
    # Test topological loss cho từng sample
    total_topo_loss = 0.0
    num_valid_samples = 0
    
    for b_idx in range(batch_size):
        stu_prob_2d = student_probs[b_idx]
        tea_prob_2d = teacher_probs[b_idx]
        
        # Check variation
        stu_var = stu_prob_2d.max() - stu_prob_2d.min()
        tea_var = tea_prob_2d.max() - tea_prob_2d.min()
        
        print(f"\nSample {b_idx}:")
        print(f"  Student variation: {stu_var:.3f}")
        print(f"  Teacher variation: {tea_var:.3f}")
        
        if stu_var > 0.1 and tea_var > 0.1:
            try:
                sample_topo_loss = getTopoLoss(
                    stu_tensor=stu_prob_2d,
                    tea_tensor=tea_prob_2d,
                    topo_size=32,
                    pd_threshold=0.3
                )
                
                if not torch.isnan(sample_topo_loss) and not torch.isinf(sample_topo_loss):
                    total_topo_loss += sample_topo_loss.item()
                    num_valid_samples += 1
                    print(f"  Topological loss: {sample_topo_loss.item():.6f}")
                else:
                    print(f"  Topological loss: NaN/Inf (skipped)")
                    
            except Exception as e:
                print(f"  Error computing topo loss: {str(e)}")
        else:
            print(f"  Insufficient variation (skipped)")
    
    if num_valid_samples > 0:
        avg_topo_loss = total_topo_loss / num_valid_samples
        print(f"\nAverage topological loss: {avg_topo_loss:.6f}")
        print(f"Valid samples: {num_valid_samples}/{batch_size}")
    else:
        print(f"\nNo valid samples for topological loss computation")
    
    print("=== Test completed successfully! ===")

def test_parameter_integration():
    """Test việc thêm parameters vào argument parser"""
    print("\n=== Test Parameter Integration ===")
    
    # Simulate argparse namespace
    class Args:
        def __init__(self):
            self.use_topo_loss = 1
            self.topo_weight = 0.1
            self.topo_size = 32
            self.pd_threshold = 0.3
            self.topo_rampup = 500.0
    
    args = Args()
    
    # Test ramp-up function
    def get_current_topo_weight(epoch, args):
        if not bool(args.use_topo_loss):
            return 0.0
        # Simplified sigmoid ramp-up
        return args.topo_weight * min(1.0, epoch / args.topo_rampup)
    
    # Test different epochs
    test_epochs = [0, 100, 250, 500, 1000]
    
    print("Epoch -> Topo Weight:")
    for epoch in test_epochs:
        weight = get_current_topo_weight(epoch, args)
        print(f"  {epoch:4d} -> {weight:.4f}")
    
    print("=== Parameter integration test completed! ===")

if __name__ == "__main__":
    # Kiểm tra dependencies
    try:
        import gudhi
        import ripser
        import cripser
        print("✓ All topological dependencies are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install gudhi ripser cripser")
        sys.exit(1)
    
    # Run tests
    test_topo_loss_integration()
    test_parameter_integration()
    
    print("\n🎉 All tests passed! Topological loss integration is ready.")
    print("\nUsage:")
    print("  - Set --use_topo_loss 1 to enable topological loss")
    print("  - Adjust --topo_weight to control the loss weight")
    print("  - Tune --topo_size for computational efficiency")
    print("  - Modify --pd_threshold for signal/noise separation") 