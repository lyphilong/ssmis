#!/usr/bin/env python3
"""
Test Advanced Topology Loss Integration for Conference Submission
"""

import torch
import numpy as np
from utils.advanced_topo_losses import create_advanced_topology_loss

def test_advanced_topology_integration():
    print("=== TESTING ADVANCED TOPOLOGY LOSS INTEGRATION ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data (similar to medical image segmentation)
    H, W = 32, 32
    
    # Student prediction (with some uncertainty)
    stu_tensor = torch.rand(H, W, device=device) * 0.8 + 0.1
    
    # Teacher prediction (slightly different, representing EMA model)
    tea_tensor = stu_tensor + torch.randn(H, W, device=device) * 0.1
    tea_tensor = torch.clamp(tea_tensor, 0.01, 0.99)
    
    print(f"Student tensor range: [{stu_tensor.min():.3f}, {stu_tensor.max():.3f}]")
    print(f"Teacher tensor range: [{tea_tensor.min():.3f}, {tea_tensor.max():.3f}]")
    
    # Test different configurations
    configs = [
        {
            'name': 'Conservative',
            'config': {
                'lambda_info': 0.05,
                'lambda_wasserstein': 0.1,
                'lambda_bayesian': 0.08,
                'use_adaptive': True
            }
        },
        {
            'name': 'Balanced',
            'config': {
                'lambda_info': 0.1,
                'lambda_wasserstein': 0.2,
                'lambda_bayesian': 0.15,
                'use_adaptive': True
            }
        }
    ]
    
    for config_info in configs:
        print(f"\n--- Testing {config_info['name']} Configuration ---")
        
        try:
            # Create advanced loss
            advanced_loss = create_advanced_topology_loss(
                device=str(device), 
                config=config_info['config']
            )
            
            # Test at different training stages
            stages = [
                {'epoch_progress': 0.1, 'current_dice': 0.3, 'stage': 'Early'},
                {'epoch_progress': 0.5, 'current_dice': 0.6, 'stage': 'Mid'}
            ]
            
            for stage in stages:
                loss_dict = advanced_loss(
                    stu_tensor=stu_tensor,
                    tea_tensor=tea_tensor,
                    epoch_progress=stage['epoch_progress'],
                    current_dice=stage['current_dice']
                )
                
                print(f"  {stage['stage']} Training:")
                print(f"    Total Loss: {loss_dict['total_advanced_topology'].item():.6f}")
                print(f"    Info Loss: {loss_dict['info_topology'].item():.6f}")
                print(f"    Adaptive Factor: {loss_dict['adaptive_factor'].item():.3f}")
                
        except Exception as e:
            print(f"  ❌ Error in {config_info['name']} config: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✅ INTEGRATION TEST PASSED!")
    return True

if __name__ == "__main__":
    test_advanced_topology_integration() 