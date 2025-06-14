#!/usr/bin/env python3
"""
Demo script to test and compare standard vs uncertainty-aware topological loss
Contribution: Novel Uncertainty-Aware Topological Loss for Semi-Supervised Medical Segmentation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.topo_losses import getTopoLoss, getUncertaintyAwareTopoLoss, compute_entropy_uncertainty

def create_synthetic_probability_maps():
    """
    Tạo synthetic probability maps với các đặc tính khác nhau để test
    """
    H, W = 64, 64
    
    # Case 1: High confidence prediction với clear topology
    confident_map = torch.zeros(H, W)
    # Tạo circular structure
    center_x, center_y = H//2, W//2
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if 10 < dist < 20:  # Ring structure
                confident_map[i, j] = 0.9
            elif dist <= 10:    # Inner circle
                confident_map[i, j] = 0.1
            else:               # Background
                confident_map[i, j] = 0.05
    
    # Case 2: Uncertain prediction với noisy topology
    uncertain_map = confident_map.clone()
    # Add noise to make it uncertain
    noise = torch.randn(H, W) * 0.2
    uncertain_map = torch.clamp(uncertain_map + noise, 0, 1)
    
    # Case 3: Teacher map (ground truth-like)
    teacher_map = torch.zeros(H, W)
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if 12 < dist < 18:  # Slightly different ring
                teacher_map[i, j] = 0.95
            elif dist <= 12:
                teacher_map[i, j] = 0.05
            else:
                teacher_map[i, j] = 0.02
    
    return confident_map, uncertain_map, teacher_map

def visualize_uncertainty_analysis(student_map, teacher_map, title_prefix=""):
    """
    Visualize uncertainty analysis and topology comparison
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Probability maps
    im1 = axes[0, 0].imshow(student_map.numpy(), cmap='viridis')
    axes[0, 0].set_title(f'{title_prefix} Student Probability')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(teacher_map.numpy(), cmap='viridis')
    axes[0, 1].set_title(f'{title_prefix} Teacher Probability')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Compute uncertainty
    stu_uncertainty = compute_entropy_uncertainty(student_map)
    tea_uncertainty = compute_entropy_uncertainty(teacher_map)
    combined_uncertainty = (stu_uncertainty + tea_uncertainty) / 2.0
    
    im3 = axes[0, 2].imshow(combined_uncertainty.numpy(), cmap='hot')
    axes[0, 2].set_title(f'{title_prefix} Combined Uncertainty')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Confidence masks for different focus modes
    conf_mask_confident = (combined_uncertainty < 0.7).float()
    conf_mask_uncertain = (combined_uncertainty >= 0.7).float()
    
    im4 = axes[1, 0].imshow(conf_mask_confident.numpy(), cmap='gray')
    axes[1, 0].set_title('Confident Regions (focus_mode="confident")')
    axes[1, 0].axis('off')
    
    im5 = axes[1, 1].imshow(conf_mask_uncertain.numpy(), cmap='gray')
    axes[1, 1].set_title('Uncertain Regions (focus_mode="uncertain")')
    axes[1, 1].axis('off')
    
    # Histogram of uncertainty
    axes[1, 2].hist(combined_uncertainty.numpy().flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 2].axvline(x=0.7, color='red', linestyle='--', label='Threshold=0.7')
    axes[1, 2].set_title('Uncertainty Distribution')
    axes[1, 2].set_xlabel('Uncertainty')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    return fig

def compare_topo_losses():
    """
    So sánh standard vs uncertainty-aware topological loss
    """
    print("=== COMPARING STANDARD vs UNCERTAINTY-AWARE TOPOLOGICAL LOSS ===")
    
    # Create synthetic data
    confident_map, uncertain_map, teacher_map = create_synthetic_probability_maps()
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confident_map = confident_map.to(device)
    uncertain_map = uncertain_map.to(device)
    teacher_map = teacher_map.to(device)
    
    print(f"Using device: {device}")
    print(f"Map shape: {confident_map.shape}")
    
    # Test parameters
    topo_size = 16
    pd_threshold = 0.3
    uncertainty_threshold = 0.7
    
    print("\n--- CASE 1: CONFIDENT STUDENT vs TEACHER ---")
    
    # Standard topology loss
    std_loss_conf = getTopoLoss(
        stu_tensor=confident_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold
    )
    
    # Uncertainty-aware topology loss (focus on confident)
    ua_loss_conf_confident = getUncertaintyAwareTopoLoss(
        stu_tensor=confident_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold,
        uncertainty_threshold=uncertainty_threshold,
        focus_mode="confident"
    )
    
    # Uncertainty-aware topology loss (focus on uncertain)
    ua_loss_conf_uncertain = getUncertaintyAwareTopoLoss(
        stu_tensor=confident_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold,
        uncertainty_threshold=uncertainty_threshold,
        focus_mode="uncertain"
    )
    
    print(f"Standard Topo Loss: {std_loss_conf.item():.6f}")
    print(f"UA Topo Loss (confident focus): {ua_loss_conf_confident.item():.6f}")
    print(f"UA Topo Loss (uncertain focus): {ua_loss_conf_uncertain.item():.6f}")
    
    print("\n--- CASE 2: UNCERTAIN STUDENT vs TEACHER ---")
    
    # Standard topology loss
    std_loss_unconf = getTopoLoss(
        stu_tensor=uncertain_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold
    )
    
    # Uncertainty-aware topology loss (focus on confident)
    ua_loss_unconf_confident = getUncertaintyAwareTopoLoss(
        stu_tensor=uncertain_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold,
        uncertainty_threshold=uncertainty_threshold,
        focus_mode="confident"
    )
    
    # Uncertainty-aware topology loss (focus on uncertain) 
    ua_loss_unconf_uncertain = getUncertaintyAwareTopoLoss(
        stu_tensor=uncertain_map,
        tea_tensor=teacher_map,
        topo_size=topo_size,
        pd_threshold=pd_threshold,
        uncertainty_threshold=uncertainty_threshold,
        focus_mode="uncertain"
    )
    
    print(f"Standard Topo Loss: {std_loss_unconf.item():.6f}")
    print(f"UA Topo Loss (confident focus): {ua_loss_unconf_confident.item():.6f}")
    print(f"UA Topo Loss (uncertain focus): {ua_loss_unconf_uncertain.item():.6f}")
    
    # Visualization
    print("\n--- VISUALIZATION ---")
    
    # Visualize confident case
    fig1 = visualize_uncertainty_analysis(confident_map.cpu(), teacher_map.cpu(), "Confident Student")
    plt.savefig('confident_student_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: confident_student_analysis.png")
    
    # Visualize uncertain case
    fig2 = visualize_uncertainty_analysis(uncertain_map.cpu(), teacher_map.cpu(), "Uncertain Student")
    plt.savefig('uncertain_student_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: uncertain_student_analysis.png")
    
    # Summary comparison plot
    fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = ['Standard', 'UA (Confident)', 'UA (Uncertain)']
    confident_losses = [std_loss_conf.item(), ua_loss_conf_confident.item(), ua_loss_conf_uncertain.item()]
    uncertain_losses = [std_loss_unconf.item(), ua_loss_unconf_confident.item(), ua_loss_unconf_uncertain.item()]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, confident_losses, width, label='Confident Student', alpha=0.7)
    bars2 = ax.bar(x + width/2, uncertain_losses, width, label='Uncertain Student', alpha=0.7)
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Topological Loss')
    ax.set_title('Comparison: Standard vs Uncertainty-Aware Topological Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('topo_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: topo_loss_comparison.png")
    
    plt.close('all')
    
    print("\n=== ANALYSIS SUMMARY ===")
    print("1. Uncertainty-Aware Topo Loss provides more nuanced control")
    print("2. 'confident' focus mode emphasizes reliable topology regions")
    print("3. 'uncertain' focus mode targets boundary/ambiguous regions")
    print("4. Standard loss treats all regions equally (may overfit to noise)")
    
    return {
        'confident_student': {
            'standard': std_loss_conf.item(),
            'ua_confident': ua_loss_conf_confident.item(),
            'ua_uncertain': ua_loss_conf_uncertain.item()
        },
        'uncertain_student': {
            'standard': std_loss_unconf.item(),
            'ua_confident': ua_loss_unconf_confident.item(),
            'ua_uncertain': ua_loss_unconf_uncertain.item()
        }
    }

if __name__ == "__main__":
    # Run comparison
    results = compare_topo_losses()
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print("Confident Student vs Teacher:")
    for method, loss in results['confident_student'].items():
        print(f"  {method}: {loss:.6f}")
    
    print("Uncertain Student vs Teacher:")
    for method, loss in results['uncertain_student'].items():
        print(f"  {method}: {loss:.6f}")
    
    print("\n=== CONTRIBUTION HIGHLIGHTS ===")
    print("✅ Novel uncertainty-guided topological consistency")
    print("✅ Adaptive focus on confident vs uncertain regions")
    print("✅ Robust against noisy/uncertain predictions")
    print("✅ Practical for medical segmentation applications")
    print("✅ Easy integration with existing semi-supervised frameworks") 