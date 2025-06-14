# # Description: Run the training code for DyCON-BraTS2019 with different topological loss variants
# # ================================================================
# # MULTIPLE TRAINING CONFIGURATIONS FOR COMPARISON
# # ================================================================

# echo "=== DYCON-BraTS19 Training with Topological Loss Variants ==="

# # ================================================================
# # 1. BASELINE: Standard DyCON (no topology loss)
# # ================================================================
# echo "Starting BASELINE training (no topology loss)..."
# python train_DyCON_BraTS19.py \
# --root_dir "../data/BraTS2019" \
# --exp "BraTS2019-baseline-no-topo" \
# --model "unet_3D" \
# --max_iterations 20000 \
# --temp 0.6 \
# --batch_size 8 \
# --labelnum 25 \
# --gpu_id 0 \
# --use_topo_loss 0 \
# --use_uncertainty_topo 0

# echo "Baseline training completed!"

# # ================================================================  
# # 2. STANDARD TOPOLOGY LOSS (existing approach)
# # ================================================================
# echo "Starting STANDARD topology loss training..."
# python train_DyCON_BraTS19.py \
# --root_dir "../data/BraTS2019" \
# --exp "BraTS2019-standard-topo" \
# --model "unet_3D" \
# --max_iterations 20000 \
# --temp 0.6 \
# --batch_size 8 \
# --labelnum 25 \
# --gpu_id 0 \
# --use_topo_loss 1 \
# --topo_weight 0.01 \
# --topo_size 32 \
# --pd_threshold 0.3 \
# --topo_rampup 500 \
# --use_uncertainty_topo 0

# echo "Standard topology training completed!"

# ================================================================
# 3. NEW: UNCERTAINTY-AWARE TOPOLOGY LOSS (confident focus)
# ================================================================
# echo "Starting UNCERTAINTY-AWARE topology loss training (confident focus)..."
# python train_DyCON_BraTS19.py \
# --root_dir "../data/BraTS2019" \
# --exp "BraTS2019-uncertainty-topo-confident" \
# --model "unet_3D" \
# --max_iterations 20000 \
# --temp 0.6 \
# --batch_size 8 \
# --labelnum 25 \
# --gpu_id 0 \
# --use_topo_loss 0 \
# --use_uncertainty_topo 1 \
# --topo_weight 0.01 \
# --topo_size 32 \
# --pd_threshold 0.3 \
# --topo_rampup 500 \
# --uncertainty_threshold 0.7 \
# --topo_focus_mode confident

echo "Uncertainty-aware topology (confident) training completed!"

# ================================================================
# 4. NEW: UNCERTAINTY-AWARE TOPOLOGY LOSS (uncertain focus)  
# ================================================================
echo "Starting UNCERTAINTY-AWARE topology loss training (uncertain focus)..."
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "BraTS2019-uncertainty-topo-uncertain" \
--model "unet_3D" \
--max_iterations 20000 \
--temp 0.6 \
--batch_size 8 \
--labelnum 25 \
--gpu_id 0 \
--use_topo_loss 1 \
--use_uncertainty_topo 1 \
--topo_weight 0.3 \
--topo_size 32 \
--pd_threshold 0.3 \
--topo_rampup 500 \
--uncertainty_threshold 0.7 \
--topo_focus_mode confident

echo "Uncertainty-aware topology (uncertain) training completed!"

echo "=== ALL TRAINING EXPERIMENTS COMPLETED ==="
echo "Check results in ../models/ directory for comparison:"
echo "  - BraTS2019-baseline-no-topo/"
echo "  - BraTS2019-standard-topo/"  
echo "  - BraTS2019-uncertainty-topo-confident/"
echo "  - BraTS2019-uncertainty-topo-uncertain/"