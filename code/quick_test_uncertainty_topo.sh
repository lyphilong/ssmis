#!/bin/bash
# Quick test script for Uncertainty-Aware Topological Loss
# For rapid prototyping and debugging (shorter iterations)

echo "=== QUICK TEST: Uncertainty-Aware Topological Loss ==="

# Test with short iterations for debugging
ITERATIONS=1000
BATCH_SIZE=4
LABELNUM=8

# ================================================================
# TEST 1: Uncertainty-Aware Topology Loss (confident focus)
# ================================================================
echo "Testing uncertainty-aware topology loss (confident focus)..."
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "TEST-uncertainty-topo-confident" \
--model "unet_3D" \
--max_iterations $ITERATIONS \
--temp 0.6 \
--batch_size $BATCH_SIZE \
--labelnum $LABELNUM \
--gpu_id 0 \
--use_topo_loss 0 \
--use_uncertainty_topo 1 \
--topo_weight 0.01 \
--topo_size 16 \
--pd_threshold 0.3 \
--topo_rampup 200 \
--uncertainty_threshold 0.7 \
--topo_focus_mode confident

echo "Test 1 completed!"

# ================================================================
# TEST 2: Uncertainty-Aware Topology Loss (uncertain focus)
# ================================================================
echo "Testing uncertainty-aware topology loss (uncertain focus)..."
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "TEST-uncertainty-topo-uncertain" \
--model "unet_3D" \
--max_iterations $ITERATIONS \
--temp 0.6 \
--batch_size $BATCH_SIZE \
--labelnum $LABELNUM \
--gpu_id 0 \
--use_topo_loss 0 \
--use_uncertainty_topo 1 \
--topo_weight 0.01 \
--topo_size 16 \
--pd_threshold 0.3 \
--topo_rampup 200 \
--uncertainty_threshold 0.7 \
--topo_focus_mode uncertain

echo "Test 2 completed!"

# ================================================================
# TEST 3: Standard topology loss for comparison  
# ================================================================
echo "Testing standard topology loss for comparison..."
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "TEST-standard-topo" \
--model "unet_3D" \
--max_iterations $ITERATIONS \
--temp 0.6 \
--batch_size $BATCH_SIZE \
--labelnum $LABELNUM \
--gpu_id 0 \
--use_topo_loss 1 \
--use_uncertainty_topo 0 \
--topo_weight 0.01 \
--topo_size 16 \
--pd_threshold 0.3 \
--topo_rampup 200

echo "Test 3 completed!"

echo "=== QUICK TESTS COMPLETED ==="
echo "Check tensorboard logs and wandb for preliminary results:"
echo "  - TEST-uncertainty-topo-confident/"
echo "  - TEST-uncertainty-topo-uncertain/" 
echo "  - TEST-standard-topo/"

# Also run the demo comparison
echo "Running demo comparison..."
python demo_uncertainty_topo.py

echo "All tests and demos completed!" 