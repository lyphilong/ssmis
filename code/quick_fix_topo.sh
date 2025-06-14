#!/bin/bash
# Quick fix to see topology loss immediately

echo "=== QUICK FIX: Immediate Topology Loss ==="

# Test with very fast ramp-up and lower threshold
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "QUICKFIX-immediate-topo" \
--model "unet_3D" \
--max_iterations 1000 \
--temp 0.6 \
--batch_size 4 \
--labelnum 8 \
--gpu_id 0 \
--use_topo_loss 0 \
--use_uncertainty_topo 1 \
--topo_weight 0.1 \
--topo_size 16 \
--pd_threshold 0.2 \
--topo_rampup 10 \
--uncertainty_threshold 0.5 \
--topo_focus_mode confident

echo "Quick fix completed! Check logs for topo loss values." 