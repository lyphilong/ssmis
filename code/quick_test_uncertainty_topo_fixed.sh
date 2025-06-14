#!/bin/bash

echo "=== QUICK TEST: UNCERTAINTY-AWARE TOPOLOGY LOSS (FIXED) ==="

python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "BraTS2019-uncertainty-topo-FIXED" \
--model "unet_3D" \
--max_iterations 100 \
--temp 0.6 \
--batch_size 4 \
--labelnum 8 \
--gpu_id 0 \
--use_topo_loss 0 \
--use_uncertainty_topo 1 \
--topo_weight 0.05 \
--topo_size 16 \
--pd_threshold 0.2 \
--topo_rampup 20 \
--uncertainty_threshold 0.6 \
--topo_focus_mode confident

echo "Quick test completed! Check for non-zero topology loss in output." 