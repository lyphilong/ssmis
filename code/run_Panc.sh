# Description: Run the training code for DyCON-PancreasCT

# Main training script
python train_DyCON_Pancreas.py \
--root_dir "../data/Pancreas" \
--exp "PancreasCT" \
--model "unet_3D" \
--max_iterations 20000 \
--temp 0.6 \
--batch_size 8 \
--labelnum 12 \
--gpu_id 3 \
--use_topo_loss 1 \
--topo_weight 0.1 \
--topo_size 32 \
--pd_threshold 0.3 \
--topo_rampup 500
