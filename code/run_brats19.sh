# Description: Run the training code for DyCON-BraTS2019

# Main training script
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "BraTS2019-topo-2d-only" \
--model "unet_3D" \
--max_iterations 2000 \
--temp 0.6 \
--batch_size 8 \
--labelnum 25 \
--gpu_id 1 \
--use_topo_loss 1 \
--topo_size 32 \
--pd_threshold 0.3 \
--topo_rampup 500 