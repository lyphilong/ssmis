#!/bin/bash

# =================================================================
# Conference-Level Training Script for Advanced Topology Loss
# Target: CVPR/ICCV/ECCV/NeurIPS/ICML/ICLR Submission
# =================================================================

echo "🎯 Starting Conference-Level Training with Advanced Topology Loss"
echo "=================================================="

# Training configurations for different experiments
declare -A configs

# Configuration 1: Conservative (stable baseline)
configs["conservative"]="--use_advanced_topo 1 --lambda_info 0.05 --lambda_wasserstein 0.1 --lambda_bayesian 0.08 --use_adaptive_weight 1"

# Configuration 2: Balanced (recommended for paper)
configs["balanced"]="--use_advanced_topo 1 --lambda_info 0.1 --lambda_wasserstein 0.2 --lambda_bayesian 0.15 --use_adaptive_weight 1"

# Configuration 3: Aggressive (for ablation study)
configs["aggressive"]="--use_advanced_topo 1 --lambda_info 0.2 --lambda_wasserstein 0.3 --lambda_bayesian 0.25 --use_adaptive_weight 1"

# Configuration 4: No adaptive (ablation)
configs["no_adaptive"]="--use_advanced_topo 1 --lambda_info 0.1 --lambda_wasserstein 0.2 --lambda_bayesian 0.15 --use_adaptive_weight 0"

# Base training parameters
BASE_ARGS="
--root_dir ../data/BraTS2019
--exp BraTS2019_Conference
--gpu_id 1
--seed 1337
--deterministic 1
--model unet_3D
--in_ch 1
--num_classes 2
--max_iterations 20000
--batch_size 8
--labeled_bs 4
--base_lr 0.01
--labelnum 8
--ema_decay 0.99
--consistency 0.1
--consistency_type mse
--consistency_rampup 200.0
--gamma 2.0
--beta_min 0.5
--beta_max 5.0
--temp 0.6
--l_weight 1.0
--u_weight 0.5
--use_focal 1
--use_teacher_loss 1
--topo_weight 0.01
--topo_size 32
--pd_threshold 0.3
--topo_rampup 500.0
"

# Function to run single experiment
run_experiment() {
    local config_name=$1
    local config_args=$2
    
    echo ""
    echo "🚀 Running Experiment: $config_name"
    echo "Configuration: $config_args"
    echo "----------------------------------------"
    
    # Create experiment directory
    exp_dir="../experiments/conference_${config_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$exp_dir"
    
    # Run training
    python train_DyCON_BraTS19_v2_conference.py \
        $BASE_ARGS \
        $config_args \
        --exp "Conference_${config_name}" \
        2>&1 | tee "$exp_dir/training.log"
    
    # Check if training completed successfully
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Experiment $config_name completed successfully"
        echo "📁 Results saved to: $exp_dir"
    else
        echo "❌ Experiment $config_name failed"
        echo "📁 Logs saved to: $exp_dir/training.log"
    fi
}

# Main execution
main() {
    echo "Available configurations:"
    for config in "${!configs[@]}"; do
        echo "  - $config"
    done
    echo ""
    
    # Check if specific configuration is requested
    if [ $# -eq 1 ]; then
        config_name=$1
        if [[ -n "${configs[$config_name]}" ]]; then
            run_experiment "$config_name" "${configs[$config_name]}"
        else
            echo "❌ Unknown configuration: $config_name"
            echo "Available: ${!configs[@]}"
            exit 1
        fi
    else
        # Run all configurations
        echo "🎯 Running all configurations for comprehensive evaluation..."
        
        for config_name in "${!configs[@]}"; do
            run_experiment "$config_name" "${configs[$config_name]}"
            
            # Brief pause between experiments
            echo "⏳ Waiting 30 seconds before next experiment..."
            sleep 30
        done
        
        echo ""
        echo "🏆 All conference experiments completed!"
        echo "🔬 Ready for paper submission analysis"
    fi
}

# Usage information
usage() {
    echo "Usage: $0 [configuration_name]"
    echo ""
    echo "Available configurations:"
    echo "  conservative  - Stable baseline with conservative hyperparameters"
    echo "  balanced      - Recommended configuration for paper results"  
    echo "  aggressive    - High-impact configuration for ablation study"
    echo "  no_adaptive   - Ablation study without adaptive weighting"
    echo ""
    echo "If no configuration is specified, all will be run sequentially."
    echo ""
    echo "Examples:"
    echo "  $0 balanced           # Run only balanced configuration"
    echo "  $0                    # Run all configurations"
}

# Handle command line arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Ensure we're in the right directory
if [ ! -f "train_DyCON_BraTS19_v2_conference.py" ]; then
    echo "❌ Error: train_DyCON_BraTS19_v2_conference.py not found"
    echo "Please run this script from the DyCON/code directory"
    exit 1
fi

# Run main function
main "$@"

echo ""
echo "🎯 Conference Training Script Complete!"
echo "📊 Results ready for CVPR/ICCV/ECCV/NeurIPS analysis" 