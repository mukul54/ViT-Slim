#!/bin/bash

# List of all VTAB-1K datasets
VTAB_DATASETS=(
  "caltech101"
  "cifar"
  "dtd"
  "flowers102"
  "oxford_iiit_pet"
  "svhn"
  "sun397"
  "patch_camelyon"
  "eurosat"
  "resisc45"
  "retinopathy"
  "clevr_count"
  "clevr_dist"
  "dmlab"
  "kitti"
  "dsprites_loc"
  "dsprites_ori"
  "smallnorb_azi"
  "smallnorb_ele"
)

# Default paths (can be overridden with command-line arguments)
MODEL_PATH="/l/users/mukul.ranjan/glora/models/ViT-B_16.npz"
ROOT_DIR="/l/users/mukul.ranjan/glora/data"
SAVE_PATH="/l/users/mukul.ranjan/glora/models/glora_eval"
LOAD_PATH="/l/users/mukul.ranjan/glora/models/glora_cifar"
MAX_EPOCHS=20  # 20 epochs for vision tasks as per paper
POPULATION_NUM=50  # Initial population of 50 subnets
SELECT_NUM=10  # Select top K parents
M_PROB=0.2  # Mutation probability of 0.2 as per paper
RANK=4  # LoRA rank
SEED=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --root-dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    --save-path)
      SAVE_PATH="$2"
      shift 2
      ;;
    --load-path)
      LOAD_PATH="$2"
      shift 2
      ;;
    --max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --population-num)
      POPULATION_NUM="$2"
      shift 2
      ;;
    --select-num)
      SELECT_NUM="$2"
      shift 2
      ;;
    --m-prob)
      M_PROB="$2"
      shift 2
      ;;
    --rank)
      RANK="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p $SAVE_PATH

# Function to run evolution on a dataset with dataset-specific learning rates
run_evolution() {
  local dataset=$1
  local learning_rate="5e-4" # Default learning rate for most datasets
  
  # Set dataset-specific learning rates according to the paper
  case $dataset in
    "retinopathy"|"clevr_count"|"clevr_dist"|"dmlab"|"smallnorb_ele")
      learning_rate="1e-4"
      ;;
    *)
      learning_rate="5e-4"
      ;;
  esac
  
  echo "====================================="
  echo "Evaluating on $dataset with LR=$learning_rate..."
  echo "====================================="
  python evolution.py \
    --dataset $dataset \
    --model_path "$MODEL_PATH" \
    --root_dir "$ROOT_DIR" \
    --save_path "$SAVE_PATH" \
    --load_path "$LOAD_PATH" \
    --max-epochs $MAX_EPOCHS \
    --population-num $POPULATION_NUM \
    --select-num $SELECT_NUM \
    --m_prob $M_PROB \
    --rank $RANK \
    --seed $SEED \
    --learning-rate $learning_rate
  
  echo "Finished evaluating $dataset"
  echo ""
}

# Loop through each dataset
for dataset in "${VTAB_DATASETS[@]}"; do
  run_evolution $dataset
done

echo "All evaluations complete!"
echo "Results saved to $SAVE_PATH"
