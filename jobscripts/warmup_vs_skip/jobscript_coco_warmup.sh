#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_cocowarmup
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."

nvidia-smi
module swap python3/3.9.11
module swap cuda/11.6

source /work3/s220493/venv/bin/activate

n_epochs=3000

davis_part=1



# exp_simple_davis or exp1
augmentations=exp_multi_data
augm_datasets=['coco']
exp_name="davis-coco-warmup"
load_model="checkpoint_$exp_name\_checkpoint.pth"
augm_probs=[0]

torchrun --nproc_per_node=2 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 save_model_path='/work3/s220493/saves/warmup_vs_skip/' \
 davis_part=$davis_part \
 load_model=$load_model \
 augmentations.augmentation_p=$augm_probs \
 +augmentations.augmentation_datasets=$augm_datasets \
