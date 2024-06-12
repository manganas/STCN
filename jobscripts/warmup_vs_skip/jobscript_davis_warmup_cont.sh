#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_daviswarmupc
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

n_epochs=6000

davis_part=1


save_model_path="/work3/s220493/saves/warmup_vs_skip/"

augmentations=exp_multi_data
augm_datasets=['davis']
exp_name="davis-davis-warmup"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"
augm_probs=[0.75,0.5,0.25]


torchrun --nproc_per_node=2 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 save_model_path=$save_model_path \
 davis_part=$davis_part \
 load_model=$load_model \
 augmentations.augmentation_p=$augm_probs \
 +augmentations.augmentation_datasets=$augm_datasets
