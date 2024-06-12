#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_augm_30
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

source ../venv/bin/activate

n_epochs=6000

davis_part=0.5
yv_part=0


save_model_path="/work3/s220493/saves/various_sizes_datasets/"
exp_name="davis-$davis_part-yv-$yv_part-2"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"

augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
augm_datasets=['davis']
augm_p=[0.75]


torchrun --nproc_per_node=2 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path=$save_model_path \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 load_model=$load_model \
 +augmentations.augmentation_datasets=$augm_datasets \
 +augmentations.augmentation_p=$augm_p
