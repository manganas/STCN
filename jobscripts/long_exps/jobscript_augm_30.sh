#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_augm_30
#BSUB -n 8
#BSUB -W 48:00
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



# exp_simple_davis or exp1
augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
augm_datasets=['davis']
augm_p=[0.75]

load_model="/work3/s220493/saves/various_sizes_datasets/checkpoint_davis-0.5-yv-0-2_checkpoint.pth"

torchrun --nproc_per_node=2 --standalone train.py exp_name="davis-$davis_part-yv-$yv_part-2"\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path='/work3/s220493/saves/various_sizes_datasets/' \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 load_model=$load_model \
 +augmentations.augmentation_datasets=$augm_datasets \
 +augmentations.augmentation_p=$augm_p
