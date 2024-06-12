#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J davis_from_statpretrained
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
yv_part=0

augm_p=[0]
augmentation_datasets=[]


# exp_simple_davis or exp1
augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
stage=2

save_model_path="/work3/s220493/saves/comparison/"
exp_name="davis-$davis_part-from-pretrained_static"
load_model="/work3/s220493/saves/STCN_stage0.pth"


torchrun --nproc_per_node=1 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs augmentations.augmentation_p=$augm_p\
 davis_root=$davis_root \
 stage=$stage \
 save_model_path=$save_model_path \
 load_network=$load_model \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 augmentations.augmentation_datasets=$augmentation_datasets
