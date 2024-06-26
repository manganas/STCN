#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J davis_pretraining
#BSUB -n 4
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

augm_p=[0]
davis_part=1
yv_part=0

stage=0

save_model_path="/work3/s220493/saves/comparison/"
exp_name="davis_static_pretraining"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"

wandb_log=False
validation_step=False
online_validation=False


# exp_simple_davis or exp1
augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS/2017/trainval"
torchrun --nproc_per_node=1 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs augmentations.augmentation_p=$augm_p\
 davis_root=$davis_root \
 stage=$stage \
 save_model_path=$save_model_path \
 load_model=$load_model \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 wandb_log=$wandb_log \
 validation_step=$validation_step \
 online_validation=$online_validation

