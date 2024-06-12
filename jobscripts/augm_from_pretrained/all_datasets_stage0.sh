#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J allDatasetsStage0
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

source /work3/s220493/venv/bin/activate

n_epochs=3000

davis_part=1
yv_part=0

save_model_path="/work3/s220493/saves/from_pretrained/all_datasets"
exp_name="davis-$davis_part-yv-$yv_part-all-augm-datasets"
load_model="/work3/s220493/from_pretrained/all_datasets/checkpoint_davis-1-yv-0-all-augm-datasets_checkpoint.pth"

augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"

torchrun --nproc_per_node=1 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path=$save_model_path \
 load_model=$load_model \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 augmentations=$augmentations