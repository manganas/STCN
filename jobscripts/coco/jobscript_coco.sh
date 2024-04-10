#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_coco
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

n_epochs=4000

augmentations=exp2
davis_part=1
yv_part=0



davis_root="/work3/s220493/DAVIS"
torchrun --nproc_per_node=2 --standalone train.py exp_name="COCO-multi-coco"\
 n_epochs=$n_epochs augmentations=$augmentations\
 augmentations.use_coco=True \
 save_model_path='/work3/s220493/saves/various_sizes_datasets/' \
 davis_part=$davis_part \
 yt_vos_part=$yv_part
