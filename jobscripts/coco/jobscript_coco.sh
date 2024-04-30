#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_coco
#BSUB -n 8
#BSUB -W 36:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."

nvidia-smi
module swap python3/3.9.11
module swap cuda/11.6

source ../venv/bin/activate

n_epochs=3000

augmentations=exp_multi_data
augm_datasets=['coco']

davis_part=1
yv_part=0



davis_root="/work3/s220493/DAVIS"
torchrun --nproc_per_node=2 --standalone train.py exp_name="COCO-multi-coco"\
 n_epochs=$n_epochs augmentations=$augmentations\
 save_model_path='/work3/s220493/saves/coco/' \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 augmentations.augmentation_datasets=$augm_datasets
