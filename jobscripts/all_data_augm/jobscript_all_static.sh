#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_1_all_static
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

n_epochs=3000

davis_part=1
yv_part=0



# exp_simple_davis or exp1
augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
augm_datasets=['fss','ecssd','BIG_small','DUTS-TE','DUTS-TR','HRSOD_small','coco']

torchrun --nproc_per_node=2 --standalone train.py exp_name="davis-$davis_part-yv-$yv_part-all-static"\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path='/work3/s220493/saves/augmentations_static/' \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 +augmentations.augmentation_datasets=$augm_datasets
