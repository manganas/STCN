#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis_augmented_disk
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."

nvidia-smi
module swap python3/3.9.11
module swap cuda/11.6

source ../venv/bin/activate

n_epochs=7000
# '' or '_Augmented'
davis_dataset='_Augmented'

# exp_simple_davis or exp1
augmentations=exp_simple_davis
davis_root="/work3/s220493/DAVIS$davis_dataset"
torchrun --nproc_per_node=2 --standalone train.py exp_name="DAVIS_$davis_dataset-$augmentations-$n_epochs"\
 n_epochs=$n_epochs augmentations=$augmentations\
 davis_root=$davis_root \
 save_model_path='/work3/s220493/saves/mean_exps/'
