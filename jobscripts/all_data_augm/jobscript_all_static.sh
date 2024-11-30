#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J allStatic7k_agn
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."

nvidia-smi
module swap python3/3.9.11
module swap cuda/11.6

source /work3/s220493/venv/bin/activate

# #BSUB -R "select[gpu32gb]"
n_epochs=7000
batch_size=8


davis_part=1
yv_part=0


save_model_path="/work3/s220493/saves/rerun_all_static_7k/"
exp_name="davis-${davis_part}-all_static_7k"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"



augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
augm_datasets=['fss','ecssd','BIG_small','DUTS-TE','DUTS-TR','HRSOD_small','coco']

torchrun --nproc_per_node=2 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path=$save_model_path \
 load_model=$load_model \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 augmentations.augmentation_datasets=$augm_datasets \
 augmentations.dataset_probabilities=null
