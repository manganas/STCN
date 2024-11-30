#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J davis1Yv007_Augm
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

# #BSUB -R "select[gpu32gb]"
source /work3/s220493/venv/bin/activate

n_epochs=3000

batch_size=8

# to select v100 with 32gb BSUB -R "select[gpu32gb]"

davis_part=1
yv_part=0.07

save_model_path="/work3/s220493/saves/long_rerun_l/"
exp_name="davis-${davis_part}-yt-${yv_part}-augm-"
# load_network="/work3/s220493/saves/STCN_stage0.pth"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"


augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
augm_datasets=['davis']


echo $augm_datasets

torchrun --nproc_per_node=2 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path=$save_model_path \
 load_model=$load_model \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 augmentations.augmentation_datasets=$augm_datasets \
