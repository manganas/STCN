#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J p_augm_075
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

# # BSUB -R "select[gpu32gb]"
source /work3/s220493/venv/bin/activate

n_epochs=2500

batch_size=8

# to select v100 with 32gb BSUB -R "select[gpu32gb]"

davis_part=1

augmentation_p=[0.75]
augmentation_p_txt=075

save_model_path="/work3/s220493/saves/hparams_rerun/p_augm/"
exp_name="davis-augm_p-${augmentation_p_txt}-"
# load_network="/work3/s220493/saves/STCN_stage0.pth"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"


augmentations=rerun
davis_root="/work3/s220493/DAVIS"

torchrun --nproc_per_node=1 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path=$save_model_path \
 load_model=$load_model \
 ++augmentations.augmentation_p=$augmentation_p \

