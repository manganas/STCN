#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J generatedDatasetMistral7k
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

# #BSUB -R "select[gpu32gb]"
n_epochs=7000

batch_size=8

# to select v100 with 32gb BSUB -R "select[gpu32gb]"

davis_part=1
yv_part=0

save_model_path="/work3/s220493/saves/generated/sdxl-turbo/mistal_nltk_simple_pipeline/"
exp_name="davis-$davis_part-generated-mistral_nltk_simple_7k"
# load_network="/work3/s220493/saves/STCN_stage0.pth"
load_model="${save_model_path}checkpoint_${exp_name}_checkpoint.pth"


augmentations=exp_multi_data
davis_root="/work3/s220493/DAVIS"
augm_datasets=['/work3/s220493/Generated_datasets/mistal_nltk_simple_pipeline/']

torchrun --nproc_per_node=1 --standalone train.py exp_name=$exp_name\
 n_epochs=$n_epochs\
 davis_root=$davis_root \
 save_model_path=$save_model_path \
 load_model=$load_model \
 davis_part=$davis_part \
 yt_vos_part=$yv_part \
 augmentations.augmentation_datasets=$augm_datasets
