 #!/bin/sh
 #BSUB -q gpua100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 4
 #BSUB -W 48:00
 #BSUB -R "span[hosts=1]"
 #BSUB -R "rusage[mem=39GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 echo "Running script..."

nvidia-smi
module swap python3/3.9.11
module swap cuda/11.6

source ../venv/bin/activate

#0, 0.25, 0.5, 0.75
augm_p=0
torchrun --nproc_per_node=1 train.py  exp_name="DAVIS_augmentation_p_$augm_p" augmentations.augmentation_p=$augm_p