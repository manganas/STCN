#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = STCN
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Activate virtual environment
.PHONY: environment
environment:
	source ~/git/venv/bin/activate
## Install Python Dependencies
.PHONY: requirements
requirements: environment
	$(PYTHON_INTERPRETER) -m pip install torch torchvision
	$(PYTHON_INTERPRETER) -m pip install pillow-simd
	$(PYTHON_INTERPRETER) -m pip install progressbar2 opencv-python gitpython gdown git+https://github.com/cheind/py-thin-plate-spline

## Set up python interpreter environment
.PHONY: eval_davis
eval_davis:
	$(PYTHON_INTERPRETER) eval_davis.py --davis_path '/work3/s220493/DAVIS/2017' --output outputs/outputs_davis

## $(PYTHON_INTERPRETER) train.py --davis_path '/work3/s220493/DAVIS' --output outputs --static_root /work3/s220493/static
.PHONY: train_static
train_static:
	CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 $(PYTHON_INTERPRETER) -m torch.distributed.launch --nproc_per_node=1 train.py --stage 0 --static_root /work3/s220493/static --exp_name 'static' --save_model_path './saves'
	
.PHONY: train_davis
train_davis:
	CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train.py --stage 2 --davis_root '/work3/s220493/DAVIS' 
	
	
.PHONY: train_davis_augmented
train_davis_augmented:
	CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train.py --stage 2 --davis_root '/work3/s220493/DAVIS_Augmented' --exp_name davis_augmented --iterations 600000
	
#--load_model './saves/checkpoint_static_checkpoint.pth'
	
