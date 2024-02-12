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
	$(PYTHON_INTERPRETER) eval_davis.py --davis_path '/work3/s220493/DAVIS/2017' --output outputs
