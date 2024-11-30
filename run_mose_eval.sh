#! /usr/bin/bash


source /work3/s220493/venv/bin/activate


## Saved model relevant
base_path="/work3/s220493/saves/"
augm_type_dir="hparams_rerun/p_augm/" 
file_part="checkpoint_davis-augm_p-075-_"

STCN_path="/zhome/39/c/174709/git/STCN_mine"
evaluation_path="/zhome/39/c/174709/git/stcn_scores_multidatasets"



## Output files relevant
output_path="outputs/mose"

        
saved_model="${base_path}${augm_type_dir}${file_part}7500.pth"

if ! [ -f "${saved_model}" ]; then
    echo "${saved_model} does not exist. Skipping."
    exit 1
fi

echo ${saved_model}

# cd ${STCN_path}
# python eval_mose.py --mose_path '/work3/s220493/MOSE/' --output ${output_path} --model ${saved_model}

cd ${evaluation_path}
python evaluation_method.py --task semi-supervised --results_path ~/git/STCN_mine/${output_path} --dataset_path /work3/s220493/MOSE/ --dataset_name mose --set valid


cd ${STCN_path}

