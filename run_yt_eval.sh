#! /usr/bin/bash


source /work3/s220493/venv/bin/activate


## Saved model relevant
base_path="/work3/s220493/saves/"
augm_type_dir="hparams_rerun/p_augm/" 
file_part="checkpoint_davis-augm_p-075-_"

STCN_path="/zhome/39/c/174709/git/STCN_mine"
davis2017_path="/zhome/39/c/174709/git/davis2017-evaluation"


## Output files relevant
output_path="outputs/ytvos"

cleanup_names=("bike-packing" "blackswan" "bmx-trees" "breakdance" "camel" "car-roundabout" "car-shadow" "cows" "dance-twirl" "dog" "dogs-jump" "drift-chicane" "drift-straight" "goat" "gold-fish" "horsejump-high" "india" "judo" "kite-surf" "lab-coat" "libby" "loading" "mbike-trick" "motocross-jump" "paragliding-launch" "parkour" "pigs" "scooter-black" "shooting" "soapbox")

saved_model="${base_path}${augm_type_dir}${file_part}7500.pth"

if ! [ -f "${saved_model}" ]; then
    echo "${saved_model} does not exist. Skipping."
    exit 1
fi

echo ${saved_model}

cd ${STCN_path}
python eval_youtube.py --yv_path '/work3/s220493/YouTube/' --output ${output_path} --model ${saved_model}

    # cd ${davis2017_path}
    # python evaluation_method.py --task semi-supervised --results_path ~/git/STCN_mine/${output_path}${i} --davis_path /work3/s220493/DAVIS/2017/trainval

    

