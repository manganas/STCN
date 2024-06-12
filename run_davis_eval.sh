#! /usr/bin/bash


source /work3/s220493/venv/bin/activate


## Saved model relevant
base_path="/work3/s220493/saves/"
augm_type_dir="generated/sdxl-turbo/mistral/"
file_part="checkpoint_davis-1-generated-mistral_"

STCN_path="/zhome/39/c/174709/git/STCN_mine"
davis2017_path="/zhome/39/c/174709/git/davis2017-evaluation"


## Output files relevant
output_path="outputs/generated/sdxl_turbo_mistral/all_clases"

cleanup_names=("bike-packing" "blackswan" "bmx-trees" "breakdance" "camel" "car-roundabout" "car-shadow" "cows" "dance-twirl" "dog" "dogs-jump" "drift-chicane" "drift-straight" "goat" "gold-fish" "horsejump-high" "india" "judo" "kite-surf" "lab-coat" "libby" "loading" "mbike-trick" "motocross-jump" "paragliding-launch" "parkour" "pigs" "scooter-black" "shooting" "soapbox")

for i in {2000..9007}
do
        
    saved_model="${base_path}${augm_type_dir}${file_part}${i}.pth"

    if ! [ -f "${saved_model}" ]; then
        # echo "${saved_model} does not exist. Skipping."
        continue
    fi

    echo ${saved_model}

    cd ${STCN_path}
    python eval_davis.py --davis_path '/work3/s220493/DAVIS/2017' --output ${output_path}${i} --model ${saved_model}

    cd ${davis2017_path}
    python evaluation_method.py --task semi-supervised --results_path ~/git/STCN_mine/${output_path}${i} --davis_path /work3/s220493/DAVIS/2017/trainval

    cd ${STCN_path}
    for to_del in "${cleanup_names[@]}"; do
        rm -rf "${output_path}${i}/${to_del}"
    done

done