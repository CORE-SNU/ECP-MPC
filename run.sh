#!/bin/bash

datasets=("zara1" "zara2" "hotel" "univ" "eth")

declare -A models
models["zara1"]="zara1_pretrained"
models["zara2"]="zara2_pretrained"
models["hotel"]="hotel_pretrained"
models["univ"]="univ_pretrained"
models["eth"]="eth_pretrained"


controllers=("cp-mpc" "eacp-mpc" "cc")
for dataset in "${datasets[@]}"; do
    model="${models[$dataset]}"
    for controller in "${controllers[@]}"; do
        echo "Evaluating dataset: $dataset using model: $model and controller: $controller"
        python evaluate_controller_prev.py \
            --dataset "$dataset" \
            --model "./models/$model" \
            --checkpoint 100 \
            --data "../processed/${dataset}_test.pkl" \
            --output_path "${dataset}_${controller}.csv" \
            --controller "$controller" \
            # --visualize
    done
done
