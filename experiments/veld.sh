#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
model_name=${2:-"Qwen/Qwen3-30B-A3B"}
seed=${3:-42}
compression_ratio=${4:-0.50}
dataset_name=${5:-"theblackcat102/evol-codealpaca-v1"}
veld_lambda=${6:-1e-4}
run_lm_eval=${7:-true}
run_evalplus=${8:-true}
run_livecodebench=${9:-true}
run_math=${10:-false}
run_wildbench=${11:-false}
singleton_super_experts=${12:-"false"}
singleton_outlier_experts=${13:-"false"}
num_samples=1024
output_file_name="observations_${num_samples}_cosine-seed_${seed}.pt"

server_log_file_name="veld-${FIRST_DEVICE}.log"

echo "Running VELD pruning: model=$model_name devices=$CUDA_VISIBLE_DEVICES compression=$compression_ratio lambda=$veld_lambda"

python src/reap/prune.py \
    --model-name $model_name \
    --dataset-name $dataset_name \
    --compression-ratio $compression_ratio \
    --prune-method veld \
    --veld_lambda $veld_lambda \
    --profile false \
    --vllm_port $port \
    --server-log-file-name $server_log_file_name \
    --do-eval false \
    --distance_measure cosine \
    --seed $seed \
    --output_file_name ${output_file_name} \
    --singleton_super_experts ${singleton_super_experts} \
    --singleton_outlier_experts ${singleton_outlier_experts} \
    --samples_per_category ${num_samples}

short_model_name=$(echo $model_name | cut -d'/' -f2)
short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)

pruned_model_dir_name="veld"
if [[ "${singleton_super_experts}" == "true" ]]; then
    pruned_model_dir_name="${pruned_model_dir_name}-perserve_super"
elif [[ "${singleton_outlier_experts}" == "true" ]]; then
    pruned_model_dir_name="${pruned_model_dir_name}-perserve_outlier"
fi
pruned_model_dir_name="${pruned_model_dir_name}-seed_${seed}-${compression_ratio}"

model_dir="artifacts/${short_model_name}/${short_dataset_name}/pruned_models/${pruned_model_dir_name}"

echo "evaluating model: ${model_dir}"
bash experiments/eval.sh \
    $model_dir \
    $seed \
    $port \
    $server_log_file_name \
    ${run_lm_eval} \
    ${run_evalplus} \
    ${run_livecodebench} \
    ${run_math} \
    ${run_wildbench}
echo "Finished evaluating model: ${model_dir}"
