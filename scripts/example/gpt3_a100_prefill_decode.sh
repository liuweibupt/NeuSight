#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)"

python3 ../pred.py \
    --predictor_name neusight \
    --predictor_path ../asplos/data/predictor/MLP_WAVE \
    --device_config_path ../asplos/data/device_configs/NVIDIA_A100-PCIE-40GB.json \
    --model_config_path ../asplos/data/DLmodel_configs/gpt3_175b_a100.json \
    --sequence_length 128 \
    --batch_size 1 \
    --execution_type prefill \
    --tile_dataset_dir ../asplos/data/dataset/train \
    --result_dir ../asplos/results

python3 ../pred.py \
    --predictor_name neusight \
    --predictor_path ../asplos/data/predictor/MLP_WAVE \
    --device_config_path ../asplos/data/device_configs/NVIDIA_A100-PCIE-40GB.json \
    --model_config_path ../asplos/data/DLmodel_configs/gpt3_175b_a100.json \
    --sequence_length 2048 \
    --batch_size 1 \
    --execution_type decode \
    --tile_dataset_dir ../asplos/data/dataset/train \
    --result_dir ../asplos/results
