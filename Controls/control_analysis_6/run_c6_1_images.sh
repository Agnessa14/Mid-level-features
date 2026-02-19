#!/bin/bash

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

echo "Hyperparam tuning"
python ../../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images" 

echo "Encoding"
python ../../EEG/Encoding/encoding.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images"

echo "Variance partitioning"
python ./control_analysis_6.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images" \
    --idea 1
    
echo "Encoding with significance stats"
python ../../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images"

python ../../EEG/Stats/encoding_significance_stats.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images"

python ./control_analysis_6_plotting.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images" \
    --legend

python ./control_analysis_6_plotting.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images" \
    --legend \
    --zoom_in