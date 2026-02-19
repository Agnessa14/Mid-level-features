#!/bin/bash
# Run control analysis 9 for miniclips (excluding guitar trials)

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
    
python ../../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ../config.ini \
    --config control_9 \
    --input_type "images" \
    --exclude_guitar_trials

# Fourth step: Encoding
python ../../EEG/Encoding/encoding.py \
    --config_dir ../config.ini \
    --config control_9 \
    --input_type "images" \
    --exclude_guitar_trials

# Fifth step: Bootstrapping
python ../../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_9 \
    --input_type "images"

# Sixth step: Stats
python ../../EEG/Stats/encoding_significance_stats.py \
    --config_dir ../config.ini \
    --config control_9 \
    --input_type "images"

# Seventh step: Plotting 
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_9 \
    --input_type "images"