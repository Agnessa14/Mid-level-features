#!/bin/bash
# Control analysis 3: Encoding analysis with image annotations for miniclips EEG data

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

echo "Starting hyperparameter optimization for miniclips..."
# First step: Hyperparameter optimization
python ./hyperparameter_optimization_c3.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips" 
echo "Hyperparameter optimization completed."

# Wait for the hyperparameter optimization to finish
echo "Starting encoding analysis for miniclips..."
# Second step: Encoding
python ./encoding_c3.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"
echo "Encoding analysis completed for miniclips."

# Wait for the encoding to finish
echo "Stating bootstrapping and stats for miniclips..."
# Third step: Bootstrapping
python ../../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"

# Fourth step: Stats
python ../../EEG/Stats/encoding_significance_stats.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"

# Fifth step: Plotting 
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"
