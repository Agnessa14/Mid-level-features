#!/bin/bash
# Control analysis 2

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# First step: MVNN
echo "MVNN Encoding Analysis for Miniclips"
python ../../EEG/Encoding/mvnn_encoding.py \
    --config_dir ../config.ini \
    --config control_2 \
    --input_type "miniclips"

echo "MVNN Encoding Analysis for Miniclips completed"
echo "Preprocessing features for miniclips"
# Second step: Preprocess the features of the first frame in each video
python ../../EEG/Encoding/annotation_prep_videos.py \
    --config_dir ../config.ini \
    --config control_2 
    
echo "Preprocessing features for miniclips completed"
echo "Starting hyperparam optimization for miniclips"
# Third step: Hyperparameter optimization
python ../../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ../config.ini \
    --config control_2 \
    --input_type "miniclips" 

echo "Hyperparameter optimization for miniclips completed"
echo "Starting encoding analysis for miniclips"
# Fourth step: Encoding
python ../../EEG/Encoding/encoding.py \
    --config_dir ../config.ini \
    --config control_2 \
    --input_type "miniclips"

echo "Encoding analysis for miniclips completed"
echo "Starting bootstrapping for miniclips"
# Fifth step: Bootstrapping
python ../../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_2 \
    --input_type "miniclips"

echo "Bootstrapping for miniclips completed"
echo "Starting stats for miniclips"
# Sixth step: Stats
python ../../EEG/Stats/encoding_significance_stats.py \
    --config_dir ../config.ini \
    --config control_2 \
    --input_type "miniclips"

# echo "Stats for miniclips completed"
echo "Starting plotting for miniclips"
# Seventh step: Plotting 
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_2 \
    --input_type "miniclips" \
    --legend