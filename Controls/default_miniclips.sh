#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

# First step: MVNN
python ../EEG/Encoding/mvnn_encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"

python ../EEG/Encoding/annotation_prep_videos.py \
    --config_dir ./config.ini \
    --config default 
    
# Third step: Hyperparameter optimization
python ../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips" 

# Fourth step: Encoding
python ../EEG/Encoding/encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"

# Fifth step: Bootstrapping
python ../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"

# Sixth step: Stats
python ../EEG/Stats/encoding_significance_stats.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"

# Seventh step: Plotting 
python ../EEG/Plotting/plot_encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"