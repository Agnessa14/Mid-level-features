#!/bin/bash
source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# First step: MVNN
python ../EEG/Encoding/mvnn_encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

# Second step: Preprocess the features of the first frame in each video
python ../EEG/Encoding/annotation_prep_images.py \
    --config_dir ./config.ini \
    --config default 
    
# Third step: Hyperparameter optimization
python ../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

# Fourth step: Encoding
python ../EEG/Encoding/encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

# Fifth step: Bootstrapping
echo "Bootstrapping..."
python ../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

# Sixth step: Stats
echo "Running significance stats..."
python ../EEG/Stats/encoding_significance_stats.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

# Seventh step: Plotting 
echo "Plotting results..."
python ../EEG/Plotting/plot_encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images" \
    --legend