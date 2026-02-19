#!/bin/bash
# After running decoding scripts for images and miniclips

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

echo "Decoding bootstrapping images..."
python ../EEG/Stats/decoding_bootstrapping.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

echo "Decoding bootstrapping videos..."
python ../EEG/Stats/decoding_bootstrapping.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"

echo "Decoding significance images..."
python ../EEG/Stats/decoding_significance_stats.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images"

echo "Decoding significance videos..."
python ../EEG/Stats/decoding_significance_stats.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "miniclips"

echo "Decoding bootstrapping difference..."
python ../EEG/Stats/decoding_difference_bootstrapping.py \
    --config_dir ./config.ini \
    --config default

echo "Decoding significance difference..."
python ../EEG/Stats/decoding_difference_significance_stats.py \
    --config_dir ./config.ini \
    --config default

echo "Plotting decoding results..."
python ../EEG/Plotting/plot_decoding.py \
    --config_dir ./config.ini \
    --config default