#!/bin/bash
# Control analysis 8

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# python ./time_generalization.py \
#     --config_dir ../config.ini \
#     --config default \
#     --input_type "miniclips"

# echo "Time gen done for miniclips"

# python ./time_gen_stats.py \
#     --config_dir ../config.ini \
#     --config default \
#     --input_type "miniclips"

echo "Time gen stats done for miniclips"

python ./time_gen_plotting.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips"

# python ./time_generalization.py \
#     --config_dir ../config.ini \
#     --config default \
#     --input_type "images"

# echo "Time gen done for images"

# python ./time_gen_stats.py \
#     --config_dir ../config.ini \
#     --config default \
#     --input_type "images"

echo "Time gen stats done for images"

python ./time_gen_plotting.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"

# python ./time_gen_stats_diff.py \
#     --config_dir ../config.ini \
#     --config default

python ./time_gen_diff_plotting.py \
    --config_dir ../config.ini \
    --config default
