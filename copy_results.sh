#!/bin/bash
#SBATCH --job-name=copy_results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --qos=standard
#SBATCH --partition=main
#SBATCH --time=04:00:00

cp -r /scratch/alexandel91/mid_level_features/results/EEG /scratch/alexandel91/mid_level_features/results_mvnn_epochs/

