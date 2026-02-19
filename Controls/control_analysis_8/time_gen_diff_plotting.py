#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIME GENERALIZATION ANALYSIS - CONTROL ANALYSIS 8

@author: Alexander Lenders
"""
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)

# This allows to save the plot as .svg graphic with high res of the font:
plt.rcParams["svg.fonttype"] = "none"

parser = argparse.ArgumentParser()

# add arguments / inputs
parser.add_argument(
    "--config_dir",
    type=str,
    help="Directory to the configuration file.",
    required=True,
)
parser.add_argument(
    "--config",
    type=str,
    help="Configuration.",
    required=True,
)
parser.add_argument(
    "-f", "--font", default="Arial", type=str, metavar="", help="Font"
)
parser.add_argument(
    "-i",
    "--input_type",
    default="images",
    type=str,
    metavar="",
    help="images, miniclips or difference",
)

args = parser.parse_args()  # to get values for the arguments

config = load_config(args.config_dir, args.config)
workDir = config.get(args.config, "save_dir")
feature_names = parse_list(config.get(args.config, "feature_names"))
font = args.font

list_sub_vid = [
    6,
    7,
    8,
    9,
    10,
    11,
    17,
    18,
    20,
    21,
    23,
    25,
    27,
    28,
    29,
    30,
    31,
    32,
    34,
    36,
]
list_sub_img = [
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
]


# -------------------------------------------------------------------------
# Load results
# -------------------------------------------------------------------------
workDir_img = os.path.join(workDir, "images")
workDir_vid = os.path.join(workDir, "miniclips")
saveDir = os.path.join(workDir, "plots")

if os.path.exists(saveDir) is False:
    os.makedirs(saveDir)

n_sub_vid = len(list_sub_vid)
n_sub_img = len(list_sub_img)
timepoints = 70  # number of timepoints in the time generalization matrix

identifierDir = f"seq_50hz_posteriortime_gen_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

results = []  # list of dictionaries

stats_dir = os.path.join(workDir, "difference", "stats")
id_stats = "time_gen_diff_stats_both.pkl"
time_dir = os.path.join(stats_dir, id_stats)

with open(time_dir, "rb") as file:
    time_stats = pickle.load(file)

results_videos = {}
results_images = {}
for subject in list_sub_vid:
    fileDir_vid = os.path.join(workDir_vid, f"{subject}_{identifierDir}")
    encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)
    results_videos[str(subject)] = encoding_results_vid

for subject in list_sub_img:
    fileDir_img = os.path.join(workDir_img, f"{subject}_{identifierDir}")
    encoding_results_img = np.load(fileDir_img, allow_pickle=True)
    results_images[str(subject)] = encoding_results_img


for feature in feature_names:
    # -----------------------------------------------------------------------------
    # Create time generalization plot with seaborn (correlation)
    # -----------------------------------------------------------------------------
    results_vid = np.zeros((n_sub_vid, timepoints, timepoints))
    results_img = np.zeros((n_sub_img, timepoints, timepoints))

    for index, subject in enumerate(list_sub_vid):
        subject_result = results_videos[str(subject)][feature]["correlation"]

        # averaged over all channels
        subject_result_averaged = np.mean(subject_result, axis=2)
        results_vid[index, :, :] = subject_result_averaged

    for index, subject in enumerate(list_sub_img):
        subject_result = results_images[str(subject)][feature]["correlation"]

        # averaged over all channels
        subject_result_averaged = np.mean(subject_result, axis=2)
        results_img[index, :, :] = subject_result_averaged

    mean_time_gen = np.mean(results_img, axis=0) - np.mean(results_vid, axis=0)
    mean_time_gen = np.flip(mean_time_gen, axis=0)
    reduced_time_gen = mean_time_gen[0:60, 10:70]

    ticks = np.arange(0.5, 0.72, 0.02)
    tick_labels = [
        int(tick * 100) for tick in ticks
    ]  # Convert ticks to integers

    fig, ax = plt.subplots()

    heatmap = ax.imshow(
        reduced_time_gen,
        cmap="jet",
        vmin=-0.12,
        vmax=0.12,
        extent=[0, reduced_time_gen.shape[1], 60, 0],
    )

    # Add colorbar
    cbar = plt.colorbar(
        heatmap, ax=ax, pad=0.02
    )  # Adjust the pad value as needed
    cbar.ax.set_ylabel(
        "Pearson's r", fontdict={"family": font, "size": 13}, labelpad=10
    )

    plt.xlabel("Testing Time (ms)", fontdict={"family": font, "size": 13})
    plt.ylabel("Training Time (ms)", fontdict={"family": font, "size": 13})

    plt.yticks(
        ticks=[60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
        labels=[-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        fontdict={"family": font, "size": 11},
    )

    plt.xticks(
        ticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        labels=[-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        fontdict={"family": font, "size": 11},
        rotation=45,
    )

    plt.axvline(
        x=10, color="lightgrey", linestyle="solid", linewidth=2, alpha=1
    )
    plt.axhline(
        y=50, color="lightgrey", linestyle="solid", linewidth=2, alpha=1
    )

    ax.tick_params(axis="both", direction="inout")

    cbar.ax.tick_params(axis="both", direction="inout")

    ax.tick_params(
        axis="x", which="both", length=6, width=2
    )  # Adjust the length and width as needed
    ax.tick_params(
        axis="y", which="both", length=6, width=2
    )  # Adjust the length and width as needed

    plt.tight_layout()

    plt.show()

    plotDir_svg = os.path.join(
        saveDir, f"time_gen_diff_{feature}_correlation.svg"
    )
    plotDir = os.path.join(saveDir, f"time_gen_diff_{feature}_correlation.png")

    plt.savefig(plotDir_svg, format="svg", dpi=300, transparent=True)
    plt.savefig(plotDir, format="png", dpi=300, transparent=True)

    plt.close()

    # -----------------------------------------------------------------------------
    # STEP 4: Create time generalization plot with seaborn (stats) - Fig 3D
    # -----------------------------------------------------------------------------
    bool_time = time_stats[feature]["Boolean_statistical_map"]
    bool_time = np.flip(bool_time, axis=0)
    bool_time_reduced = bool_time[0:60, 10:70]

    fig, ax = plt.subplots()

    heatmap = ax.imshow(
        bool_time_reduced,
        cmap="jet",
        vmin=0,
        vmax=2,
        extent=[0, bool_time_reduced.shape[1], 60, 0],
    )

    plt.xlabel("Testing Time (ms)", fontdict={"family": "Arial", "size": 13})
    plt.ylabel("Training Time (ms)", fontdict={"family": "Arial", "size": 13})

    plt.yticks(
        ticks=[60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
        labels=[-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        fontdict={"family": "Arial", "size": 11},
    )

    plt.xticks(
        ticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        labels=[-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        fontdict={"family": "Arial", "size": 11},
        rotation=45,
    )

    # Create custom patches for the legend
    significant_patch = mpatches.Patch(
        color="greenyellow", label="Significant"
    )
    non_significant_patch = mpatches.Patch(
        color="darkblue", label="Not Significant"
    )

    # Add the legend
    legend_font_props = {"family": font, "size": 12}
    plt.legend(
        handles=[significant_patch, non_significant_patch],
        loc="lower right",
        prop=legend_font_props,
    )

    plt.axvline(
        x=10, color="lightgrey", linestyle="solid", linewidth=2, alpha=1
    )
    plt.axhline(
        y=50, color="lightgrey", linestyle="solid", linewidth=2, alpha=1
    )

    ax.tick_params(axis="both", direction="inout")
    ax.tick_params(
        axis="x", which="both", length=6, width=2
    )  # Adjust the length and width as needed
    ax.tick_params(
        axis="y", which="both", length=6, width=2
    )  # Adjust the length and width as needed

    plt.tight_layout()
    plt.show()

    plotDir_svg = os.path.join(saveDir, f"time_gen_diff_{feature}_stats.svg")
    plotDir = os.path.join(saveDir, f"time_gen_diff_{feature}_stats.png")
    plt.savefig(plotDir_svg, format="svg", dpi=300, transparent=True)
    plt.savefig(plotDir, format="png", dpi=300, transparent=True)
    plt.close()
