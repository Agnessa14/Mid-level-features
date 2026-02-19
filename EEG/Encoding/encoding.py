#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENCODING - AVERAGE FEATURES BEFORE PCA - UNREAL ENGINE

This script implements the multivariate linear ridge regression for the
scene features from the Unreal Engine for a single frame.

@author: Alexander Lenders, Agnessa Karapetian
"""
from utils import (
    load_eeg,
    load_feature_set,
    load_alpha,
    OLS_pytorch,
    vectorized_correlation,
    load_config,
    parse_list,
)
import os
import numpy as np
import torch
import pickle
import argparse
from sklearn.metrics import r2_score


def encoding(
    sub: int,
    freq: int,
    region: str,
    input_type: str,
    feat_dir: str,
    save_dir: str,
    eeg_dir: str,
    frame: int,
    feature_names: list,
    exclude: bool,
    full_feat: bool = False,
    alpha_tp: bool = True,
):
    """
    Input:
    ----------
    I. Test, Training and Validation EEG data sets, which are already
    preprocessed + MVNN. The input are dictionaries, which include:
        a. EEG-Data (eeg_data, 5400 Images/Videos x 64 Channels x 70 Timepoints)
        b. Image/Video Categories (img_cat, 5400 x 1) - Each image/video has one specific ID
        c. Channel names (channels, 64 x 1 OR 19 x 1)
        d. Time (time, 70 x 1) - Downsampled timepoints of an image/video
        In case of the validation data set there are 900 images/videos instead of 5400.
    II. Image/Video features
        a. image_features or video_features.pkl: Canny edges, World normals, Lighting, Scene Depth,
        Reflectance, Action Identity, Skeleton Position after
        PCA (if necessary), saved in a dictionary "image_features" or "video_features"
            - Dictionary contains matrix for each feature with the dimension
            num_videos x num_components
        b. exp_variance_pca.pkl: Explained variance for each feature after PCA
        with n_components.
    III. Hyperparameter
    regression_features, dictionary with the following outputs for each feature:
        a. Root mean square error (RMSE) matrix (Timepoints x Alpha) - rmse_score
        b. Pearson correlation between true EEG data and predicted EEG data - correlation
        c. Best alpha for each timepoint based on RMSE - best_alpha_rmse
        d. Best alpha for each timepoint based on correlation - best_alpha_corr
        e. Best alpha averaged over timepoints based on RMSE - best_alpha_a_rmse
        f. Best alpha averaged over timepoints based on correlation - best_alpha_a_corr

    Returns:
    regression_features, dictionary with the following outputs for each feature:
    a. Root mean square error (RMSE) matrix (Timepoints x Channels) - rmse_score
    b. Pearson correlation between true EEG data and predicted EEG data (Timepoints x channels) - correlation
    ...

    Parameters
    ----------
    sub : int
        Subject number
    freq : int
          Downsampling frequency (default is 50)
    region : str
        The region for which the EEG data should be analyzed.
    input_type: str
        Miniclips or images
    feat_dir : str
        Directory where the features are stored.
    save_dir : str
        Directory where the results should be saved.
    eeg_dir : str
        Directory where the EEG data is stored.
    frame : int
        The frame number to be used for the analysis.
    feature_names : list
        List of feature names to be used in the analysis.
    exclude : bool
        If True, guitar trials will be excluded from the analysis.
    full_feat : bool
        If True, the full feature set will be used. If False, a reduced feature set
        will be used based on the length of feature_names.
    alpha_tp : bool
        If True, the alpha value will be loaded for each timepoint. If False, the
        alpha value will be loaded only once for the entire analysis.
    """
    if input_type == "images":
        if full_feat:
            featuresDir = os.path.join(
                feat_dir,
                f"img_features_frame_{frame}_redone_7_features_onehot.pkl",
            )
        else:
            featuresDir = os.path.join(
                feat_dir,
                f"img_features_frame_{frame}_redone_{len(feature_names)}_features_onehot.pkl",
            )
    elif input_type == "miniclips":
        if full_feat:
            featuresDir = os.path.join(
                feat_dir,
                f"video_features_avg_frame_redone_7.pkl",
            )
        else:
            featuresDir = os.path.join(
                feat_dir,
                f"video_features_avg_frame_redone_{len(feature_names)}.pkl",
            )

    # Device agnostic code: Use gpu if possible, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Number of channels
    if region == "wholebrain":
        n_channels = 64
    elif region == "posterior":
        n_channels = 19

    if input_type == "miniclips":
        y_train, timepoints = load_eeg(
            sub, "training", region, freq, input_type, eeg_dir=eeg_dir
        )

    elif input_type == "images":
        y_train, timepoints = load_eeg(
            sub, "train", region, freq, input_type, eeg_dir=eeg_dir
        )

    y_test, _ = load_eeg(
        sub, "test", region, freq, input_type, eeg_dir=eeg_dir
    )

    output_names = ("rmse_score", "correlation", "var_explained")

    # define matrix where to save the values
    regression_features = {
        (f"{', '.join(f)}" if isinstance(f, (tuple, list)) else str(f)): None
        for f in feature_names
    }

    if exclude:
        print(
            "Excluding guitar trials from the analysis (control analysis 9)."
        )

        X_train, _, X_test = load_feature_set("action", featuresDir)

        # Find all rows in X_train and X_test that contain a 1 in column 5
        guitar_trials_train = np.where(X_train[:, 5] == 1)[0]
        guitar_trials_test = np.where(X_test[:, 5] == 1)[0]

        # Remove these rows from the EEG data
        y_train = np.delete(y_train, guitar_trials_train, axis=0)
        y_test = np.delete(y_test, guitar_trials_test, axis=0)

    for feature in feature_names:
        X_train, _, X_test = load_feature_set(feature, featuresDir)

        if exclude:
            # Remove guitar trials from the feature set
            X_train = np.delete(X_train, guitar_trials_train, axis=0)
            X_test = np.delete(X_test, guitar_trials_test, axis=0)

        if alpha_tp is False:
            alpha = load_alpha(
                sub,
                freq,
                region,
                feature,
                input_type,
                feat_dir=save_dir,
                feat_len=len(feature_names),
            )

        output = dict.fromkeys(output_names)

        rmse = np.zeros((timepoints, n_channels))
        corr = np.zeros((timepoints, n_channels))
        var_explained = np.zeros((timepoints, n_channels))
        n_trials = y_test.shape[0]
        residuals = np.zeros((timepoints, n_trials, n_channels))

        for tp in range(timepoints):
            if alpha_tp is True:
                alpha = load_alpha(
                    sub,
                    freq,
                    region,
                    feature,
                    input_type,
                    feat_dir=save_dir,
                    tp=tp,
                    feat_len=len(feature_names),
                )

            y_train_tp = y_train[:, :, tp]
            y_test_tp = y_test[:, :, tp]
            regression = OLS_pytorch(alpha=alpha)
            try:
                regression.fit(X_train, y_train_tp, solver="cholesky")
            except Exception as error:
                regression.fit(X_train, y_train_tp, solver="solve")
            prediction = regression.predict(X_test)
            rmse_score = regression.score(entry=X_test, y=y_test_tp)
            correlation = vectorized_correlation(prediction, y_test_tp)
            rmse[tp, :] = rmse_score
            corr[tp, :] = correlation
            var_explained[tp, :] = r2_score(
                y_test_tp, prediction, multioutput="raw_values"
            )
            residuals[tp, :] = y_test_tp - prediction

        output["rmse_score"] = rmse
        output["correlation"] = corr
        output["var_explained"] = var_explained
        output["residuals"] = residuals
        output["y_true"] = y_test

        if isinstance(feature, list):
            regression_features[", ".join(feature)] = output
        else:
            regression_features[feature] = output

    # -------------------------------------------------------------------------
    # STEP 2.8 Save hyperparameters and scores
    # -------------------------------------------------------------------------
    # Save the dictionary
    saveDir = os.path.join(save_dir, f"{input_type}")

    fileDir = (
        str(sub)
        + "_seq_"
        + str(freq)
        + "hz_"
        + region
        + f"_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot"
        + ".pkl"
    )

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(regression_features, f)

    return regression_features


# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------

if __name__ == "__main__":

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
        "-f",
        "--freq",
        default=50,
        type=int,
        metavar="",
        help="downsampling frequency",
    )
    parser.add_argument(
        "-r",
        "--region",
        default="posterior",
        type=str,
        metavar="",
        help="Electrodes to be included, posterior (19) or wholebrain (64)",
    )
    parser.add_argument(
        "-i",
        "--input_type",
        default="images",
        type=str,
        metavar="",
        help="Font",
    )
    parser.add_argument(
        "--exclude_guitar_trials",
        action="store_true",
        help="Exclude guitar trials from the analysis.",
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)

    freq = args.freq
    region = args.region
    input_type = args.input_type
    exclude_guitar_trials = args.exclude_guitar_trials
    frame = config.getint(args.config, "img_frame")
    save_dir = config.get(args.config, "save_dir")
    feature_names = parse_list(config.get(args.config, "feature_names"))
    eeg_dir = config.get(args.config, "eeg_dir")

    # Hardcoded for now
    if args.config == "control_12":
        ALPHA_PER_TP = True
        print("Using alpha per timepoint for control_12")
    else:
        ALPHA_PER_TP = False

    # -------------------------------------------------------------------------
    # STEP 3 Run function
    # -------------------------------------------------------------------------
    if input_type == "miniclips":
        subjects = [
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
        feat_dir = config.get(args.config, "save_dir_feat_video")
    elif input_type == "images":
        subjects = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        feat_dir = config.get(args.config, "save_dir_feat_img")

    if args.config == "control_6_1" or args.config == "control_6_2":
        for sub in subjects:
            result = encoding(
                sub,
                freq,
                region,
                input_type,
                feat_dir=feat_dir,
                save_dir=save_dir,
                eeg_dir=eeg_dir,
                frame=frame,
                feature_names=feature_names,
                exclude=exclude_guitar_trials,
                full_feat=True,
                alpha_tp=ALPHA_PER_TP,
            )
    else:
        for sub in subjects:
            result = encoding(
                sub,
                freq,
                region,
                input_type,
                feat_dir=feat_dir,
                save_dir=save_dir,
                eeg_dir=eeg_dir,
                frame=frame,
                feature_names=feature_names,
                exclude=exclude_guitar_trials,
                alpha_tp=ALPHA_PER_TP,
            )
