#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DECODING
This script conducts a decoding analysis for a single subject on the videos.
This version does the decoding only on the test data of the encoding analysis.
The MVNN transformer is based on the MVNN preprocessing script. No standard
scaler is used. The removal of the mean pattern (cocktail-blank removal) is not
implemented in the script, see Guggenmos et al. (2018) for discussion.

@author: Alexander Lenders, Agnessa Karapetian
"""
import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import LedoitWolf
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
import scipy
from utils import load_eeg
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from EEG.Encoding.utils import load_config


# -----------------------------------------------------------------------------
# STEP 1: Define Decoding Function
# -----------------------------------------------------------------------------
def decoding_single_subject_func(
    sub: int,
    mvnn_dim: str,
    freq: int,
    region: str,
    eeg_dir: str,
    save_dir: str,
    input_type: str,
    it: int,
    action_dir: str = None,
):
    """
    Preprocessing (MVNN, standardization) is fitted on the training data, as
    recommended.
    The C hyperparameter for SVM is set to default-value 1, as common in
    neuroscientific literature. A 6-fold stratified CV is used.

    Input:
    ----------
    Test data set for the encoding models, which contains preprocessed EEG data
    The input is a dictionary which includes:
    a. EEG-Data (eeg_data, 5400 Videos/Images x 64 Channels x 70 Timepoints)
    b. Video/Image Categories (img_cat, 5400 x 1) - Each video/image has one specific ID
    c. Channel names (channels, 64 x 1 OR 19 x 1)
    d. Time (time, 70 x 1) - Downsampled timepoints of a video

    Returns:
    ----------

    Decoding results, saved in a dictionary which contains:
    1. final_results_mean (70 Timepoints x 16110 Pairwise Decoding between Videos/Images):
        - Contains the pairwise decoding results (RDM) for each timepoint
    2. mean_accuracies_over_conditions (70 Timepoints x 1)
        - Contains the pairwise decoding results for each timepoint averaged over
        conditions

    Parameters
    ----------
    sub : int
          Subject number
    mvnn_dim : str
          Whether to compute the mvnn covariace matrices
          for each time point["time"] or for each epoch["epochs"]
          (default is "epochs").
    freq : int
          Downsampling frequency (default is 50)
    region : str
        The region for which the EEG data should be analyzed.
    eeg_dir: str
        Directory to the EEG data.
    save_dir: str
        Directory to save the results.
    input_type: str
        Performing analysis on images (default) or miniclips
    it: int
        Iteration (for parallelizing the computations on remote server and setting the random seed)
    action_dir: str, optional
        Directory to the action data, which is used to exclude guitar trials from the analysis.
        If None, no guitar trials are excluded (default is None).
    -------

    """
    # -------------------------------------------------------------------------
    # STEP 1.1 Define Variables, Define Transformer
    # -------------------------------------------------------------------------
    # Define the number of channels:
    if region == "posterior":
        n_chan = 19
    if region == "wholebrain":
        n_chan = 64

    class MVNN_Transformer(BaseEstimator, TransformerMixin):
        def __init__(
            self, F, n_chan=n_chan, timepoints=freq, mvnn_dim=mvnn_dim
        ):
            self.n_chan = n_chan
            self.timepoints = timepoints
            self.mvnn_dim = mvnn_dim
            self.F = F

        def fit(self, X=None, y=None):
            # Covariance Function
            _cov = lambda z: LedoitWolf().fit(z).covariance_

            covariance_matrix = np.zeros(
                (self.n_chan, self.n_chan), dtype=float
            )
            self.F = self.F.transpose(1, 2, 0)

            # ******************************************************************
            # THIS IS THE CORRECT EPOCHS METHOD acc. to Guggenmos et al. (2018) 
            # ******************************************************************
            if (
                self.mvnn_dim == "epochs"
            ):  # if computing covariance matrices for each time point
                # Computing sigma for each time point, then averaging across time
                covariance_matrix = np.mean(
                    [_cov(self.F[:, :, t]) for t in range(self.timepoints)],
                    axis=0,
                )

            # Compute the inverse matrix (whitening matrix)
            """
            This code is using the scipy.linalg.fractional_matrix_power function 
            from the SciPy library to compute the inverse of a square matrix sigma.
            For more information, see Guggenmos et al. (2018), equation (1), p. 436
            """
            covariance_matrix_inv = scipy.linalg.fractional_matrix_power(
                covariance_matrix, -0.5
            )

            self.covariance_matrix_inv_real_ = np.real(covariance_matrix_inv)

            return self

        def transform(self, X, y=None):
            X = X @ self.covariance_matrix_inv_real_
            return X

    # -------------------------------------------------------------------------
    # STEP 1.2 Define Variables and Directory
    # -------------------------------------------------------------------------
    # In this version, the decoding analysis is only done on the test data:
    img_type = "test"

    # Define random seed (for reproduction):
    rng = np.random.default_rng(it)

    # Print a summary of the arguments:
    print(
        "\n\n\n>>> Mvnn %s, %dhz, sub %s, brain region: %s <<<"
        % (mvnn_dim, freq, sub, region)
    )
    # -------------------------------------------------------------------------
    # STEP 1.3 Load the EEG Data + Define Futher Variables
    # -------------------------------------------------------------------------
    eeg_data, img_cat, time = load_eeg(
        sub, img_type, region, freq, input_type, eeg_dir
    )

    # if action_dir:
    #     print(
    #         "Excluding guitar trials from the analysis (control analysis 9)."
    #     )

    #     action_data = pd.read_csv(action_dir, header=None)

    #     # Action indices
    #     indices = action_data.index[action_data[1] == 30].tolist()

    #     # Create a set of indices to exclude
    #     exclude_indices = set(indices)

    #     # Filter out exclude_indices from img_cat
    #     img_cat = np.array([cat for i, cat in enumerate(img_cat) if i not in exclude_indices])

    # Check if there are NA's within the EEG data
    num_nan = np.isnan(eeg_data).sum()
    print(f"There are {num_nan} NAN values in the input data")

    # Number of repetitions for the test data:
    max_rep = 30

    # Number of pseudo-trials
    n_pseudo = 6

    # k (number of folds)
    k = 6

    # Number of original trials (repetitions) per pseudo-trial:
    n_original_trial_pseudo = int(max_rep / n_pseudo)

    # List
    pseudo_trials = [(i + 1) * 5 for i in range(n_pseudo)]

    # timepoints
    timepoints = len(time)

    # Check if there are 180 conditions (different stimuli) within the test
    # data:
    n_conditions = len(np.unique(img_cat))

    # number of entries in lower triangle exluding main diagonal
    num_entries = (n_conditions - 1) * n_conditions // 2

    # Finding the max amount of repetitions of the different conditions
    img_rep = np.zeros(len(np.unique(img_cat)), dtype=int)

    # Finding the indices of the selected image
    for c in range(n_conditions):  # image category

        idx = np.where(img_cat == np.unique(img_cat)[c])[0]
        img_rep[c] = len(idx)

    if not np.all(img_rep == max_rep):
        print("Not all stimuli have the same number of presentations")

    # -------------------------------------------------------------------------
    # STEP 1.4 Decoding Analysis
    # -------------------------------------------------------------------------
    # Following the proposed preprocessing order in Guggenmos et al. (2018).
    # For each stimulus combination (180*180/2) or in other words pairwise
    # combination of the conditions:
    eeg_con_A = np.zeros((max_rep, n_chan, timepoints), dtype=float)
    eeg_con_B = np.zeros((max_rep, n_chan, timepoints), dtype=float)
    pseudo_eeg_con_A = np.zeros((n_pseudo, n_chan, timepoints), dtype=float)
    pseudo_eeg_con_B = np.zeros((n_pseudo, n_chan, timepoints), dtype=float)

    # triangle matrix
    triangle_mean = np.zeros(
        (n_conditions, n_conditions, timepoints), dtype=float
    )

    # count variable
    count_total = 0

    for conA in tqdm(range(n_conditions)):
        num_of_comparisons = conA + 1

        # For videos: need to find the idx of the condition A and B stimuli
        if input_type == "miniclips":
            conditionA = np.unique(img_cat)[conA]

            # Get all the trials (repetitions) of condition A
            idx_con_A = np.where(img_cat == conditionA)[0]

            for r in range(max_rep):  # image repetitions
                eeg_con_A[r, :, :] = eeg_data[idx_con_A[r], :, :]

        # For images: the data is already organized in terms of categories
        elif input_type == "images":
            eeg_con_A = eeg_data[conA, :, :, :]

        for conB in range(num_of_comparisons):
            scores = np.zeros((k, timepoints), dtype=float)
            if conA == conB:
                continue

            if input_type == "miniclips":
                conditionB = np.unique(img_cat)[conB]

                # Get all the trials (repetitions) of condition B
                idx_con_B = np.where(img_cat == conditionB)[0]
                for rep in range(max_rep):  # image repetitions
                    eeg_con_B[rep, :, :] = eeg_data[idx_con_B[rep], :, :]

            elif input_type == "images":
                eeg_con_B = eeg_data[conB, :, :, :]

            # Create an array which contains the scores for each time-point
            # averaged over k
            scores_mean = np.zeros((timepoints), dtype=float)

            # -------------------------------------------------------------
            # STEP 1.4.1 Shuffle the Data
            # -------------------------------------------------------------

            # Shuffle the data for condition A and condition B
            ran_eeg_con_A = rng.permutation(eeg_con_A, axis=0)
            ran_eeg_con_B = rng.permutation(eeg_con_B, axis=0)

            # -------------------------------------------------------------
            # STEP 1.4.2 Create Pseudo-Trials
            # -------------------------------------------------------------
            # Condition A
            for p in range(n_pseudo):
                start = 0 + pseudo_trials[p] - 5
                end = pseudo_trials[p]
                subdataset = ran_eeg_con_A[start:end, :, :]
                pseudo_eeg_con_A[p, :, :] = (
                    np.sum(subdataset, axis=0)
                ) / n_original_trial_pseudo

            # Condition B
            for p in range(n_pseudo):
                start = 0 + pseudo_trials[p] - 5
                end = pseudo_trials[p]
                subdataset = ran_eeg_con_B[start:end, :, :]
                pseudo_eeg_con_B[p, :, :] = (
                    np.sum(subdataset, axis=0)
                ) / n_original_trial_pseudo

            # -------------------------------------------------------------
            # STEP 1.4.3 Create Full Dataset and Labels
            # -------------------------------------------------------------
            # Concatenate pseudo-trials for condition A and condition B
            full_data = np.concatenate(
                (pseudo_eeg_con_A, pseudo_eeg_con_B), axis=0
            )

            # Transpose array
            full_data = full_data.transpose(2, 0, 1)

            # Create y labels
            # Those indicate to which condition the sample in the training-data
            # belong. #1 indicates A, #0 indicates B.
            full_data_label = np.empty((n_pseudo * 2), dtype=float)
            full_data_label[:n_pseudo] = 1
            full_data_label[n_pseudo:] = 0

            # -------------------------------------------------------------
            # STEP 1.4.4 Create Cross-validation Pipeline Parameter
            # -------------------------------------------------------------
            # Stratified k-fold CV guarantees that the number of trials
            # for each condition is equal in the test set (see also King et al., 2014)
            # ----
            cv_k = StratifiedKFold(n_splits=k, shuffle=False)

            # Define model (linear SVM) using the default value of C = 1
            svm_def = svm.SVC(kernel="linear", C=1, random_state=42)

            # -------------------------------------------------------------
            # STEP 1.4.5 Cross Validation (for each time-point)
            # -------------------------------------------------------------

            dummy_data = full_data[0, :, :]

            count = 0
            for train_index, test_index in cv_k.split(
                dummy_data, full_data_label
            ):
                # print(f'{train_index}, {test_index}')

                F_train = full_data[:, train_index, :]

                # Estimate MVNN based on the training data
                mvnn_transformer = MVNN_Transformer(F=F_train)
                mvnn_transformer.fit()

                for time_eeg in range(timepoints):
                    data_tp = full_data[time_eeg, :, :]

                    X_train, X_test = (
                        data_tp[train_index, :],
                        data_tp[test_index, :],
                    )
                    y_train, y_test = (
                        full_data_label[train_index],
                        full_data_label[test_index],
                    )
                    # Apply MVNN to the training data
                    X_train_mvnn = mvnn_transformer.transform(X=X_train)

                    # Create pipeline for the inner loop
                    estimators = [("svm", svm_def)]
                    pipe = Pipeline(estimators)

                    # Fit
                    pipe.fit(X_train_mvnn, y_train)

                    data_time_gen = data_tp
                    X_test = data_time_gen[test_index, :]
                    y_test = full_data_label[test_index]
                    # Apply MVNN to the test data
                    X_test_mvnn = mvnn_transformer.transform(X=X_test)

                    # Score
                    score_pipe = pipe.score(X_test_mvnn, y_test)

                    # save score
                    scores[count, time_eeg] = score_pipe

                # count variable
                count += 1
                print(f"k = {count}")
                # print progress
                count_total += 1
                progress_total = (count_total / (num_entries * k)) * 100
                print(f"Progress {progress_total}%")

            # average accuracy over all k iterations - 2 dimensions
            scores_mean = np.mean(scores, axis=0)

            # store it in the upper triangle (RDM)
            triangle_mean[conA, conB, :] = scores_mean

    # -------------------------------------------------------------------------
    # STEP 1.5 Vectorize Results and Prepare Outputs
    # -------------------------------------------------------------------------
    # Vectorized results (no time generalization)
    triangle_mean_3 = np.zeros((180, 180, timepoints))
    for i in range(n_conditions):
        for j in range(n_conditions):
            for k in range(timepoints):
                value = triangle_mean[i, j, k]
                triangle_mean_3[i, j, k] = value
    triangle_mean_3 = np.transpose(triangle_mean_3, (2, 0, 1))

    final_results_mean = np.zeros((timepoints, num_entries), dtype=float)
    for tp in range(timepoints):
        # only save the lower triangle and vectorize it (excluding main diagonal)
        # get the indices of the lower triangle, excluding main diagonal
        tp_matrix_mean = triangle_mean_3[tp, :, :]
        i, j = np.tril_indices_from(tp_matrix_mean, k=-1)
        lower_triangle_mean = tp_matrix_mean[i, j]
        vectorized_mean = lower_triangle_mean.flatten()

        final_results_mean[tp, :] = vectorized_mean

    # calculate mean decoding accuracy for each timepoint
    mean_accuracy = final_results_mean.mean(axis=1)

    # -------------------------------------------------------------------------
    # STEP 1.6 Save Results
    # -------------------------------------------------------------------------
    # Putting the results into a dictionary
    decoding_single_subject = {
        "mean_accuracies_over_conditions": mean_accuracy,
        "final_results_mean": final_results_mean,
    }

    if sub < 10:
        fileDir = "decoding_{}_".format(input_type) + "sub-0{}_redone".format(
            sub
        )
    else:
        fileDir = "decoding_{}_".format(input_type) + "sub-{}_redone".format(
            sub
        )

    save_dir = os.path.join(save_dir, "decoding", input_type)

    # Creating the directory if not existing
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, fileDir), decoding_single_subject)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()

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
        "-s",
        "--sub",
        default=9,
        type=int,
        metavar="",
        help="subject number. images: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],"
        "videos: [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]",
    )
    parser.add_argument(
        "--mvnn_dim", default="epochs", type=str,
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
        "-inp",
        "--input_type",
        default="images",
        metavar="",
        type=str,
        help="miniclips or images",
    )
    parser.add_argument(
        "--it", default=1, type=int, metavar="", help="Iteration"
    )
    parser.add_argument(
        "--exclude_guitar_trials",
        action="store_true",
        help="Exclude guitar trials from the analysis.",
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    eeg_dir = config.get(args.config, "eeg_dir")
    save_dir = config.get(args.config, "save_dir")

    sub = args.sub
    freq = args.freq
    mvnn_dim = args.mvnn_dim
    region = args.region
    input_type = args.input_type
    it = args.it

    # -----------------------------------------------------------------------------
    # STEP 3: Run Decoding Function
    # -----------------------------------------------------------------------------
    decoding_single_subject_func(
        sub, mvnn_dim, freq, region, eeg_dir, save_dir, input_type, it
    )
