#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
MULTIVARIATE NOISE NORMALIZATION (WHITENING) FOR ENCODING

This script conducts the MVNN for the enconding analysis. Note that the MVNN
for the decoding analysis is implemented in a different script. Thus, only
use this MVNN for encoding.

Acknowledgments: This script is based on a script by Raphael Leuner and Vanshika
Bawa from the Cichy Lab.

@author: Alexander Lenders, Agnessa Karapetian
"""
import argparse
from utils import load_config
import os
import numpy as np
from sklearn.covariance import LedoitWolf
import scipy


# -----------------------------------------------------------------------------
# STEP 2: Define MVNN Fit Function
# -----------------------------------------------------------------------------
def mvnn_fit(sub, mvnn_dim, freq, region, input_type, data_dir):
    """
    MVNN is fitted only on the training data, but applied to the training,
    test and validation data.

    Input:
    -------
    Training EEG data set, which is already preprocessed. The input is a
    dictionary which includes:
        a. EEG-Data (eeg_data, 5400 Videos x 64 Channels x 70 Timepoints)
        b. Video Categories (img_cat, 5400 x 1) - Each video has one specific ID
        c. Channel names (channels, 64 x 1 OR 19 x 1)
        d. Time (time, 70 x 1) - Downsampled timepoints of a video

    Returns:
    -------
    The inverse of the covariance matrix (sigma), calculated for each time-point
    or epoch of each condition, and then averaged.

    More information:
        Guggenmos et al. (2018)
        https://scikit-learn.org/stable/modules/covariance.html

    Parameters:
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
    input_type : str
        Miniclips or images
    """
    _cov = lambda x: LedoitWolf().fit(x).covariance_

    print(
        "\n\n\n>>> Mvnn %s, %dhz, sub %s, brain region: %s <<<"
        % (mvnn_dim, freq, sub, region)
    )

    if input_type == "miniclips":
        if sub < 10:
            folderDir = os.path.join(
                data_dir,
                "{}".format(input_type)
                + "/sub-0{}".format(sub)
                + "/eeg/preprocessing/ica"
                + "/"
                + "/training"
                + "/"
                + region
                + "/",
            )
            fileDir = (
                ("sub-0{}".format(sub))
                + "_seq_"
                + "training"
                + "_"
                + str(freq)
                + "hz_"
                + region
                + ".npy"
            )
        else:
            folderDir = os.path.join(
                data_dir,
                "{}".format(input_type)
                + "/sub-{}".format(sub)
                + "/eeg/preprocessing/ica"
                + "/"
                + "/training"
                + "/"
                + region
                + "/",
            )
            fileDir = (
                ("sub-{}".format(sub))
                + "_seq_"
                + "training"
                + "_"
                + str(freq)
                + "hz_"
                + region
                + ".npy"
            )
    elif input_type == "images":
        if sub < 10:
            folderDir = os.path.join(
                data_dir,
                "{}".format(input_type)
                + "/prepared"
                + "/sub-0{}".format(sub)
                + "/train/{}/{}hz/".format(region, freq),
            )
            fileDir = "train_img_data_{}hz_sub_00{}.npy".format(freq, sub)
        else:
            folderDir = os.path.join(
                data_dir,
                "{}".format(input_type)
                + "/prepared"
                + "/sub-{}".format(sub)
                + "/train/{}/{}hz/".format(region, freq),
            )
            fileDir = "train_img_data_{}hz_sub_0{}.npy".format(freq, sub)

    total_dir = os.path.join(folderDir, fileDir)
    data = np.load(total_dir, allow_pickle=True).item()

    eeg_data = data["eeg_data"]
    img_cat = data["img_cat"]
    channels = data["channels"]
    time = data["time"]

    del data

    num_nan = np.isnan(eeg_data).sum()
    print(f"There are {num_nan} NAN values in the input data")

    # Appending the data
    n_conditions = len(np.unique(img_cat))

    # n_conditions * n_channels * n_channels
    sigma_ = np.empty((n_conditions, len(channels), len(channels)))
    sigma_[:] = np.nan  # fill the entries with nan
    count_2 = 0

    for c in range(n_conditions):  # image category
        if input_type == "miniclips":
            # give me the indices for the cth condition:
            idx = img_cat == np.unique(img_cat)[c]
            cond_data = eeg_data[idx, :, :]
        elif input_type == "images":
            cond_data = eeg_data[
                c, :, :, :
            ]  # due to slight differences in preprocessing

        # ******************************************************************
        # THIS IS THE CORRECT EPOCHS METHOD acc. to Guggenmos et al. (2018) 
        # ******************************************************************
        # Computing the covariance matrices
        # count_2 refers to the condition (since sigma_ has the dimensions
        # n_cond, number of channels, number of channels)
        if (
            mvnn_dim == "epochs"
        ): 
            # Computing sigma for each time point, then averaging across time
            sigma_[count_2] = np.mean(
                [_cov(cond_data[:, :, t]) for t in range(len(time))], axis=0
            )
            count_2 += 1

    # Averaging sigma across conditions
    sigma_nan = np.isnan(sigma_).sum()
    print(f"There are {sigma_nan} NAN values in the covariance matrix")
    sigma = sigma_.mean(axis=0)
    sigma_nan_2 = np.isnan(sigma).sum()
    print(f"There are {sigma_nan_2} NAN values in the covariance matrix")

    # Computing the inverse matrix of sigma
    """
    This code is using the scipy.linalg.fractional_matrix_power function 
    from the SciPy library to compute the inverse of a square matrix sigma.
    For more information, see Guggenmos et al. (2018), equation (1), p. 436
    """

    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

    return sigma_inv


# -----------------------------------------------------------------------------
# STEP 2: Define MVNN Transform Function
# -----------------------------------------------------------------------------
def mvnn_transform(
    img_type, sub, mvnn_dim, freq, region, sigma_inv, input_type, data_dir
):
    """
    MVNN is fitted only on the training data, but applied to the training,
    test and validation data.
    In addition, the trials are shuffled.

    Input:
    -------
    Test, Training and Validation EEG data sets, which are already
    preprocessed. The input are dictionaries, which include:
        a. EEG-Data (eeg_data, 5400 Videos x 64 Channels x 70 Timepoints)
        b. Video Categories (img_cat, 5400 x 1) - Each video has one specific ID
        c. Channel names (channels, 64 x 1 OR 19 x 1)
        d. Time (time, 70 x 1) - Downsampled timepoints of a video
        In case of the validation data set there are 900 videos instead of 5400.

    Returns:
    -------
    EEG data after MVNN for all the different EEG data sets. The dictionaries
    have the same stucture, i.e. keys and values.

    More information:
        Guggenmos et al. (2018)
        https://scikit-learn.org/stable/modules/covariance.html

    Parameters:
    ----------
    img_type: str
        data set (training, validation or test)
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
    sigma_inv: numpy array
        Inverse covariance matrix
    input_type: str
        Miniclips or images
    """
    if input_type == "miniclips":
        if sub < 10:
            main_folderDir = os.path.join(
                data_dir,
                "{}".format(input_type)
                + "/sub-0{}".format(sub)
                + "/eeg/preprocessing/ica",
            )

            folderDir = os.path.join(
                main_folderDir, img_type + "/" + region + "/"
            )
            fileDir = (
                ("sub-0{}".format(sub))
                + "_seq_"
                + img_type
                + "_"
                + str(freq)
                + "hz_"
                + region
                + ".npy"
            )
        else:
            main_folderDir = os.path.join(
                data_dir,
                "{}".format(input_type)
                + "/sub-{}".format(sub)
                + "/eeg/preprocessing/ica",
            )
            folderDir = os.path.join(
                main_folderDir, img_type + "/" + region + "/"
            )
            fileDir = (
                ("sub-{}".format(sub))
                + "_seq_"
                + img_type
                + "_"
                + str(freq)
                + "hz_"
                + region
                + ".npy"
            )
    elif input_type == "images":
        if sub < 10:
            main_folderDir = os.path.join(
                data_dir,
                "{}".format(input_type) + "/prepared" + "/sub-0{}".format(sub),
            )
            fileDir = "{}_img_data_{}hz_sub_00{}.npy".format(
                img_type, freq, sub
            )
        else:
            main_folderDir = os.path.join(
                data_dir,
                "{}".format(input_type) + "/prepared" + "/sub-{}".format(sub),
            )
            fileDir = "{}_img_data_{}hz_sub_0{}.npy".format(
                img_type, freq, sub
            )
        folderDir = os.path.join(
            main_folderDir + "/{}/{}/{}hz/".format(img_type, region, freq)
        )

    total_dir = os.path.join(folderDir, fileDir)

    # Loading the data
    data = np.load(total_dir, allow_pickle=True).item()
    eeg_data = data["eeg_data"]
    img_cat = data["img_cat"]
    time = data["time"]
    channels = data["channels"]

    del data

    # Whitening using the epoch method
    """
    Whitening is a preprocessing step that is often used in the analysis of EEG
    data. It is a technique that is used to remove correlations and reduce 
    the redundancy. Goal is to transform the data so that covariance matrix is
    the identitiy matrix.
    """

    # Correcting the data with the inverse of sigma (Whitening)
    # Matrix multiplication (@) between the eeg data and the whitening matrix
    # swapaxes changes the axes of the matrix (the dimensions)
    if input_type == "images":
        eeg_data = np.reshape(
            eeg_data,
            (
                eeg_data.shape[0] * eeg_data.shape[1],
                eeg_data.shape[2],
                eeg_data.shape[3],
            ),
        )
    eeg_data = np.real(eeg_data.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)

    # Storing the data in a dictionary
    eeg_data = {
        "eeg_data": eeg_data,
        "img_cat": img_cat,
        "time": time,
        "channels": channels,
    }

    if sub < 10:
        fileDir = (
            ("sub-0{}".format(sub))
            + "_seq_"
            + img_type
            + "_"
            + str(freq)
            + "hz_"
            + region
            + "_prepared_"
            + mvnn_dim
            + "_redone.npy"
        )
    else:
        fileDir = (
            ("sub-{}".format(sub))
            + "_seq_"
            + img_type
            + "_"
            + str(freq)
            + "hz_"
            + region
            + "_prepared_"
            + mvnn_dim
            + "_redone.npy"
        )

    np.save(os.path.join(folderDir, fileDir), eeg_data)


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
        "-i",
        "--input_type",
        default="miniclips",
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)

    freq = args.freq
    mvnn_dim = args.mvnn_dim
    region = args.region
    input_type = args.input_type
    data_dir = config.get(args.config, "eeg_dir")

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
    elif input_type == "images":
        subjects = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    for sub in subjects:
        sigma_inv = mvnn_fit(
            sub, mvnn_dim, freq, region, input_type, data_dir=data_dir
        )
        if input_type == "miniclips":
            image_types = ["training", "test", "validation"]
        elif input_type == "images":
            image_types = ["train", "test", "val"]
        for type in image_types:
            mvnn_transform(
                type,
                sub,
                mvnn_dim,
                freq,
                region,
                sigma_inv,
                input_type,
                data_dir=data_dir,
            )
