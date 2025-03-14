#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR ENCODING ANALYSIS (PERMUTATION TESTS) - DIFFERENCE: STATIC IMAGES - VIDEOS 

This script implements the statistical analysis for the encoding analysis, more precisely
it tests whether the DIFFERENCES between videos and images are significant for each tp. 
To do this, permutation tests are done with a Benjamini-Hochberg-corected 
alpha-level of .05. The statistical tests are two-sided per default.
Chance-level of pairwise decoding is 0.5. 

@author: AlexanderLenders, AgnessaKarapetian

Anaconda Environment on local machine: mne

"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    
    # add arguments / inputs
    parser.add_argument('-ls_v', "--list_sub_vid", default=0, type=int, 
                        metavar='', help="list of subjects for videos (see below)")
    parser.add_argument('-ls_i', "--list_sub_img", default=0, type=int, 
                        metavar='', help="list of subjects for images (see below)")
    parser.add_argument('-d', "--workdirvid", 
                        default = 'Z:/Unreal/Results/Encoding/',
                        type = str, metavar='', help="Results of the decoding analysis with videos")
    parser.add_argument('-id', "--workdirimg", 
                        default = 'Z:/Unreal/images_results/encoding/',
                        type = str, metavar='', help="Results of the decoding analysis with images")
    parser.add_argument('-sd', "--savedir", 
                        default = 'Z:/Unreal/images_results/encoding/redone/stats/',
                        type = str, metavar='', help="Where to save results")
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-tp', "--num_tp", default = 70, type = int, 
                        metavar='', help="Number of timepoints")
    parser.add_argument('-a', "--alpha", default = 0.05, type = int, 
                        metavar='', help="Signifance level (alpha)")
    parser.add_argument('-t', "--tail", 
                        default = 'both',
                        type = str, metavar='', 
                        help="One-sided: right, two-sided: both")

    
    args = parser.parse_args() # to get values for the arguments
    
    list_sub_vid = args.list_sub_vid     
    list_sub_img = args.list_sub_img      
    workDir_img = args.workdirimg
    workDir_vid = args.workdirvid
    saveDir = args.savedir
    n_perm = args.num_perm
    timepoints = args.num_tp
    n_perm = args.num_perm
    alpha = args.alpha
    tail = args.tail
    
# -----------------------------------------------------------------------------
# STEP 2: Define Permutation Test Function
# -----------------------------------------------------------------------------

def permutation_test(list_sub_vid, list_sub_img, workDir_vid, workDir_img, saveDir, n_perm, tail, alpha, 
                     timepoints): 
    """
    Inputs: 
    ----------
    Decoding results with videos AND static images.

    
    Returns
    ----------
    Results of the statistical analysis (videos-images), dictionary with the following keys: 
        a. Uncorrected_p_values_map 
            - Contains uncorrected p-value for each element in time gen matrix
        b. Corrected_p_values_map
            - Contains corrected p-value for each element in time gen matrix 
        c. Boolean_statistical_map
            - Contains boolean values for each element in time gen matrix, if 
            True -> corrected p-value is lower than .05
    
    Parameters
    ----------
    list_sub_vid : list 
        List with subjects which should be included in the statistical analysis for decoding with videos
    list_sub_img : list 
        List with subjects which should be included in the statistical analysis for decoding with images
    workDir_vid : str 
        Directory with the results of the decoding analysis with videos
    workDir_img : str 
        Directory with the results of the decoding analysis with static images
    saveDir : str 
        Directory where to save the results of the statistical analysis. 
    n_perm : int 
        Number of permutations (Default: 10,000)
    tail : str
        Whether two conduct an one-sided test (right) or two-sided test (both)
        (Default: both)
    alpha: int 
        Alpha significance level (Default: 0.05)
    timepoints: int 
        Number of timepoints in EEG epoch

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import os
    import numpy as np
    from scipy.stats import rankdata
    import statsmodels
    from statsmodels.stats.multitest import multipletests
    import pickle
    
    n_sub_vid = len(list_sub_vid)
    n_sub_img = len(list_sub_img)
    
    # set random seed (for reproduction)
    np.random.seed(42)
        
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------   

    identifierDir = 'seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_7features_onehot.pkl'
    feature_names = ('edges','world_normal', 'lighting','scene_depth', 'reflectance', 'skeleton','action')

    results_vid = {}
    results_img = {}
    for subject in list_sub_vid:
        fileDir_vid = (workDir_vid + 'redone/7_features/{}_'.format(subject) + identifierDir)
        encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)
        results_vid[str(subject)] = encoding_results_vid

    for subject in list_sub_img:
        fileDir_img = (workDir_img + 'redone/7_features/{}_'.format(subject) + identifierDir)
        encoding_results_img = np.load(fileDir_img, allow_pickle=True)
        results_img[str(subject)] = encoding_results_img

    # -------------------------------------------------------------------------
    # STEP 2.3 Permutation (Create null distribution)
    # -------------------------------------------------------------------------
    ### Loop over features for this one ###
    feature_results = {}
    for feature in feature_names: 
        
        results_f_vid = np.zeros((n_sub_vid, timepoints)) 

        for index, subject in enumerate(list_sub_vid): 
    
            subject_result = results_vid[str(subject)][feature]['correlation']
            subject_result_averaged = np.mean(subject_result, axis = 1) # averaged over all channels 
            results_f_vid[index, :] = subject_result_averaged
                
        results_f_img = np.zeros((n_sub_img, timepoints)) 

        for index, subject in enumerate(list_sub_img): 
    
            subject_result = results_img[str(subject)][feature]['correlation']
            subject_result_averaged = np.mean(subject_result, axis = 1) # averaged over all channels 
            results_f_img[index, :] = subject_result_averaged    
       
        # create statistical map for all permutation 
        stat_map = np.zeros((n_perm, timepoints))
        
        # create mean for each timepoint over all participants 
        # this is our "original data" and permutation 1 in the stat_map 
        mean_orig_vid = np.mean(results_f_vid, axis = 0)
        mean_orig_img = np.mean(results_f_img, axis = 0)

       
        t_stat = mean_orig_vid - mean_orig_img

        stat_map[0, :] = t_stat
        
        for permutation in range(1, n_perm): 
            # create array with -1 and 1 (randomization)
            perm_vid = np.expand_dims(np.random.choice([-1, 1], size=(n_sub_vid,), replace=True), 1)
            perm_img = np.expand_dims(np.random.choice([-1, 1], size=(n_sub_img,), replace=True), 1)

            
            # create randomization matrix
            rand_matrix_vid = np.broadcast_to(perm_vid, (n_sub_vid, timepoints))
            rand_matrix_img = np.broadcast_to(perm_img, (n_sub_img, timepoints))
            
            # elementwise multiplication 
            permutation_mat_vid = np.multiply(results_f_vid, rand_matrix_vid)
            permutation_mat_img = np.multiply(results_f_img, rand_matrix_img)
            
            mean_orig_vid = np.mean(permutation_mat_vid, axis = 0)
            mean_orig_img = np.mean(permutation_mat_img, axis = 0)            
           
            t_stat = mean_orig_img - mean_orig_vid

            # calculate mean and put it in stats map 
            stat_map[permutation, :] = t_stat
                
        # -------------------------------------------------------------------------
        # STEP 2.4 Calculate ranks and p-values
        # -------------------------------------------------------------------------
        # get ranks (over all permutations), this gives us a distribution
        if tail == 'right':
            ranks = (np.apply_along_axis(rankdata, 0, stat_map))
            
        elif tail == 'both': 
            abs_values = np.absolute(stat_map)
            ranks = (np.apply_along_axis(rankdata, 0, abs_values))
        
        # calculate p-values
        # create a matrix with nperm+1 values in every element (to account for 
        # the observed test statistic)
        sub_matrix = np.full((n_perm, timepoints), (n_perm+1))
        p_map = (sub_matrix - ranks)/n_perm
        p_values = p_map[0, :]

        # -------------------------------------------------------------------------
        # STEP 2.5 Benjamini-Hochberg correction
        # -------------------------------------------------------------------------
        """
        Please note, that we assume a positive dependence between the different 
        statistical tests. 
        """
        rejected, p_values_corr = statsmodels.stats.multitest.fdrcorrection(
            p_values, alpha = alpha, is_sorted = False)
        
        stats_results = {}
        stats_results['Uncorrected_p_values_map'] = p_values
        stats_results['Corrected_p_values_map'] = p_values_corr
        stats_results['Boolean_statistical_map'] = rejected

        feature_results[feature] = stats_results
    # -------------------------------------------------------------------------
    # STEP 2.6 Save results of analysis
    # -------------------------------------------------------------------------  

    # Save the dictionary
    fileDir = ('encoding_unreal_before_pca_difference_stats_{}_nonstd.pkl'.format(tail))  
    
    savefileDir = os.path.join(saveDir, fileDir) 
     
    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False: # if not a directory
        os.makedirs(os.path.join(saveDir))
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(feature_results, f)
        
# -----------------------------------------------------------------------------
# STEP 3: Run function
# -----------------------------------------------------------------------------  
list_sub_vid = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]

list_sub_img = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

permutation_test(list_sub_vid, list_sub_img, workDir_vid, workDir_img, saveDir, n_perm, tail, alpha,
                 timepoints) 
    

    