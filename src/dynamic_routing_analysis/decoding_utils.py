"""Utilities for decoding neural activity.

This module provides functions and utilities for training and evaluating
machine learning models to decode behavioral or task variables from neural
activity patterns.
"""

import gc
import logging
import pickle
import time

import npc_lims
import numpy as np
import pandas as pd
import polars as pl
import upath
import zarr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from dynamic_routing_analysis import data_utils, spike_utils

logger = logging.getLogger(__name__)

class NotEnoughBlocksError(Exception):
    """Raised when there are insufficient blocks for blockwise cross-validation."""
    pass



def decoder_helper(input_data,labels,decoder_type='linearSVC',crossval='5_fold',
                   crossval_index=None,labels_as_index=False,train_test_split_input=None,train_test_split_label=None,
                   regularization=None,penalty=None,solver=None,n_jobs=None,set_random_state=None,
                   other_data=None,scaler='robust'):
    """Train and evaluate a decoder to predict labels from input data.
    
    This function provides a flexible interface for training various classification
    models using different cross-validation strategies. It supports multiple decoder
    types, scaling methods, and cross-validation approaches.
    
    Parameters
    ----------
    input_data : array-like, shape (n_samples, n_features)
        Training data matrix where rows are samples and columns are features.
    labels : array-like, shape (n_samples,)
        Target labels for each sample.
    decoder_type : str, default='linearSVC'
        Type of decoder to use. Options:
        - 'linearSVC': Linear Support Vector Classification
        - 'nonlinearSVC': Non-linear SVC with RBF kernel
        - 'LDA': Linear Discriminant Analysis
        - 'RandomForest': Random Forest Classifier
        - 'LogisticRegression': Logistic Regression
    crossval : str, default='5_fold'
        Cross-validation strategy. Options:
        - '5_fold': 5-fold stratified cross-validation with shuffling
        - '5_fold_constant': 5-fold with pre-defined splits
        - '5_fold_set_random_state': 5-fold with specified random state
        - 'blockwise': Leave-one-block-out cross-validation
        - 'forecast_train_2': Forecast with 2 consecutive training blocks
        - 'forecast_train_3': Forecast with 3 consecutive training blocks
        - 'custom': User-defined train/test splits
    crossval_index : array-like, optional
        Block numbers for each trial, required for blockwise or forecast cross-validation.
    labels_as_index : bool, default=False
        If True, convert labels to integer indices.
    train_test_split_input : list of tuples, optional
        Pre-defined train/test splits for 'custom' or '5_fold_constant' cross-validation.
    train_test_split_label : string, optional
        label to easily identify results from custom train_test_split_input
    regularization : float, optional
        Regularization parameter (C) for SVC and Logistic Regression.
        If None, uses default value of 1.0.
    penalty : str, optional
        Penalty type ('l1' or 'l2') for LinearSVC and Logistic Regression.
        If None, uses 'l2' as default.
    solver : str, optional
        Solver algorithm. Used for LDA and Logistic Regression.
        If None, uses 'svd' for LDA and 'lbfgs' for Logistic Regression.
    n_jobs : int, optional
        Number of parallel jobs for Random Forest and Logistic Regression.
    set_random_state : int, optional
        Random state for reproducible 5-fold cross-validation.
    other_data : array-like, shape (n_samples_other, n_features), optional
        Additional data to predict on after training. Will be scaled using
        the same scaler fitted on training data.
    scaler : str, default='robust'
        Type of feature scaling. Options:
        - 'robust': RobustScaler (resistant to outliers)
        - 'standard': StandardScaler (z-score normalization)
        - 'none': No scaling
    
    Returns
    -------
    output : dict
        Dictionary containing:
        
        - 'cr': Classification reports for test data (list)
        - 'pred_label': Predicted labels from training on all data
        - 'true_label': Original labels
        - 'pred_label_all': Predicted labels for each cross-validation fold
        - 'trials_used': Test trial indices for each fold
        - 'decision_function': Decision values from cross-validation
        - 'decision_function_all': Decision values from training on all data
        - 'predict_proba': Predicted probabilities from cross-validation
        - 'predict_proba_all_trials': Predicted probabilities from all data
        - 'coefs': Model coefficients from training on all data
        - 'coefs_all': Coefficients from each cross-validation fold
        - 'classes': Class labels from each fold
        - 'intercept': Model intercept
        - 'params': Model parameters
        - 'balanced_accuracy_test': Mean balanced accuracy on test data
        - 'balanced_accuracy_train': Mean balanced accuracy on training data
        - 'pred_label_train': Training predictions for each fold
        - 'true_label_train': True training labels for each fold
        - 'cr_train': Classification reports for training data
        - 'train_trials': Training trial indices for each fold
        - 'test_trials': Test trial indices for each fold
        - 'models': Trained model objects for each fold
        - 'scaler': Fitted scaler object
        - 'label_names': Unique label values
        - 'labels': Label indices
        - 'pred_label_other': Predictions for other_data (if provided)
        - 'decision_function_other': Decision values for other_data (if provided)
        - 'predict_proba_other': Predicted probabilities for other_data (if provided)
    
    Raises
    ------
    ValueError
        If required parameters for specific cross-validation methods are not provided.
    
    Notes
    -----
    - All models use balanced class weights to handle imbalanced datasets.
    - For blockwise and forecast cross-validation, crossval_index must be provided.
    - Not all decoders support all features (e.g., RandomForest doesn't have coef_).
    - The function trains a final model on all data in addition to cross-validation.
    
    Examples
    --------
    >>> # Basic usage with linear SVC
    >>> output = decoder_helper(neural_data, task_labels)
    >>> print(f"Accuracy: {output['balanced_accuracy_test']:.3f}")
    
    >>> # Blockwise cross-validation
    >>> output = decoder_helper(neural_data, task_labels, 
    ...                        crossval='blockwise',
    ...                        crossval_index=block_numbers)
    
    >>> # Using a different decoder with custom regularization
    >>> output = decoder_helper(neural_data, task_labels,
    ...                        decoder_type='LogisticRegression',
    ...                        regularization=0.1,
    ...                        penalty='l1',
    ...                        solver='saga')
    """
    
    #helper function to decode labels from input data using different decoder models

    if decoder_type=='linearSVC':
        from sklearn.svm import LinearSVC
        if regularization is None:
            regularization = 1.0
        if penalty is None:
            penalty = 'l2'
        if solver is not None:
            logger.warning('Solver not used for LinearSVC')
        clf=LinearSVC(max_iter=5000,dual='auto',class_weight='balanced',
                      C=regularization,penalty=penalty)
    elif decoder_type=='nonlinearSVC':
        from sklearn.svm import SVC
        kernel='rbf'
        if regularization is None:
            regularization = 1.0
        if penalty is not None:
            logger.warning('Penalty not used for non-linear SVC')
        if solver is not None:
            logger.warning('Solver not used for non-linear SVC')
        clf=SVC(max_iter=5000,probability=True,class_weight='balanced',
                      C=regularization,kernel=kernel)
    elif decoder_type=='LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        if solver is None:
            solver = 'svd'
        if regularization is not None:
            logger.warning('Regularization not used for LDA')
        if penalty is not None:
            logger.warning('Penalty not used for LDA')
        clf=LinearDiscriminantAnalysis(solver=solver)
    elif decoder_type=='RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        if regularization is not None:
            logger.warning('Regularization not used for RandomForest')
        if penalty is not None:
            logger.warning('Penalty not used for RandomForest')
        if solver is not None:
            logger.warning('Solver not used for RandomForest')
        clf=RandomForestClassifier(class_weight='balanced',n_jobs=n_jobs)
    elif decoder_type=='LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        if regularization is None:
            regularization = 1.0
        if penalty is None:
            penalty = 'l2'
        if solver is None:
            solver = 'lbfgs'
        clf=LogisticRegression(max_iter=5000,class_weight='balanced',
                               C=regularization,penalty=penalty,solver=solver,n_jobs=n_jobs)

    output={}

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()

    if scaler == 'none':
        X = input_data
    else:
        scaler.fit(input_data)
        X = scaler.transform(input_data)

    unique_labels=np.unique(labels)
    if labels_as_index==True:
        labels=np.array([np.where(unique_labels==x)[0][0] for x in labels])
    y = labels

    if other_data is not None:
        if scaler == 'none':
            X_other = other_data
        else:
            X_other = scaler.transform(other_data)

    if len(np.unique(labels))>2:
        y_dec_func=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    else:
        y_dec_func=np.full(len(y), fill_value=np.nan)

    if type(y[0])==bool:
        ypred=np.full(len(y), fill_value=False)
    elif type(y[0])==str:
        ypred=np.full(len(y), fill_value='       ')
    else:
        ypred=np.full(len(y), fill_value=np.nan)
    
    ypred_proba=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    if len(np.unique(labels))>2:
        decision_function=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    else:
        decision_function=np.full((len(y)), fill_value=np.nan)

    tidx_used=[]

    coefs=[]
    classes=[]
    intercept=[]
    params=[]
    ypred_all=[]
    ypred_train=[]
    ytrue_train=[]
    train_trials=[]
    test_trials=[]
    models=[]
    cr_dict_train = []
    balanced_accuracy_train = []
    coefs_all = []

    cr_dict_test = []
    balanced_accuracy_test = []

    #make train, test splits based on block number
    if crossval=='blockwise':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        block_numbers=np.unique(block_number)
        for bb in block_numbers:
            not_block_inds=np.where(block_number!=bb)[0]
            train.append(not_block_inds)
            block_inds=np.where(block_number==bb)[0]
            test.append(block_inds)
        train_test_split=zip(train,test)

    elif 'forecast' in crossval:
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        block_numbers=np.unique(block_number)

        #make list of training and test blocks
        #training = adjacent blocks
        #testing = block before and block after (if exist)
        train_blocks=[]
        test_blocks=[]

        if crossval=='forecast_train_2':
            for bb in block_numbers[:-1]:
                train_blocks.append([bb,bb+1])
                if bb==0:
                    test_blocks.append([bb+2])
                elif bb==block_numbers[-2]:
                    test_blocks.append([bb-1])
                else:
                    test_blocks.append([bb-1,bb+2])

        elif crossval=='forecast_train_3':
            for bb in block_numbers[:-2]:
                train_blocks.append([bb,bb+1,bb+2])
                if bb==0:
                    test_blocks.append([bb+3])
                elif bb==block_numbers[-3]:
                    test_blocks.append([bb-1])
                else:
                    test_blocks.append([bb-1,bb+3])

        for bb in range(0,len(train_blocks)):
            train.append(np.where(np.isin(block_number,train_blocks[bb]))[0])
            test.append(np.where(np.isin(block_number,test_blocks[bb]))[0])

        train_test_split=zip(train,test)

    elif crossval=='5_fold':
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        train_test_split = skf.split(input_data, labels)

    elif crossval=='5_fold_constant' or crossval=='custom':
        if train_test_split_input is None:
            raise ValueError('Must provide train_test_split_input')
        train_test_split = train_test_split_input

    elif crossval=='5_fold_set_random_state':
        if set_random_state==None:
            set_random_state=0
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=set_random_state)
        train_test_split = skf.split(input_data, labels)

    for train,test in train_test_split:

        clf.fit(X[train],y[train])
        prediction=clf.predict(X[test])
        balanced_accuracy_train.append(balanced_accuracy_score(y[train], clf.predict(X[train]),
                                                               sample_weight=None, adjusted=False))

        balanced_accuracy_test.append(balanced_accuracy_score(y[test], clf.predict(X[test]),
                                                               sample_weight=None, adjusted=False))

        ypred_all.append(prediction)
        ypred_train.append(clf.predict(X[train]))
        ytrue_train.append(y[train])
        tidx_used.append([test])
        classes.append(clf.classes_)
        # intercept.append(clf.intercept_)
        params.append(clf.get_params())
        train_trials.append(train)
        test_trials.append(test)

        if decoder_type == 'LDA' or decoder_type == 'RandomForest' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
            ypred_proba[test,:] = clf.predict_proba(X[test])
        else:
            ypred_proba[test,:] = np.full((len(test),len(np.unique(labels))), fill_value=False)

        if decoder_type=='LDA' or decoder_type=='linearSVC' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
            if len(np.unique(labels))>2:
                decision_function[test,:]=clf.decision_function(X[test])
            else:
                decision_function[test]=clf.decision_function(X[test])
        else:
            if len(np.unique(labels))>2:
                decision_function[test,:]=np.full((len(test),len(np.unique(labels))), fill_value=False)
            else:
                decision_function[test]=np.full((len(test)), fill_value=False)

        if decoder_type == 'LDA' or decoder_type == 'linearSVC' or decoder_type == 'LogisticRegression':
            coefs_all.append(clf.coef_)
        else:
            coefs_all.append(np.full((X.shape[1]), fill_value=False))

        models.append(clf)

    #fit on all trials
    clf.fit(X, y)
    ypred = clf.predict(X)

    if other_data is not None:
        ypred_other = clf.predict(X_other)

    if decoder_type == 'LDA' or decoder_type == 'RandomForest' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
        predict_proba_all_trials = clf.predict_proba(X)
        if other_data is not None:
            predict_proba_other = clf.predict_proba(X_other)

    else:
        predict_proba_all_trials = np.full((X.shape[0],len(np.unique(labels))), fill_value=False)
        if other_data is not None:
            predict_proba_other = np.full((X_other.shape[0],len(np.unique(labels))), fill_value=False)

    if decoder_type == 'LDA' or decoder_type == 'linearSVC' or decoder_type == 'LogisticRegression':
        coefs = clf.coef_
        intercept = clf.intercept_
        dec_func_all_trials = clf.decision_function(X)
        if other_data is not None:
            dec_func_other = clf.decision_function(X_other)
    elif decoder_type == 'nonlinearSVC':
        coefs = np.full((X.shape[1]), fill_value=False)
        intercept = np.full((1), fill_value=False)
        dec_func_all_trials = clf.decision_function(X)
        if other_data is not None:
            dec_func_other = clf.decision_function(X_other)
    else:
        coefs = np.full((X.shape[1]), fill_value=False)
        intercept = np.full((1), fill_value=False)
        dec_func_all_trials = np.full((X.shape[0]), fill_value=np.nan)
        if other_data is not None:
            dec_func_other = np.full((X_other.shape[0]), fill_value=np.nan)

    #scikit-learn's classification report
    output['cr']=cr_dict_test

    #predicted label from training/testing on all trials
    output['pred_label']=ypred
    #true original label
    output['true_label']=y
    #predicted label for trials in each train-test fold
    output['pred_label_all']=ypred_all
    #indices of trials used in test for each fold
    output['trials_used']=tidx_used

    #decision function using cross-validated folds
    output['decision_function']=decision_function
    #decision function from training/testing on all trials
    output['decision_function_all']=dec_func_all_trials
    #predict probability using cross-validated folds
    output['predict_proba']=ypred_proba
    #predict probability from training/testing on all trials
    output['predict_proba_all_trials']=predict_proba_all_trials if 'predict_proba_all_trials' in locals() else None
    #coefficients for each feature
    output['coefs']=coefs
    output['coefs_all']=coefs_all
    output['classes']=classes
    output['intercept']=intercept
    #input parameters
    output['params']=params
    #mean balanced accuracy across folds
    output['balanced_accuracy_test']=np.nanmean(balanced_accuracy_test)

    output['pred_label_train']=ypred_train
    output['true_label_train']=ytrue_train

    output['cr_train']=cr_dict_train
    #balanced accuracy for training data (all folds)
    output['balanced_accuracy_train']=np.nanmean(balanced_accuracy_train)

    output['train_trials']=train_trials
    output['test_trials']=test_trials

    output['models']=models
    output['scaler']=scaler
    output['label_names']=unique_labels
    output['labels']=labels

    if train_test_split_input is not None:
        output['train_test_split_input']=train_test_split_input

    if train_test_split_label is not None:
        output['train_test_split_label']=train_test_split_label

    #predicted labels for other data input (if provided)
    if other_data is not None:
        output['pred_label_other']=ypred_other
        output['decision_function_other']=dec_func_other
        output['predict_proba_other']=predict_proba_other
    else:
        output['pred_label_other']=None
        output['decision_function_other']=None
        output['predict_proba_other']=None

    return output


def get_multi_probe_expr(combine_multi_probe_rec):
    """Create a polars expression to filter multi-probe recordings.
    
    Generates a boolean expression that determines whether to combine results from
    multiple probe insertions targeting the same brain structure, or to keep them
    separate. This is useful when analyzing recordings where multiple probes recorded
    from the same structure simultaneously.
    
    Parameters
    ----------
    combine_multi_probe_rec : bool
        If True, combine results across probe insertions for the same structure.
        If False, keep individual probe insertion results separate.
    
    Returns
    -------
    polars.Expr
        A boolean polars expression that can be used to filter a DataFrame.
        Returns True for rows that should be included based on the combination setting.
    
    Notes
    -----
    - When combine_multi_probe_rec is True, includes recordings with multiple
      electrode groups OR recordings that are the sole recording for a structure.
    - When combine_multi_probe_rec is False, includes only recordings with a single
      electrode group OR recordings that are the sole recording for a structure.
    - The 'is_sole_recording' flag identifies cases where only one probe recorded
      from a given structure, making the combine decision irrelevant.
    """
    if combine_multi_probe_rec:
        return pl.col('electrode_group_names').list.len().gt(1) | pl.col('is_sole_recording').eq(True)
    else:
        return pl.col('electrode_group_names').list.len().eq(1) | pl.col('is_sole_recording').eq(True)


def get_structure_grouping(keep_original_structure=False):
    """Get structure grouping dictionary for consolidating brain regions.
    
    Returns a mapping that consolidates over-split brain structures into their
    parent regions. This is needed because some structures were split into
    sub-layers during processing, but should be analyzed as unified regions.
    
    Parameters
    ----------
    keep_original_structure : bool, default=False
        If True, keep both original and grouped structure labels (n_repeats=2).
        If False, only use grouped structure labels (n_repeats=1).
        Note: Currently this parameter is overridden to False within the function.
    
    Returns
    -------
    structure_grouping : dict
        Mapping from sub-structure names to parent structure names:
        - Superior colliculus layers (SCop, SCsg, SCzo) -> 'SCs'
        - Superior colliculus intermediate/deep layers (SCig, SCiw, SCdg, SCdw) -> 'SCm'
        - Entorhinal cortex layers (ECT1-6) -> 'ECT'
    n_repeats : int
        Number of times to repeat rows when merging. 
    
    Notes
    -----
    This grouping is necessary when merging columns from the units table to
    decoder results, ensuring consistent structure labels across datasets.
    
    Examples
    --------
    >>> grouping, n_repeats = get_structure_grouping()
    >>> grouping['SCop']
    'SCs'
    >>> n_repeats
    1
    """
    structure_grouping = {
        'SCop': 'SCs',
        'SCsg': 'SCs',
        'SCzo': 'SCs',
        'SCig': 'SCm',
        'SCiw': 'SCm',
        'SCdg': 'SCm',
        'SCdw': 'SCm',
        "ECT1": 'ECT',
        "ECT2/3": 'ECT',    
        "ECT6b": 'ECT',
        "ECT5": 'ECT',
        "ECT6a": 'ECT', 
        "ECT4": 'ECT',
    }

    if keep_original_structure:
        n_repeats = 2
    else:
        n_repeats = 1

    return structure_grouping, n_repeats


def exclude_structures_from_df(df, exclude_redundant_structures=True, exclude_general_structures=True):
    """Filter out redundant and general structures from decoder results.
    
    Removes structures that should not be included in analysis, either because
    they are redundant sub-divisions of parent structures or because they represent
    non-specific anatomical regions (e.g., fiber tracts, ventricles).
    
    Parameters
    ----------
    df : polars.DataFrame
        Decoder results DataFrame with a 'structure' column.
    exclude_redundant_structures : bool, default=True
        If True, exclude structures that were over-split during processing.
        These include superior colliculus and entorhinal cortex sub-layers that
        should be analyzed as unified regions.
    exclude_general_structures : bool, default=True
        If True, exclude non-specific anatomical regions including:
        - General parent structures (CTXsp, STR, TH, etc.)
        - Fiber tracts and white matter
        - Ventricles and regions outside the brain
        - Structures with lowercase names (indicating fiber tracts)
    
    Returns
    -------
    polars.DataFrame
        Filtered DataFrame with specified structures removed.
    
    Notes
    -----
    Redundant structures are those oversplit by the processing pipeline:
    - Superior colliculus layers: SCop, SCsg, SCzo (superficial) and 
      SCig, SCiw, SCdg, SCdw (intermediate/deep)
    - Entorhinal cortex layers: ECT1, ECT2/3, ECT4, ECT5, ECT6a, ECT6b
    
    General structures include broad anatomical divisions and ventricles or fiber tracts:
    - Parent structures: CTXsp, STR, PAL, TH, HY, MB, P, MY, CB
    - Ventricles: VL, V3, V4, SEZ
    - Fiber tracts/not in brain: fiber tracts, scwm, root, lot, out of brain, undefined
    - Any structure name beginning with lowercase letter
    
    Examples
    --------
    >>> df_filtered = exclude_structures_from_df(df, exclude_redundant_structures=True)
    >>> # SCop, SCsg, etc. will be removed, keeping only grouped 'SCs' and 'SCm'
    """
    redundant_structures =['SCop', 'SCsg', 'SCzo', 'SCig', 'SCiw', 'SCdg', 'SCdw', 'ECT1', 'ECT2/3', 'ECT4', 'ECT5', 'ECT6a', 'ECT6b']
    general_structures =['CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CB', 'VL', 'V3', 'V4', 'SEZ', 
                         'fiber tracts', 'scwm', 'root', 'lot', 'out of brain', 'undefined']

    if exclude_redundant_structures:
        df = df.filter(~pl.col('structure').is_in(redundant_structures))
    if exclude_general_structures:
        df = df.filter(~pl.col('structure').is_in(general_structures),)
        #also remove any structure beginning with lowercase letter - indicating fiber tracts or out of brain
        unique_structures=df['structure'].unique()
        lowercase_structures=[]
        for ss in unique_structures:
            if ss[0].islower():
                lowercase_structures.append(ss)
        if len(lowercase_structures)>0:
            df = df.filter(~pl.col('structure').is_in(lowercase_structures),)

    return df

def load_single_session_decoder_accuracy(results_path, sel_session, combine_multi_probe_rec=True):
    """Load decoder accuracy results for a single session.
    
    Loads and processes decoder accuracy results from parquet files for one session,
    including handling of multi-probe recordings and temporal shift analysis.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
    sel_session : str
        Session ID to load (e.g., '670180_2023-07-27').
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    
    Returns
    -------
    polars.DataFrame
        DataFrame with columns:
        - session_id : Session identifier
        - structure : Brain structure name
        - unit_subsample_size : Number of units used for decoding
        - bin_size : Size of time bins in seconds
        - bin_center : Center time of each bin relative to event
        - repeat_idx : Index of repeated decoder runs
        - balanced_accuracy_test : List of test accuracies for each temporal shift
        - shift_idx : List of shift indices corresponding to accuracies
        
        Sorted by unit_subsample_size and repeat_idx.
    
    Notes
    -----
    - Filters out results where is_all_trials is True (using held-out test sets only)
    - Groups across electrode groups when combine_multi_probe_rec=True
    - Results are aggregated over repeat runs for each shift index
    - shift_idx==0 corresponds to the true (aligned) temporal relationship
    - Other shift indices represent null distributions from shuffled data
    
    Examples
    --------
    >>> df = load_single_session_decoder_accuracy(
    ...     's3://bucket/results/', 
    ...     '670180_2023-07-27'
    ... )
    """
    
    #define grouping columns
    grouping_cols = {
        'session_id',
        'structure',
        'electrode_group_names',
        'unit_subsample_size',
        'bin_size',
        'bin_center',
    }
    
    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    example_session_df = (
        pl.scan_parquet(results_path)
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over(grouping_cols - {'electrode_group_names'}).alias('is_sole_recording'),
        )
        .filter(
            pl.col('session_id').eq(sel_session),
            combine_multi_probe_expr,
            pl.col('is_all_trials').not_(),
        )
        .sort('shift_idx')
        .group_by(
            grouping_cols - {'electrode_group_names'}| {'repeat_idx'}, 
            maintain_order=True,
        )
        .agg(
            pl.col('balanced_accuracy_test', 'shift_idx'),
        ).sort('unit_subsample_size','repeat_idx')
        .collect()
    )

    return example_session_df



def load_structure_average_decoder_accuracy(results_path, session_list, combine_multi_probe_rec=True, exclude_redundant_structures=True, exclude_general_structures=True):
    """Load and average decoder accuracy across sessions for each brain structure.
    
    Computes structure-wise mean decoding accuracies by aggregating results across
    sessions. Separates aligned (true) accuracies from null distributions obtained
    via linear shifting trials.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
    session_list : list of str
        List of session IDs to include in analysis.
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    exclude_redundant_structures : bool, default=True
        If True, exclude over-split structures (e.g., superior colliculus sub-layers).
    exclude_general_structures : bool, default=True
        If True, exclude general anatomical regions and non-neural tissue.
    
    Returns
    -------
    polars.DataFrame
        DataFrame with structure-averaged metrics:
        - structure : Brain structure name
        - unit_subsample_size : Number of units used for decoding
        - bin_size : Size of time bins in seconds
        - bin_center : Center time of each bin relative to event
        - mean_true : Mean aligned decoding accuracy across sessions
        - sem_true : Standard error of mean for true accuracy
        - median_null : Mean of median null accuracies across sessions
        - sem_null : Standard error for null accuracy
        - mean_diff : Mean difference (true - null) across sessions
        - sem_diff : Standard error for the difference
        - num_sessions : Number of sessions contributing to each average
        
        Sorted by mean_diff in descending order.
    
    Notes
    -----
    Processing pipeline:
    1. Filters to specified sessions and multi-probe handling
    2. Averages over repeated unit resampling and decoder runs for each recording
    3. Separates aligned results (shift_idx==0) from null (shift_idx!=0)
    4. Computes median null accuracy for each recording
    5. Calculates true - null difference
    6. Averages across sessions and computes SEM
    
    Structures are retained only if at least one session has data for that
    structure and unit subsample size combination.
    
    Examples
    --------
    >>> df = load_structure_average_decoder_accuracy(
    ...     's3://bucket/results/',
    ...     ['session_id_1', 'session_id_2'],
    ...     combine_multi_probe_rec=True
    ... )
    """

    #define grouping columns
    grouping_cols = {
        'session_id',
        'structure',
        'electrode_group_names',
        'unit_subsample_size',
        'bin_size',
        'bin_center',
    }

    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    # structure-wise average decoding accuracy
    results_df = (
        pl.scan_parquet(results_path)
        .filter(
            pl.col('session_id').is_in(session_list),
        )
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over(grouping_cols - {'electrode_group_names'}).alias('is_sole_recording'),
        )
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials').not_(),
            pl.col('session_id').n_unique().ge(1).over('structure', 'unit_subsample_size')#, 'unit_criteria'),
        )
        
        # get the means for each recording over repeats:
        .group_by(grouping_cols | {'shift_idx'}, maintain_order=True)
        .agg(
            pl.col('balanced_accuracy_test').mean(), # over repeats
        )
        # get the aligned result and median over shifts:
        .group_by(grouping_cols)
        .agg(
            pl.col('balanced_accuracy_test').filter(pl.col('shift_idx') == 0).first().alias('mean_true'),
            pl.col('balanced_accuracy_test').filter(pl.col('shift_idx') != 0).median().alias('median_null'),
        )
        # get the difference between true and null:
        .with_columns(
            pl.col('mean_true').sub(pl.col('median_null')).alias('mean_diff'),
        )
        # get the means over sessions:
        .group_by('structure', 'unit_subsample_size','bin_size','bin_center')#, 'unit_criteria')
        .agg(
            pl.col('mean_true').mean(),
            pl.col('mean_true').std().truediv(pl.col('mean_true').count().pow(0.5)).alias('sem_true'),
            pl.col('median_null').mean(),
            pl.col('median_null').std().truediv(pl.col('mean_true').count().pow(0.5)).alias('sem_null'),
            pl.col('mean_diff').mean(),
            pl.col('mean_diff').std().truediv(pl.col('mean_true').count().pow(0.5)).alias('sem_diff'),
            pl.col('session_id').n_unique().alias('num_sessions')
        )
        .with_columns(
            pl.col('num_sessions').cast(pl.Int64),
        )
        .sort(pl.col('mean_diff').mean().over('structure'), descending=True)
        .collect()
    )

    results_df = exclude_structures_from_df(
        results_df, 
        exclude_redundant_structures=exclude_redundant_structures, 
        exclude_general_structures=exclude_general_structures
    )

    return results_df


def load_session_wise_decoder_accuracy(
        results_path, session_list, session_table, 
        combine_multi_probe_rec=True, keep_original_structure=False, 
        exclude_redundant_structures=True, exclude_general_structures=True):
    """Load decoder accuracy with session-level metadata.
    
    Loads decoder results for multiple sessions and enriches them with behavioral
    and recording metadata from the session table, including unit counts,
    cross-modal d-prime, and block level behavior metrics.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
    session_list : list of str
        List of session IDs to include in analysis.
    session_table : polars.DataFrame or polars.LazyFrame
        DataFrame containing session-level metadata. Must include columns:
        - session_id
        - n_passing_blocks
        - cross_modality_dprime_vis_blocks
        - cross_modality_dprime_aud_blocks
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    keep_original_structure : bool, default=False
        If True, keep both original sub-structure labels and grouped parent structures.
        Currently not fully implemented (overridden to False in get_structure_grouping).
    exclude_redundant_structures : bool, default=True
        If True, exclude over-split structures (e.g., superior colliculus sub-layers).
    exclude_general_structures : bool, default=True
        If True, exclude general anatomical regions and non-neural tissue.
    
    Returns
    -------
    polars.DataFrame
        DataFrame with session-wise decoder metrics and metadata:
        - session_id : Session identifier
        - structure : Brain structure name
        - unit_subsample_size : Number of units used for decoding
        - bin_size : Size of time bins in seconds
        - bin_center : Center time of each bin relative to event
        - mean_true : Aligned decoding accuracy for this session
        - median_null : Median null accuracy from temporal shuffles
        - mean_diff : Difference (true - null) for this session
        - balanced_accuracy_test : List of accuracies for each temporal shift
        - shift_idx : List of shift indices
        - total_n_units : Total number of units recorded in this structure
        - n_passing_blocks : Number of behavioral blocks meeting quality criteria (cross modal dprime>=1)
        - cross_modality_dprime_vis_blocks : Visual context discrimination (list)
        - cross_modality_dprime_aud_blocks : Auditory context discrimination (list)
        
        Sorted by session_id, structure, and unit_subsample_size.
    
    Notes
    -----
    Processing pipeline:
    1. Loads decoder results and filters to specified sessions
    2. Joins with consolidated units table to get total_n_units per structure
    3. Applies structure grouping to consolidate over-split regions
    4. Joins with session_table to add behavioral metrics
    5. Averages over repeated decoder runs
    6. Separates aligned (shift_idx==0) from null results
    7. Computes true - null difference for each session
    
    The total_n_units reflects the full population recorded from each structure,
    not just the subsampled units used for decoding.
    
    Examples
    --------
    >>> session_table = pl.read_parquet('session_metadata.parquet')
    >>> df = load_session_wise_decoder_accuracy(
    ...     's3://bucket/results/',
    ...     ['session1', 'session2'],
    ...     session_table
    ... )
    """

    #define grouping columns
    grouping_cols = {
        'session_id',
        'structure',
        'electrode_group_names',
        'unit_subsample_size',
        'bin_size',
        'bin_center',
    }
    
    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)
    structure_grouping, n_repeats = get_structure_grouping(keep_original_structure)

    #add total n units, cross-modal dprime, n good blocks?
    results_session_df = (
        pl.scan_parquet(results_path)
        .filter(
            pl.col('session_id').is_in(session_list),
        )
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over(grouping_cols - {'electrode_group_names'}).alias('is_sole_recording'),
        )
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials').not_(),
        )
        #get total n units
        .join(
            other=(
                pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/units.parquet')
                .with_columns(
                    pl.col('session_id').str.split('_').list.slice(0, 2).list.join('_'),
                )
                #make new rows according to structure_grouping
                .with_columns(
                    pl.when(pl.col('structure').is_in(structure_grouping.keys()))
                    .then(pl.col('structure').repeat_by(n_repeats))
                    .otherwise(pl.col('structure').repeat_by(1))
                )
                .explode('structure')
                .with_columns(
                    pl.when(pl.col('structure').is_in(structure_grouping.keys()).is_first_distinct().over('unit_id'))
                    .then(pl.col('structure').replace(structure_grouping))
                    .otherwise(pl.col('structure'))
                )
                .group_by('session_id','structure')
                .agg(
                    pl.col('unit_id').len().alias('total_n_units')
                )
            ),
            on=['session_id','structure'],
            how='left',
        )
        #join on session table to get cross-modal dprime, etc.
        .join(
            other=session_table.filter(
                pl.col('session_id').is_in(session_list)
            ).select(
                'session_id',
                'n_passing_blocks',
                'cross_modality_dprime_vis_blocks',
                'cross_modality_dprime_aud_blocks',
            ).lazy(),
            on='session_id',
            how='left',
        )
        # get the means for each recording over repeats:
        .group_by(grouping_cols | {'shift_idx', 'n_passing_blocks', 'cross_modality_dprime_vis_blocks', 
                                'cross_modality_dprime_aud_blocks', 'total_n_units'}, maintain_order=True)
        .agg(
            pl.col('balanced_accuracy_test').mean(), # over repeats
        )
        # get the aligned result and median over shifts:
        .group_by(grouping_cols - {'electrode_group_names'} | {'n_passing_blocks', 'cross_modality_dprime_vis_blocks', 
                                                            'cross_modality_dprime_aud_blocks', 'total_n_units'})
        .agg(
            pl.col('balanced_accuracy_test').filter(pl.col('shift_idx') == 0).first().alias('mean_true'),
            pl.col('balanced_accuracy_test').filter(pl.col('shift_idx') != 0).median().alias('median_null'),
            pl.col('balanced_accuracy_test', 'shift_idx').sort_by('shift_idx'),
        )
        # get the difference between true and null:
        .with_columns(
            pl.col('mean_true').sub(pl.col('median_null')).alias('mean_diff'),
        )
        .sort('session_id', 'structure', 'unit_subsample_size', descending=False)
        .collect()
    )

    results_session_df = exclude_structures_from_df(
        results_session_df, 
        exclude_redundant_structures=exclude_redundant_structures, 
        exclude_general_structures=exclude_general_structures
    )

    return results_session_df



def get_average_session_structure_ccf_coords(results_session_df,all_units_table_path=None):
    """Calculate average CCF coordinates for each session-structure combination.
    
    Computes the mean Allen Institute Common Coordinate Framework (CCF) coordinates
    for recorded units in each brain structure for each session. This provides
    spatial localization information for decoder results.
    
    Parameters
    ----------
    results_session_df : pandas.DataFrame or polars.DataFrame
        Session-wise decoder results DataFrame. Must contain columns:
        - session_id : Session identifier
        - structure : Brain structure name
    all_units_table_path : str, optional
        Path to consolidated units table parquet file containing CCF coordinates.
        If None, uses default path:
        's3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/units.parquet'
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with average CCF coordinates:
        - session_id : Session identifier
        - structure : Brain structure name  
        - ccf_dv : Mean dorsal-ventral coordinate (µm from dorsal surface)
        - ccf_ml : Mean medial-lateral coordinate (µm from midline)
        - ccf_ap : Mean anterior-posterior coordinate (µm from bregma)
    
    Notes
    -----
    - Coordinates are averaged across all units recorded in each structure
    - Special handling for superior colliculus sub-regions:
      * 'SCs' queries for 'SCop|SCsg|SCzo' (superficial layers)
      * 'SCm' queries for 'SCig|SCiw|SCdg|SCdw' (intermediate/deep layers)
    - NaN values in unit coordinates are handled with np.nanmean
    - Input DataFrame will be converted to pandas if provided as polars
    
    Raises
    ------
    ValueError
        If results_session_df is not a pandas or polars DataFrame.
    
    Examples
    --------
    >>> coords_df = get_average_session_structure_ccf_coords(results_df)
    >>> # Plot decoding accuracy vs anatomical location
    >>> merged = results_df.merge(coords_df, on=['session_id', 'structure'])
    >>> plt.scatter(merged['ccf_ap'], merged['mean_diff'])
    """

    if type(results_session_df) == pd.DataFrame or type(results_session_df) == pl.DataFrame:
        if type(results_session_df) == pl.DataFrame:
            results_session_df = results_session_df.to_pandas()
    else:
        raise ValueError('results_session_df must be a pandas or polars DataFrame')
            
    if all_units_table_path is None:
        all_units_table_path='s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/units.parquet'
    all_units_table=pd.read_parquet(all_units_table_path)

    session_structure_ccf_coords={
    'session_id':[],
    'structure':[],
    'ccf_dv':[],
    'ccf_ml':[],
    'ccf_ap':[]
    }

    for session_id in results_session_df['session_id'].unique():
        session_df=results_session_df.query('session_id==@session_id')
        for structure in session_df['structure'].unique():
            #get SCm, SCs
            if structure=='SCs':
                structure_query='(structure=="SCop" | structure=="SCsg" | structure=="SCzo")'
            elif structure=='SCm':
                structure_query='(structure=="SCig" | structure=="SCiw" | structure=="SCdg" | structure=="SCdw")'
            else:
                structure_query='structure==@structure'
            session_structure_units=all_units_table.query('session_id==@session_id & '+structure_query)
            ccf_dv=np.nanmean(session_structure_units['ccf_dv'].values)
            ccf_ml=np.nanmean(session_structure_units['ccf_ml'].values)
            ccf_ap=np.nanmean(session_structure_units['ccf_ap'].values)

            session_structure_ccf_coords['session_id'].append(session_id)
            session_structure_ccf_coords['structure'].append(structure)
            session_structure_ccf_coords['ccf_dv'].append(ccf_dv)
            session_structure_ccf_coords['ccf_ml'].append(ccf_ml)
            session_structure_ccf_coords['ccf_ap'].append(ccf_ap)

    session_structure_ccf_coords_df=pd.DataFrame(session_structure_ccf_coords)

    return session_structure_ccf_coords_df
