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

import datetime
import os
import random
os.environ['RUST_BACKTRACE'] = '1'
#os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'

import concurrent.futures as cf
import contextlib
import itertools
import logging
import math
import multiprocessing
import random
import uuid
from typing import Annotated, Iterable, Literal, Sequence

import numpy as np
import polars as pl
import polars._typing
import pydantic_settings
import pydantic_settings.sources
import pydantic.functional_serializers
import tqdm
import upath
from sklearn.model_selection import StratifiedKFold
from dynamic_routing_analysis.decoding_utils import decoder_helper, NotEnoughBlocksError
from dynamic_routing_analysis import data_utils

import utils

logger = logging.getLogger(__name__)

class NotEnoughBlocksError(Exception):
    """Raised when there are insufficient blocks for blockwise cross-validation."""
    pass


# define run params here ------------------------------------------- #
Expr = Annotated[
    pl.Expr, pydantic.functional_serializers.PlainSerializer(lambda expr: expr.meta.serialize(format='json'), return_type=str)
]
class BinnedRelativeIntervalConfig(pydantic.BaseModel):
    event_column_name: str
    start_time: float 
    stop_time: float
    bin_size: float

    @property
    def intervals(self) -> list[tuple[float, float]]:
        start_times = np.arange(self.start_time, self.stop_time, self.bin_size)
        stop_times = start_times + self.bin_size
        return list(zip(start_times, stop_times))
    
def to_polars_expr(value: str | pl.Expr) -> Expr:
    if isinstance(value, pl.Expr):
        return value
    """Eval str to create pl.Expr instance"""
    if "pl." not in value:
        raise ValueError(f"Polars expression must access Polars objects under the `pl.` namespace {value=}")
    return eval(value)

class Params(pydantic_settings.BaseSettings):
    # ----------------------------------------------------------------------------------
    # Required parameters
    result_prefix: str
    "An identifier for the decoding run, used to name the output files (can have duplicates with different run_id)"
    # ----------------------------------------------------------------------------------
    
    # Capsule-specific parameters -------------------------------------- #
    session_id: str | None = pydantic.Field(None, exclude=True, repr=True)
    """If provided, only process this session_id. Otherwise, process all sessions that match the filtering criteria"""
    run_id: str = pydantic.Field(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) # created at runtime: same for all Params instances 
    """A unique string that should be attached to all decoding runs in the same batch"""
    skip_existing: bool = pydantic.Field(True, exclude=True, repr=True)
    test: bool = pydantic.Field(False, exclude=True)
    logging_level: str | int = pydantic.Field('INFO', exclude=True)
    update_packages_from_source: bool = pydantic.Field(False, exclude=True)
    override_params_json: str | None = pydantic.Field('{}', exclude=True)
    use_process_pool: bool = pydantic.Field(True, exclude=True, repr=True)
    max_workers: int | None = pydantic.Field(int(os.environ['CO_CPUS']), exclude=True, repr=True)
    """For process pool"""

    # Decoding parameters ----------------------------------------------- #
    session_table_query: str = "is_ephys & is_task & is_annotated & is_production & issues=='[]'"
    unit_criteria: str = pydantic.Field("loose_drift", exclude=True, repr=True) # often varied, stored in data not params file
    unit_subsample_size: int | None = pydantic.Field(None, exclude=True, repr=True) # often varied, so will be stored with data, not in the params file
    """number of units to sample for each area"""
    n_repeats: int = 25
    """number of times to repeat decoding with different randomly sampled units"""
    min_n_units: int = 20
    """only process areas with at least this many units"""
    input_data_type: Literal['spikes', 'facemap', 'LP'] = 'spikes'
    spikes_time_before: float = pydantic.Field(0.2, deprecated="Use time_interval_config instead")
    crossval: Literal['5_fold', 'blockwise', '5_fold_set_random_state', 'custom', 'leave_2_blocks_out', 'leave_2_blocks_out_adjacent',
    'leave_2_blocks_out_half_block_shifts','leave_1_half_block_out','leave_2_half_blocks_out_full_block_shifts',
    'leave_2_blocks_out_half_block_shifts_wraparound','5x_5_fold'] = '5_fold'
    """blockwise untested with linear shift"""
    labels_as_index: bool = True
    """convert labels (context names) to index [0,1]"""
    decoder_type: Literal['linearSVC', 'nonlinearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'
    regularization: float | None = None
    """ set regularization (C) for the decoder. Setting to None reverts to the default value (usually 1.0) """
    penalty: str | None = None
    """ set penalty for the decoder. Setting to None reverts to default """
    solver: str | None = None
    """ set solver for the decoder. Setting to None reverts to default """
    units_group_by: list[str] = ['session_id', 'structure', 'electrode_group_names']

    trials_filter: Annotated[str | Expr, pydantic.AfterValidator(to_polars_expr)] = pydantic.Field(default_factory = lambda:pl.lit(True))
    """ filter trials table input to decoder by boolean column or polars expression"""
    filter_units_by_metrics: bool = False
    """ option to filter units by activity metrics, defined in unit_metrics_filter """
    unit_metrics_filter: Annotated[str | Expr, pydantic.AfterValidator(to_polars_expr)] = pydantic.Field(default_factory = lambda:pl.lit(True))
    """ filter units table input to decoder by boolean column or polars expression"""
    label_to_decode: str = 'rewarded_modality'
    """ designate label to decode; corresponds to column in the trials table"""
    spike_count_intervals: Literal['pre_stim_single_bin', 'binned_stim_and_response', 'pre_stim_single_bin_0.5', 'pre_stim_single_bin_1.5', 'binned_stim_and_response_0.025', 'binned_stim_and_response_0.5','binned_stim_0.5','binned_stim_0.1','binned_stim_0.05','binned_stim_only_0.05','binned_stim_only_0.025','binned_stim_only_0.02','binned_stim_only_0.01','binned_stim_only_0.005','binned_stim_onset_only_0.01','binned_stim_onset_only_0.005','binned_response_0.025','binned_prestim_0.1'] = 'pre_stim_single_bin'
    baseline_subtraction: bool = False
    """whether to subtract the average baseline context modulation from each unit/trial"""
    n_blocks_expected: int = 6
    """ set number of blocks expected - defaults to 6"""
    use_cumulative_spike_counts: bool = False
    """ toggle using cumulative spike counts from start of first interval for decoding"""
    sliding_window_size: float | None = None
    """ set sliding time window size if different from step size in spike_count_intervals """
    linear_shift: bool = True
    """ toggle linear shift (if False, only runs decoding on aligned trials/ephys) """
    test_on_spontaneous: bool = False
    """ toggle testing the decoder model (trained on the full task) on spontaneous activity """
    scaler: Literal['robust','standard','none'] = 'robust'
    """ set data scaling method: standard = mean/stdev, robust = median/iqr, none = do not scale """
    test_across_context: bool = False
    """ toggle training decoder model on one context and testing on the other. Requires decoding something other than context, i.e. stimulus id """
    save_all_coefs: bool = False
    """ toggle saving decoder coefficients across all train/test folds """

    @property
    def data_path(self) -> upath.UPath:
        """Path to delta lake on S3"""
        return upath.UPath("s3://aind-scratch-data/dynamic-routing/decoding/results") /f"{'_'.join([self.result_prefix, self.run_id])}"

    @property
    def json_path(self) -> upath.UPath:
        """Path to params json on S3"""
        return self.data_path.with_suffix('.json')

    @property
    def min_n_units_query(self) -> Expr:
        if self.unit_subsample_size is None:
            min_ = self.min_n_units
        else:
            #determine number of units required based on max of (min_n_units, unit_subsample_size)
            min_ = max(self.min_n_units,self.unit_subsample_size)
        return pl.col('unit_id').n_unique().over(self.units_group_by).ge(min_)

    @pydantic.computed_field(repr=False)
    @property
    def units_query(self) -> Expr:
        drift_base = (pl.col('decoder_label') != "noise") & (pl.col('isi_violations_ratio') <= 0.5) & (pl.col('amplitude_cutoff') <= 0.1) & (pl.col('presence_ratio') >= 0.7)
        return {
            'medium': (pl.col('isi_violations_ratio') <= 0.5) & (pl.col('presence_ratio') >= 0.9) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'strict': (pl.col('isi_violations_ratio') <= 0.1) & (pl.col('presence_ratio') >= 0.99) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'use_sliding_rp': (pl.col('sliding_rp_violation') <= 0.1) & (pl.col('presence_ratio') >= 0.99) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'recalc_presence_ratio': (pl.col('sliding_rp_violation') <= 0.1) & (pl.col('presence_ratio_task') >= 0.99) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'no_drift': drift_base,
            
            'loose_drift': drift_base & (pl.col('activity_drift') <= 0.2),
            
            'medium_drift': drift_base & (pl.col('activity_drift') <= 0.15),
            
            'strict_drift': drift_base & (pl.col('activity_drift') <= 0.1),
        }[self.unit_criteria]
    
    @pydantic.computed_field(repr=False)
    @property
    def spike_count_interval_configs(self) -> list[BinnedRelativeIntervalConfig]:
        return {
            'pre_stim_single_bin': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.2,
                    stop_time=0,
                    bin_size=0.2,
                ),
            ],
            'pre_stim_single_bin_0.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.5,
                    stop_time=0,
                    bin_size=0.5,
                ),
            ],
            'pre_stim_single_bin_1.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=0,
                    bin_size=1.5,
                ),
            ],
            'binned_stim_and_response': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.4,
                    stop_time=2.0,
                    bin_size=0.2,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name='response_time',
                    start_time=-0.4,
                    stop_time=2.0,
                    bin_size=0.2,
                ),
            ],
            'binned_stim_and_response_0.025': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=2.0,
                    bin_size=0.025,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name='response_or_reward_time',
                    start_time=-1.0,
                    stop_time=1.0,
                    bin_size=0.025,
                ),
            ],
            'binned_stim_and_response_0.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.5,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name='response_or_reward_time',
                    start_time=-2.5,
                    stop_time=4.5,
                    bin_size=0.5,
                ),
            ],
            'binned_stim_0.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.5,
                ),
            ],
            'binned_stim_0.1': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.1,
                ),
            ],
            'binned_stim_0.05': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.05,
                ),
            ],
            'binned_stim_only_0.05': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.05,
                ),
            ],
            'binned_stim_only_0.025': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.025,
                ),
            ],
            'binned_stim_only_0.02': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.02,
                ),
            ],
            'binned_stim_only_0.01': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.01,
                ),
            ],
            'binned_stim_only_0.005': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.005,
                ),
            ],
            'binned_stim_onset_only_0.01': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.05,
                    stop_time=0.3,
                    bin_size=0.01,
                ),
            ],
            'binned_stim_onset_only_0.005': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.05,
                    stop_time=0.3,
                    bin_size=0.005,
                ),
            ],
            'binned_response_0.025': [
                BinnedRelativeIntervalConfig(
                    event_column_name='response_or_reward_time',
                    start_time=-1.0,
                    stop_time=1.0,
                    bin_size=0.025,
                ),
            ],
            'binned_prestim_0.1': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=0.0,
                    bin_size=0.1,
                ),
            ],
        }[self.spike_count_intervals]

    @pydantic.computed_field(repr=False)
    def datacube_version(self) -> str:
        return utils.get_datacube_dir().name.split('_')[-1]
        
    # set the priority of the sources:
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )
        
# end of run params ------------------------------------------------ #


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

    elif crossval=='leave_2_blocks_out':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        block_numbers=np.unique(block_number)
        #leave each pair of adjacent blocks out for testing, including the first and last as a pair
        for bb in block_numbers:
            not_block_inds=np.where((block_number!=bb) & (block_number!=bb+1))[0]
            train.append(not_block_inds)
            block_inds=np.where((block_number==bb) | (block_number==bb+1))[0]
            test.append(block_inds)
        #last and first block as a pair
        not_block_inds=np.where((block_number!=block_numbers[-1]) & (block_number!=block_numbers[0]))[0]
        train.append(not_block_inds)
        block_inds=np.where((block_number==block_numbers[-1]) | (block_number==block_numbers[0]))[0]
        test.append(block_inds)
        train_test_split=zip(train,test)

        ypred_proba_all=[]
        decision_function_all=[]

    elif crossval=='leave_2_blocks_out_adjacent':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        block_numbers=np.unique(block_number)
        #leave each pair of adjacent blocks out for testing, but do not include the first and last blocks as a pair
        for bb in block_numbers:
            not_block_inds=np.where((block_number!=bb) & (block_number!=bb+1))[0]
            train.append(not_block_inds)
            block_inds=np.where((block_number==bb) | (block_number==bb+1))[0]
            test.append(block_inds)
        train_test_split=zip(train,test)

        ypred_proba_all=[]
        decision_function_all=[]

    elif crossval=='leave_2_blocks_out_half_block_shifts' or crossval=='leave_2_blocks_out_half_block_shifts_wraparound':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        #find indices for block numbers, label second half of block as +0.5
        new_block_number=np.copy(block_number).astype(float)
        block_numbers=np.unique(block_number)
        for bb in block_numbers:
            block_inds=np.where(block_number==bb)[0]
            if len(block_inds)>0:
                half_point=block_inds[len(block_inds)//2]
                new_block_number[half_point:block_inds[-1]+1]=bb+0.5
        new_block_numbers=np.unique(new_block_number)

        for bb in new_block_numbers:
            if bb+1.5 not in new_block_numbers:
                continue
            not_block_inds=np.where((new_block_number!=bb) & (new_block_number!=bb+0.5) & (new_block_number!=bb+1.0) & (new_block_number!=bb+1.5))[0]
            train.append(not_block_inds)
            block_inds=np.where((new_block_number==bb) | (new_block_number==bb+0.5) | (new_block_number==bb+1.0) | (new_block_number==bb+1.5))[0]
            test.append(block_inds)

        if 'wraparound' in crossval:
            # get wraparound train and test sets
            for bb in [0,1,2]:
                not_block_inds=np.where(
                    (new_block_number!=new_block_numbers[-3+bb]) & (new_block_number!=new_block_numbers[-2+bb]) &
                    (new_block_number!=new_block_numbers[-1+bb]) & (new_block_number!=new_block_numbers[0+bb]))[0]
                train.append(not_block_inds)
                block_inds=np.where(
                    (new_block_number==new_block_numbers[-3+bb]) |(new_block_number==new_block_numbers[-2+bb]) | 
                    (new_block_number==new_block_numbers[-1+bb]) | (new_block_number==new_block_numbers[0+bb]))[0]
                test.append(block_inds)
        
        train_test_split=zip(train,test)

        ypred_proba_all=[]
        decision_function_all=[]

    elif crossval=='leave_1_half_block_out':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        #find indices for block numbers, label second half of block as +0.5
        new_block_number=np.copy(block_number).astype(float)
        block_numbers=np.unique(block_number)
        for bb in block_numbers:
            block_inds=np.where(block_number==bb)[0]
            if len(block_inds)>0:
                half_point=block_inds[len(block_inds)//2]
                new_block_number[half_point:block_inds[-1]+1]=bb+0.5
        new_block_numbers=np.unique(new_block_number)

        for bb in new_block_numbers:
            not_block_inds=np.where((new_block_number!=bb))[0]
            train.append(not_block_inds)
            block_inds=np.where((new_block_number==bb))[0]
            test.append(block_inds)
        train_test_split=zip(train,test)

    elif crossval=='leave_2_half_blocks_out_full_block_shifts':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        #find indices for block numbers, label second half of block as +0.5
        new_block_number=np.copy(block_number).astype(float)
        block_numbers=np.unique(block_number)
        for bb in block_numbers:
            block_inds=np.where(block_number==bb)[0]
            if len(block_inds)>0:
                half_point=block_inds[len(block_inds)//2]
                new_block_number[half_point:block_inds[-1]+1]=bb+0.5
        new_block_numbers=np.unique(new_block_number)

        loop_block_numbers=block_numbers+0.5

        for bb in loop_block_numbers[:-1]:
            not_block_inds=np.where((new_block_number!=bb) & (new_block_number!=bb+0.5))[0]
            train.append(not_block_inds)
            block_inds=np.where((new_block_number==bb) | (new_block_number==bb+0.5))[0]
            test.append(block_inds)
        #add first and last half-block
        not_block_inds=np.where(
            (new_block_number!=loop_block_numbers[-1]) & (new_block_number!=block_numbers[0]))[0]
        train.append(not_block_inds)
        block_inds=np.where(
            (new_block_number==loop_block_numbers[-1]) |(new_block_number==block_numbers[0]))[0]
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

    elif crossval=='5x_5_fold':
        train=[]
        test=[]
        n_repeats=5
        for rr in range(n_repeats):
            skf = StratifiedKFold(n_splits=5,shuffle=True)
            train_test_split_temp = skf.split(input_data, labels)
        for temp_train, temp_test in train_test_split_temp:
                train.extend(temp_train)
                test.extend(temp_test)

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

    #list of crossval strategies where I want to take a mean over all arbitrary folds
    mean_over_folds_list = ['leave_2_blocks_out', 'leave_2_blocks_out_adjacent', 'leave_2_blocks_out_half_block_shifts',
                            'leave_2_blocks_out_half_block_shifts_wraparound', 'leave_2_half_blocks_out_full_block_shifts',
                            '5x_5_fold']

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
            if crossval in mean_over_folds_list:
                temp_ypred_proba = np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
                temp_ypred_proba[test,:] = clf.predict_proba(X[test])
                ypred_proba_all.append(temp_ypred_proba)
            else:
                ypred_proba[test,:] = clf.predict_proba(X[test])
        else:
            if crossval in mean_over_folds_list:
                ypred_proba_all.append(np.full((len(y),len(np.unique(labels))), fill_value=False))
            else:
                ypred_proba[test,:] = np.full((len(test),len(np.unique(labels))), fill_value=False)

        if decoder_type=='LDA' or decoder_type=='linearSVC' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
            if crossval in mean_over_folds_list:
                if len(np.unique(labels))>2:
                    temp_decision_function = np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
                    temp_decision_function[test,:] = clf.decision_function(X[test])
                else:
                    temp_decision_function = np.full(len(y), fill_value=np.nan)
                    temp_decision_function[test] = clf.decision_function(X[test])
                decision_function_all.append(temp_decision_function)
            else:
                if len(np.unique(labels))>2:
                    decision_function[test,:]=clf.decision_function(X[test])
                else:
                    decision_function[test]=clf.decision_function(X[test])
        else:
            if crossval in mean_over_folds_list:
                if len(np.unique(labels))>2:
                    decision_function_all.append(np.full((len(y),len(np.unique(labels))), fill_value=False))
                else:
                    decision_function_all.append(np.full((len(y)), fill_value=False))
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
    
    #takes mean over all repeated crossvals for each trial
    if crossval in mean_over_folds_list:
        ypred_proba=np.nanmean(np.stack(ypred_proba_all, axis=2),axis=2)
        if len(np.unique(labels))>2:
            decision_function=np.nanmean(np.stack(decision_function_all, axis=2),axis=2)
        else:
            decision_function=np.nanmean(np.stack(decision_function_all, axis=1),axis=1)

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
    output['balanced_accuracy_test_all']=np.hstack(balanced_accuracy_test)

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
        if type(df) == pl.DataFrame:
            df = df.filter(~pl.col('structure').is_in(redundant_structures))
        elif type(df) == pd.DataFrame:
             df = df.query("~structure.isin(@redundant_structures)")
    if exclude_general_structures:
        if type(df) == pl.DataFrame:
            df = df.filter(~pl.col('structure').is_in(general_structures))
        elif type(df) == pd.DataFrame:
            df = df.query("~structure.isin(@general_structures)")
        #also remove any structure beginning with lowercase letter - indicating fiber tracts or out of brain
        unique_structures=df['structure'].unique()
        lowercase_structures=[]
        for ss in unique_structures:
            if ss[0].islower():
                lowercase_structures.append(ss)
        if len(lowercase_structures)>0:
            if type(df) == pl.DataFrame:
                df = df.filter(~pl.col('structure').is_in(lowercase_structures))
            elif type(df) == pd.DataFrame:
                df = df.query("~structure.isin(@lowercase_structures)")

    return df

def load_single_session_decoder_accuracy(results_path, sel_session, combine_multi_probe_rec=True, use_linear_shift=True):
    """Load decoder accuracy results for a single session.
    
    Loads and processes decoder accuracy results from parquet files for one session,
    including handling of multi-probe recordings and temporal shift analysis.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
    sel_session : str
        Session ID to load (e.g., '742903_2024-10-22').
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
    ...     '742903_2024-10-22'
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

    if use_linear_shift:

        example_session_df = (
            pl.scan_parquet(results_path)
            .with_columns(
                pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
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

    else:

        example_session_df = (
            pl.scan_parquet(results_path)
            .with_columns(
                pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
            )
            .filter(
                pl.col('session_id').eq(sel_session),
                combine_multi_probe_expr,
                pl.col('is_all_trials'),
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



def load_structure_average_decoder_accuracy(results_path, session_list, combine_multi_probe_rec=True, 
                                            exclude_redundant_structures=True, exclude_general_structures=True,
                                            use_linear_shift=True):
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

    if use_linear_shift:
        #use linear shift results
        results_df = (
            pl.scan_parquet(results_path)
            .filter(
                pl.col('session_id').is_in(session_list),
            )
            .with_columns(
                pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
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

    else:
        # no linear shift, use decoder results from all trials
        results_df = (
            pl.scan_parquet(results_path)
            .filter(
                pl.col('session_id').is_in(session_list),
            )
            .with_columns(
                pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
            )
            .filter(
                combine_multi_probe_expr,
                pl.col('is_all_trials'),
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
                pl.col('balanced_accuracy_test').first().alias('mean_true'),

            )
            # get the means over sessions:
            .group_by('structure', 'unit_subsample_size','bin_size','bin_center')#, 'unit_criteria')
            .agg(
                pl.col('mean_true').mean(),
                pl.col('mean_true').std().truediv(pl.col('mean_true').count().pow(0.5)).alias('sem_true'),
                pl.col('session_id').n_unique().alias('num_sessions')
            )
            .with_columns(
                pl.col('num_sessions').cast(pl.Int64),
            )
            .sort(pl.col('mean_true').mean().over('structure'), descending=True)
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
        exclude_redundant_structures=True, exclude_general_structures=True,
        is_all_trials=False, use_linear_shift=True):
    """Load decoder accuracy with session-level metadata.
    
    Loads decoder results for multiple sessions and enriches them with behavioral
    and recording metadata from the session table, including unit counts,
    cross-modal d-prime, and block level behavior metrics.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3.
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
        - median_null : Median null accuracy from linear shifts
        - mean_diff : Difference (true - null) for this session
        - balanced_accuracy_test : Decoder accuracy on held out test data across shifts
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
        'time_aligned_to',
    }
    
    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)
    structure_grouping, n_repeats = get_structure_grouping(keep_original_structure)

    if use_linear_shift==True:

        #add total n units, cross-modal dprime, n good blocks?
        results_session_df = (
            pl.scan_parquet(results_path)
            .filter(
                pl.col('session_id').is_in(session_list),
            )
            .with_columns(
                pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
            )
            .filter(
                combine_multi_probe_expr,
                pl.col('is_all_trials').eq(is_all_trials),
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
            .sort('session_id', 'structure', 'unit_subsample_size', 'bin_center', descending=False)
            .collect()
        )

    else:
        
        #add total n units, cross-modal dprime, n good blocks?
        results_session_df = (
            pl.scan_parquet(results_path)
            .filter(
                pl.col('session_id').is_in(session_list),
            )
            .with_columns(
                pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
            )
            .filter(
                combine_multi_probe_expr,
                pl.col('is_all_trials').eq(True),
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
                pl.col('balanced_accuracy_test').first().alias('mean_true'),
            )
            .sort('session_id', 'structure', 'unit_subsample_size', 'bin_center', descending=False)
            .collect()
        )

    results_session_df = exclude_structures_from_df(
        results_session_df, 
        exclude_redundant_structures=exclude_redundant_structures, 
        exclude_general_structures=exclude_general_structures
    )

    return results_session_df


def load_single_session_decoder_confidence(results_path, sel_session, combine_multi_probe_rec=True, predict_proba_alias='predict_proba'):
    """Load decoder confidence (predicted probabilities) for a single session.
    
    Loads trial-by-trial decoder predictions and enriches them with trial metadata
    including stimulus type, block structure, and behavioral responses. This provides
    detailed insight into decoder performance across different task conditions.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
    sel_session : str
        Session ID to load (e.g., '742903_2024-10-22').
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    predict_proba_alias : str, default='predict_proba'
        Column name in results that contains predicted probabilities. Default is 'predict_proba',
        but can be overridden if a different column name is used in the results i.e. in multiclass decoding
    
    Returns
    -------
    polars.DataFrame
        DataFrame with trial-level decoder predictions:
        - session_id : Session identifier
        - structure : Brain structure name
        - bin_center : Center time of temporal bin relative to event
        - unit_subsample_size : Number of units used for decoding
        - repeat_idx : Index of repeated unit resample and decoder run 
        - unit_ids : List of unit IDs used in this decoder
        - balanced_accuracy_test : Overall decoder accuracy on held out test data
        - total_n_units : Total units recorded in this structure
        - predict_proba : Predict probability for each trial (list)
        - trial_index : Trial indices (list)
        - is_vis_rewarded : Whether visual stimuli were rewarded (list)
        - stim_name : Stimulus identity for each trial (list)
        - is_response : Whether animal responded (list)
        - trial_index_in_block : Trial number within block (list)
        - block_index : Block number (list)
        - stim_start_time : Stimulus onset time in seconds (list)
        - decision_function : Decision function values (if available)
        
        Sorted by session_id, structure, unit_subsample_size, repeat_idx, and bin_center.
    
    Notes
    -----
    - Only includes results where is_all_trials is True (using full dataset)
    - Trial indices are taken from the 'trial_indices' column in decoder results
    - Joins with consolidated units table to get total_n_units for each structure
    - Joins with trials table to add task information for each trial
    - predict_proba contains probabilities for the positive class
    - If decision_function is present in results, it will be included in output
    
    Examples
    --------
    >>> df = load_single_session_decoder_confidence(
    ...     's3://bucket/results/',
    ...     '742903_2024-10-22'
    ... )
    """

    #define grouping columns - maintain compatibility with older results
    col_names=pl.scan_parquet(results_path)

    grouping_cols = {
        'session_id',
        'structure',
        'bin_center',
        'electrode_group_names',
        'unit_subsample_size',
        'unit_criteria',
        'repeat_idx', 
        'unit_ids'
    }

    if 'labels' in col_names:
        grouping_cols.add('labels')

    final_agg_cols = {
        predict_proba_alias,  
        'trial_index', 
        'is_vis_rewarded', 
        'stim_name', 
        'is_response', 
        'trial_index_in_block', 
        'block_index', 
        'stim_start_time'
    }

    explode_cols={
        predict_proba_alias,
        'trial_index',
    }

    if 'decision_function' in col_names:
        final_agg_cols.add('decision_function')
        explode_cols.add('decision_function')

    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    decoder_confidence_with_repeats_single_session = (
        pl.scan_parquet(results_path)
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),     
        )
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials'),
            pl.col('session_id').eq(sel_session),
        )
        .join(
            other=(
                pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/units.parquet')
                .with_columns(
                    pl.col('session_id').str.split('_').list.slice(0, 2).list.join('_'),
                )
                .group_by('session_id','structure')
                .agg(
                    pl.col('unit_id').len().alias('total_n_units')
                )
            ),
            on=['session_id','structure'],
            how='inner',
        )
        .with_columns(
            pl.col('trial_indices').alias('trial_index')
        )
        .drop('shift_idx', 'is_all_trials', 'electrode_group_names', 'unit_criteria', 'is_sole_recording')
        .explode(explode_cols)
        .join(
            other=(
                pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/trials.parquet')
                .with_columns(
                    pl.col('session_id').str.split('_').list.slice(0, 2).list.join('_'),
                )
                .select('session_id', 'trial_index', 'is_vis_rewarded', 'stim_name', 'is_response', 'trial_index_in_block', 'block_index', 'stim_start_time',)
            ),
            on=['session_id','trial_index'],
            how='inner',
        ) 
        .group_by(grouping_cols - {'electrode_group_names', 'unit_criteria'})
        .agg(
            pl.col('balanced_accuracy_test', 'total_n_units').first(),
            pl.col(final_agg_cols),
        )
        .sort('session_id','structure', 'unit_subsample_size', 'repeat_idx', 'bin_center')
        .collect()
    )

    return decoder_confidence_with_repeats_single_session

def load_single_session_decoder_confidence_spont_epoch(results_path, sel_session, combine_multi_probe_rec=True, predict_proba_alias='predict_proba_spont'):
    """Load decoder predictions for spontaneous (non-task) epochs in a single session.
    
    Applies trained decoders to spontaneous activity epochs to assess whether context 
    representations persist outside of active task engagement.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
        Must contain 'predict_proba_spont' or 'decision_function_spont' columns.
    sel_session : str
        Session ID to load (e.g., '742903_2024-10-22').
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    predict_proba_alias : str, default='predict_proba_spont'
        Column name in results that contains predicted probabilities for spontaneous epochs.
    
    Returns
    -------
    polars.DataFrame
        DataFrame with spontaneous epoch predictions:
        - session_id : Session identifier
        - structure : Brain structure name
        - bin_center : Center time of temporal bin relative to event
        - unit_subsample_size : Number of units used for decoding
        - repeat_idx : Index of repeated decoder run
        - unit_ids : List of unit IDs used in this decoder
        - balanced_accuracy_test : Test accuracy on task trials
        - total_n_units : Total units recorded in this structure
        - predict_proba_spont : Predicted probabilities for spontaneous epochs (list)
        - trial_index : Pseudo-trial indices for spontaneous epochs (list)
        - pred_label_spont : Predicted labels for spontaneous epochs (list)
        - spont_trial_times : Time stamps for spontaneous epochs (list)
        - spont_epoch_name : Epoch type (i.e., 'Spontaneous') (list)
        - spont_trial_is_rewarded : Whether a free reward was delivered on this trial (list)
        - decision_function_spont : Decision values (if available)
        
        Sorted by session_id, structure, unit_subsample_size, repeat_idx, and bin_center.
    
    Raises
    ------
    ValueError
        If neither 'predict_proba_spont' nor 'decision_function_spont' columns are
        found in the results file.
    
    Notes
    -----
    - Spontaneous epochs are occur outside of the task epoch
    - trial_index is generated as a sequential index, not linked to actual trials
    - spont_trial_is_rewarded indicates whether a free reward was delivered during this spontaneous epoch
    - Decoders are trained on task trials, then applied to spontaneous periods
    
    Examples
    --------
    >>> df = load_single_session_decoder_confidence_spont_epoch(
    ...     's3://bucket/results/',
    ...     '742903_2024-10-22'
    ... )
    """

    #define grouping columns - maintain compatibility with older results
    col_names=pl.scan_parquet(results_path)

    if predict_proba_alias not in col_names and 'decision_function_spont' not in col_names:
        raise ValueError(f"Neither '{predict_proba_alias}' nor 'decision_function_spont' columns found in the results. Please check the results file for the expected columns.")

    grouping_cols = {
        'session_id',
        'structure',
        'bin_center',
        'electrode_group_names',
        'unit_subsample_size',
        'unit_criteria',
        'repeat_idx', 
        'unit_ids'
    }

    if 'labels' in col_names:
        grouping_cols.add('labels')

    final_agg_cols = {
        predict_proba_alias,  
        'trial_index',
        'pred_label_spont',
        'spont_trial_times',
        'spont_epoch_name',
        'spont_trial_is_rewarded'
    }

    explode_cols={
        predict_proba_alias,
        'trial_index',
        'pred_label_spont',
        'spont_trial_times',
        'spont_epoch_name',
        'spont_trial_is_rewarded',
    }

    if 'decision_function_spont' in col_names:
        final_agg_cols.add('decision_function_spont')
        explode_cols.add('decision_function_spont')

    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    decoder_confidence_with_repeats_single_session_spontaneous = (
        pl.scan_parquet(results_path)
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
        )
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials'),
            pl.col('session_id').eq(sel_session),
        )#.drop('unit_ids') 
        # .sort('session_id', descending=True)
        .join(
            other=(
                pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/units.parquet')
                .with_columns(
                    pl.col('session_id').str.split('_').list.slice(0, 2).list.join('_'),
                )
                .group_by('session_id','structure')
                .agg(
                    pl.col('unit_id').len().alias('total_n_units')
                )
            ),
            on=['session_id','structure'],
            how='inner',
        )
        .with_columns(
            pl.int_ranges(0, pl.col(predict_proba_alias).list.len()).alias('trial_index')
        )
        .drop('shift_idx', 'is_all_trials', 'electrode_group_names', 'unit_criteria', 'is_sole_recording')
        .explode(explode_cols)
        .group_by(grouping_cols - {'electrode_group_names', 'unit_criteria'})
        .agg(
            pl.col('balanced_accuracy_test', 'total_n_units').first(),
            pl.col(final_agg_cols),
        )
        .sort('session_id','structure', 'unit_subsample_size', 'repeat_idx', 'bin_center')
        # .group_by('session_id','structure')
        .collect()
    )

    return decoder_confidence_with_repeats_single_session_spontaneous


def load_session_wise_decoder_confidence(
        results_path, session_list, combine_multi_probe_rec=True, 
        exclude_redundant_structures=True, exclude_general_structures=True, 
        predict_proba_alias='predict_proba'):
    """Load decoder confidence across multiple sessions with trial-level predictions.
    
    Aggregates trial-by-trial decoder predictions across sessions, averaging over
    repeated decoder runs while maintaining trial-level resolution. Useful for
    population-level analyses of decoder confidence patterns.
    
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
    predict_proba_alias : str, default='predict_proba'
        Column name for predicted probabilities.
    
    Returns
    -------
    polars.DataFrame
        DataFrame with session-wise trial predictions:
        - session_id : Session identifier
        - structure : Brain structure name
        - unit_subsample_size : Number of units used for decoding
        - bin_center : Center time of temporal bin relative to event
        - bin_size : Size of time bins in seconds
        - time_aligned_to : Event to which time bins are aligned
        - balanced_accuracy_test : Mean test accuracy across repeats
        - predict_proba : Mean predicted probabilities across repeats (list per trial)
        - trial_index : Trial indices (list)
        - is_vis_rewarded : Whether visual context is rewarded (list)
        - stim_name : Stimulus identity for each trial (list)
        - is_response : Whether animal responded (list)
        - trial_index_in_block : Trial number within block (list)
        - block_index : Block number (list)
        - stim_start_time : Stimulus onset time in seconds (list)
        - decision_function : Mean decision values across repeats (if available)
        
        Sorted by session_id, structure, unit_subsample_size, and bin_center.
    
    Notes
    -----
    Processing pipeline:
    1. Filters to specified sessions and multi-probe handling
    2. Keeps only rows where is_all_trials is True
    3. Explodes list columns to individual trial rows for averaging
    4. Averages predict_proba and decision_function across repeated decoder runs
    5. Joins with trials table to add behavioral metadata
    6. Re-aggregates to maintain trial lists within session-structure-subsample groups
    7. Filters out redundant and general structures
    
    The averaging step (step 4) reduces variability from unit subsampling while
    preserving trial-by-trial prediction patterns.
    
    Examples
    --------
    >>> df = load_session_wise_decoder_confidence(
    ...     's3://bucket/results/',
    ...     ['session1', 'session2'],
    ...     combine_multi_probe_rec=True
    ... )
    """
    
    col_names=pl.scan_parquet(results_path)

    grouping_cols = {
        'session_id',
        'structure',
        'bin_center',
        'bin_size',
        'electrode_group_names',
        'unit_subsample_size',
        'unit_criteria',
        'time_aligned_to',
        'trial_index',
    }

    if 'labels' in col_names:
        grouping_cols.add('labels')

    final_agg_cols = {
        predict_proba_alias,  
        'trial_index', 
        'is_vis_rewarded', 
        'stim_name', 
        'is_response', 
        'trial_index_in_block',
        'block_index',
        'stim_start_time',
    }

    explode_cols={
        predict_proba_alias,
        'trial_index',
    }

    explode_agg_expr=(
        pl.col('balanced_accuracy_test').mean(),
        pl.col(predict_proba_alias).mean(),
    )

    if 'decision_function' in col_names:
        final_agg_cols.add('decision_function')
        explode_cols.add('decision_function')
        explode_agg_expr += (pl.col('decision_function').mean(),)

    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    decoder_confidence_df = (
        pl.scan_parquet(results_path,extra_columns='ignore')
        .filter(
            pl.col('session_id').is_in(session_list),
        )
        #make new column that indicates whether a row is the sole recording from a structure in a session
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),
        )
        #Grab only rows according to combine_multi_probe_rec toggle
        #Grab only rows that have is_all_trials == True, only these have predict_proba
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials'),
        )
        .with_columns(
            pl.col('trial_indices').alias('trial_index')
        )
        .drop('shift_idx', 'is_all_trials', 'electrode_group_names', 'unit_criteria', 'is_sole_recording')
        .explode(explode_cols)
        .group_by(grouping_cols - {'electrode_group_names', 'unit_criteria'})
        .agg(explode_agg_expr)
        .join(
            other=(
                pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/trials.parquet')
                .with_columns(
                    pl.col('session_id').str.split('_').list.slice(0, 2).list.join('_'),
                    #iti column?
                )
                .select(
                    'session_id', 'trial_index', 'is_vis_rewarded', 'stim_name', 'is_response', 
                    'trial_index_in_block', 'block_index', 'stim_start_time')
            ),
            on=['session_id','trial_index'],
            how='inner',
        ) 
        .group_by(grouping_cols - {'electrode_group_names', 'unit_criteria', 'trial_index'})
        .agg(
            pl.col('balanced_accuracy_test').first(),
            pl.col(final_agg_cols).sort_by('trial_index'),

        )
        .sort('session_id','structure', 'unit_subsample_size', 'bin_center')
        .collect(engine='streaming')
    )

    decoder_confidence_df = exclude_structures_from_df(
        decoder_confidence_df, 
        exclude_redundant_structures=exclude_redundant_structures, 
        exclude_general_structures=exclude_general_structures
    )

    return decoder_confidence_df


def load_session_wise_decoder_confidence_spont_epoch(
        results_path, session_list, combine_multi_probe_rec=True, 
        exclude_redundant_structures=True, exclude_general_structures=True,
        predict_proba_alias='predict_proba_spont'):
    """Load decoder predictions for spontaneous epochs across multiple sessions.
    
    Aggregates decoder predictions during spontaneous (non-task) periods across
    sessions, averaging over repeated decoder runs. Enables population-level analyses
    of context representations during spontaneous activity.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
        Must contain 'predict_proba_spont' or 'decision_function_spont' columns.
    session_list : list of str
        List of session IDs to include in analysis.
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    exclude_redundant_structures : bool, default=True
        If True, exclude over-split structures (e.g., superior colliculus sub-layers).
    exclude_general_structures : bool, default=True
        If True, exclude general anatomical regions and non-neural tissue.
    predict_proba_alias : str, default='predict_proba_spont'
        Column name for predicted probabilities during spontaneous epochs.
    
    Returns
    -------
    polars.DataFrame
        DataFrame with session-wise spontaneous epoch predictions:
        - session_id : Session identifier
        - structure : Brain structure name
        - unit_subsample_size : Number of units used for decoding
        - bin_center : Center time of temporal bin relative to event
        - bin_size : Size of time bins in seconds
        - time_aligned_to : Event to which time bins are aligned
        - spont_trial_times : Time stamps for spontaneous epochs (list)
        - spont_epoch_name : Epoch type (e.g., 'Spontaneous') (list)
        - spont_trial_is_rewarded : Context during spontaneous epoch (list)
        - predict_proba_spont : Mean predicted probabilities across repeats (list)
        - trial_index : Pseudo-trial indices (list)
        - decision_function_spont : Mean decision values across repeats (if available)
        
        Sorted by session_id, structure, unit_subsample_size, bin_center, bin_size,
        and time_aligned_to.
    
    Raises
    ------
    ValueError
        If neither 'predict_proba_spont' nor 'decision_function_spont' columns are
        found in the results file.
    
    Notes
    -----
    Processing pipeline:
    1. Filters to specified sessions and multi-probe handling
    2. Keeps only rows where is_all_trials is True
    3. Generates sequential trial_index for spontaneous epochs
    4. Explodes list columns to individual epoch rows
    5. Averages predict_proba_spont and decision_function_spont across repeats
    6. Re-aggregates to lists within session-structure-subsample groups
    7. Filters out redundant and general structures
    
    Drops several columns not relevant to spontaneous analysis including:
    - Regular task predictions (predict_proba, decision_function)
    - Training metrics (balanced_accuracy_train)
    - Unit identifiers (unit_ids)
    - Model coefficients (coefs)
    
    Examples
    --------
    >>> df = load_session_wise_decoder_confidence_spont_epoch(
    ...     's3://bucket/results/',
    ...     ['session1', 'session2']
    ... )
    """

    #define grouping columns - maintain compatibility with older results
    col_names=pl.scan_parquet(results_path)

    if 'predict_proba_spont' not in col_names and 'decision_function_spont' not in col_names:
        raise ValueError("Neither 'predict_proba_spont' nor 'decision_function_spont' columns found in the results. Please check the results file for the expected columns.")

    grouping_cols = {
        'session_id',
        'structure',
        'bin_center',
        'bin_size',
        'electrode_group_names',
        'unit_subsample_size',
        'unit_criteria',
        'time_aligned_to',
    }

    final_agg_cols = {
        predict_proba_alias,  
        'trial_index',
    }

    explode_cols={
        predict_proba_alias,
        'trial_index',
    }

    explode_agg_expr=(
        pl.col(predict_proba_alias).mean(),
        pl.col('spont_trial_times').first(),
        pl.col('spont_epoch_name').first(),
        pl.col('spont_trial_is_rewarded').first(),
    )

    if 'decision_function_spont' in col_names:
        final_agg_cols.add('decision_function_spont')
        explode_cols.add('decision_function_spont')
        explode_agg_expr += (pl.col('decision_function_spont').mean(),)

    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    decoder_confidence_spont_df = (
        pl.scan_parquet(results_path)
        #make new column that indicates whether a row is the sole recording from a structure in a session
        .filter(
            pl.col('session_id').is_in(session_list),
        )
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),     
        )
        #Grab only rows according to combine_multi_probe_rec toggle
        #Grab only rows that have is_all_trials == True, only these have predict_proba
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials'),
        )
        
        .with_columns(
            pl.int_ranges(0, pl.col(predict_proba_alias).list.len()).alias('trial_index')
        )
        .drop(
            'shift_idx', 'is_all_trials', 'electrode_group_names', 'unit_criteria', 'is_sole_recording', 
            'predict_proba', 'predict_proba_all_trials', 'decision_function', 'decision_function_all',
            'balanced_accuracy_test','balanced_accuracy_train','unit_ids','coefs', 'trial_indices'  
        )
        .explode(explode_cols)
        .group_by('session_id', 'structure', 'unit_subsample_size', 'bin_center', 'bin_size', 'time_aligned_to', 'trial_index', )
        .agg(explode_agg_expr)
        .group_by(grouping_cols - {'electrode_group_names', 'unit_criteria'})
        .agg(
            pl.col('spont_trial_times').first(),
            pl.col('spont_epoch_name').first(),
            pl.col('spont_trial_is_rewarded').first(),
            pl.col(final_agg_cols).sort_by('trial_index')
        )
        .sort('session_id','structure', 'unit_subsample_size', 'bin_center', 'bin_size', 'time_aligned_to', )
        .collect(engine='streaming')
    )

    decoder_confidence_spont_df = exclude_structures_from_df(
        decoder_confidence_spont_df, 
        exclude_redundant_structures=exclude_redundant_structures, 
        exclude_general_structures=exclude_general_structures
    )

    return decoder_confidence_spont_df

def load_decoder_coefs(results_path, session_list, combine_multi_probe_rec=True, exclude_redundant_structures=True, exclude_general_structures=True):
    """Load decoder coefficients for each session and structure.
    
    Retrieves the feature importance weights (coefficients) from trained decoders
    for each session-structure combination. This allows analysis of which units
    contribute most to decoding performance.
    
    Parameters
    ----------
    results_path : str
        Path to parquet file(s) containing decoder results. Can be local path or S3 URI.
    session_list : list of str
        List of session IDs to include in analysis.
    combine_multi_probe_rec : bool, default=True
        If True, combine results from multiple probe insertions recording the same
        structure. If False, keep results from each probe separate.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with decoder coefficients:
        - session_id : Session identifier
        - structure : Brain structure name
        - unit_subsample_size : Number of units used for decoding
        - bin_center : Center time of temporal bin relative to event
        - coefs : List of decoder coefficients for each unit
        
        Sorted by session_id, structure, unit_subsample_size, and bin_center.
    
    Notes
    -----
    - Coefficients are averaged across repeated decoder runs if combine_multi_probe_rec is True
    - Joins with consolidated units table to get total_n_units for each structure
    
    Examples
    --------
    >>> decoder_coefs_df = load_decoder_coefs(
    ...     's3://bucket/results/',
    ...     ['session1', 'session2'],
    ...     combine_multi_probe_rec=True
    ... )
    """

    combine_multi_probe_expr = get_multi_probe_expr(combine_multi_probe_rec)

    decoder_coefs_df = (
        pl.scan_parquet(results_path)
        .filter(
            pl.col('session_id').is_in(session_list),
        )
        .with_columns(
            pl.col('electrode_group_names').flatten().n_unique().eq(1).over({'session_id','structure'}).alias('is_sole_recording'),     
        )
        .filter(
            combine_multi_probe_expr,
            pl.col('is_all_trials').eq(True),
        )
        .sort('session_id', 'structure', 'shift_idx', 'repeat_idx', 'time_aligned_to', 'bin_center', descending=False, maintain_order=True)
        .collect()
        .with_columns([
            pl.col('unit_ids').list.len().alias('n_units'),
            pl.col("electrode_group_names").list.n_unique().alias("n_probes"),
            pl.col("electrode_group_names")
            .list.eval(pl.element().str.replace("probe", ""))
            .list.join("")
            .alias("probe")
        ])
        .drop(
            'shift_idx', 'is_all_trials', 'electrode_group_names', 'unit_criteria', 'is_sole_recording', 
            'predict_proba', 'predict_proba_all_trials', 'decision_function', 'decision_function_all',
            'balanced_accuracy_test', 'balanced_accuracy_train', 'trial_indices', 'labels', 
            'train_test_split_label', 
        )
    )

    decoder_coefs_df = exclude_structures_from_df(
        decoder_coefs_df, 
        exclude_redundant_structures=exclude_redundant_structures, 
        exclude_general_structures=exclude_general_structures
    )

    return decoder_coefs_df

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


def get_session_structure_results(predict_proba_pd, sel_session, sel_structure, sel_unit_subsample_size, sel_time_aligned_to, sel_bin_center=None, round_bin_center_decimals=3):
    """
    Get the results for a specific session and structure.
    """
    predict_proba_pd['bin_center']=predict_proba_pd['bin_center'].round(round_bin_center_decimals)
    if sel_unit_subsample_size=='all':
        example_area_results=predict_proba_pd.query(f'session_id=="{sel_session}" and structure=="{sel_structure}" and \
                                                    time_aligned_to=="{sel_time_aligned_to}" and unit_subsample_size.isna()'
                                                    ).sort_values('bin_center').reset_index(drop=True)
    else:
        example_area_results=predict_proba_pd.query(f'session_id=="{sel_session}" and structure=="{sel_structure}" and \
                                                    time_aligned_to=="{sel_time_aligned_to}" and unit_subsample_size=={sel_unit_subsample_size}'
                                                    ).sort_values('bin_center').reset_index(drop=True)
    if sel_bin_center is not None:
        example_area_results=example_area_results.query(f'bin_center=={sel_bin_center}').reset_index(drop=True)
    #get context switches
    is_context_switch=np.concatenate([[0],np.diff(example_area_results['is_vis_rewarded'].iloc[0])]).astype(bool)
    context_switch_list=[]
    for rr in range(len(example_area_results)):
        context_switch_list.append(is_context_switch)
    example_area_results['is_context_switch']=context_switch_list


    return example_area_results


def get_context_switch_table(predict_proba_pd, session_list, sel_unit_subsample_size=None, sel_time_aligned_to='stim_start_time'):

    get_trials_rel_to_switch=[-3,-2,-1,0,1,2,3,4]

    all_performance=pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/performance.parquet').collect().to_pandas()
    all_trials=pl.scan_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/trials.parquet').collect().to_pandas()

    if sel_unit_subsample_size is None:
        #get the first unique unit_subsample_size
        sel_unit_subsample_size=predict_proba_pd['unit_subsample_size'].dropna().unique()[0]

    context_switch_table={
        'session_id':[],
        'structure':[],
        'predict_proba':[],
        'bin_centers':[],
        'unit_subsample_size':[],
        'time_aligned_to':[],
        'trial_index':[],
        'trial_rel_to_switch':[],
        'switch_index_in_session':[],
        'is_response':[],
        'is_vis_rewarded':[],
        'is_contingent_switch':[],
        'stim_name':[],
        'dprime_before_switch':[],
        'dprime_after_switch':[],


    }

    for sel_session in predict_proba_pd['session_id'].unique():
        if sel_session not in session_list:
            print(f"session {sel_session} not in session_list; skipping")
            continue
        for sel_structure in predict_proba_pd.query('session_id==@sel_session')['structure'].unique():

            #get session-structure results
            example_area_results=get_session_structure_results(predict_proba_pd, sel_session, sel_structure, sel_unit_subsample_size, sel_time_aligned_to)

            predict_proba_stack=np.vstack(example_area_results['predict_proba'].values).T

            session_performance=all_performance.query(f'session_id=="{sel_session}"')
            session_trials=all_trials.query(f'session_id=="{sel_session}"')

            #choose values based on their actual, not implied, trial index
            context_switch_trial_index=session_trials.query('is_block_switch')['trial_index'].values
            trial_index=example_area_results['trial_index'].iloc[0]

            #loop through context switches
            for ii,tt in enumerate(context_switch_trial_index):
                # print(f"Context switch {ii} of {len(context_switch_trial_index)}")
                # print(f"Trial {tt} of {len(example_area_results['is_context_switch'].iloc[0])}")
                dprime_before_switch=session_performance['cross_modality_dprime'].iloc[ii]
                dprime_after_switch=session_performance['cross_modality_dprime'].iloc[ii+1]

                is_contingent_switch=session_trials['is_response'].iloc[tt]

                #get the is_vis_rewarded
                is_vis_rewarded=session_trials['is_vis_rewarded'].iloc[tt]

                for t_diff in get_trials_rel_to_switch:

                    adj_tt=np.where(trial_index==tt+t_diff)[0]
                    if len(adj_tt) == 0:
                        # print(f"session {sel_session} structure {sel_structure} ERROR:")
                        # print("trial index not found in predict_proba stack;")
                        # print("skipping trial")
                        continue
                    else:
                        adj_tt=adj_tt[0]

                    if tt+t_diff!=trial_index[adj_tt]:
                        print('ERROR: trial index not matching!!')
                        break
                    
                    if adj_tt >= predict_proba_stack.shape[0]:
                        print(f"session {sel_session} structure {sel_structure} ERROR:")
                        print("trial index out of bounds of predict_proba stack;")
                        print("skipping trial")
                        continue
                        
                    #get trial from predict_proba_stack
                    predict_proba_values=predict_proba_stack[adj_tt,:]

                    #get the bin center of the trial
                    bin_centers=example_area_results['bin_center'].values

                    #get the is_response
                    is_response=example_area_results['is_response'].iloc[0][adj_tt]

                    #get the stim_name
                    stim_name=example_area_results['stim_name'].iloc[0][adj_tt]

                    #append to the context switch table
                    context_switch_table['session_id'].append(sel_session)
                    context_switch_table['structure'].append(sel_structure)
                    context_switch_table['predict_proba'].append(predict_proba_values)
                    context_switch_table['bin_centers'].append(bin_centers)
                    context_switch_table['unit_subsample_size'].append(sel_unit_subsample_size)
                    context_switch_table['time_aligned_to'].append(sel_time_aligned_to)
                    context_switch_table['trial_index'].append(trial_index[adj_tt])
                    context_switch_table['trial_rel_to_switch'].append(t_diff)
                    context_switch_table['switch_index_in_session'].append(ii)
                    context_switch_table['is_response'].append(is_response)
                    context_switch_table['is_vis_rewarded'].append(is_vis_rewarded)
                    context_switch_table['is_contingent_switch'].append(is_contingent_switch)
                    context_switch_table['stim_name'].append(stim_name)
                    context_switch_table['dprime_before_switch'].append(dprime_before_switch)
                    context_switch_table['dprime_after_switch'].append(dprime_after_switch)

    context_switch_table=pd.DataFrame(context_switch_table)
    context_switch_table['bin_centers']=context_switch_table['bin_centers'].round(4)

    delta_predict_proba_2_bins=[]
    delta_predict_proba_3_bins=[]

    for ii, rr in context_switch_table.iterrows():
        
        # get the predict proba for the trial
        predict_proba_trial = np.array(rr['predict_proba'])

        delta_2_bins=predict_proba_trial[-2:].mean() - predict_proba_trial[:2].mean()
        delta_predict_proba_2_bins.append(delta_2_bins)

        delta_3_bins=predict_proba_trial[-3:].mean() - predict_proba_trial[:3].mean()
        delta_predict_proba_3_bins.append(delta_3_bins)

    context_switch_table['delta_predict_proba_2_bins'] = delta_predict_proba_2_bins
    context_switch_table['delta_predict_proba_3_bins'] = delta_predict_proba_3_bins

    return context_switch_table


#subtract blockwise means
def subtract_blockwise_mean(input, block_indices):
    adjusted_input = np.zeros_like(input)
    unique_blocks = np.unique(block_indices)
    for block in unique_blocks:
        block_mask = (block_indices == block)
        block_mean = np.mean(input[block_mask])
        adjusted_input[block_mask] = input[block_mask] - block_mean
    return adjusted_input

#flip aud blocks
def flip_auditory_blocks(input, block_indices, is_vis_rewarded, input_type=None):
    adjusted_input = input.copy()
    unique_blocks = np.unique(block_indices)
    for block in unique_blocks:
        block_mask = (block_indices == block)
        if input_type=='predict_proba':
            if not is_vis_rewarded[block_mask].all():
                adjusted_input[block_mask] = 1.0 - input[block_mask]
        elif input_type=='decision_function':
            if not is_vis_rewarded[block_mask].all():
                adjusted_input[block_mask] = -input[block_mask]
        else:
            print('ERROR: must provide input type: predict_proba or decision_function')
            raise NotImplementedError
    return adjusted_input

#exclude is instruction
def exclude_instruction_trials(input, is_instruction_trial):
    adjusted_input = input.copy()
    adjusted_input[is_instruction_trial] = np.nan
    return adjusted_input



def group_structures(frame: polars._typing.FrameType, keep_originals=True) -> polars._typing.FrameType:
    grouping = {
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
    n_repeats = 2 if keep_originals else 1
    frame = (
        frame
        .with_columns(
            pl.when(pl.col('structure').is_in(grouping))
            .then(pl.col('structure').repeat_by(n_repeats))
            .otherwise(pl.col('structure').repeat_by(1))
        )
        .explode('structure')
        .with_columns(
            pl.when(pl.col('structure').is_in(grouping).is_first_distinct().over('unit_id'))
            .then(pl.col('structure').replace(grouping))
            .otherwise(pl.col('structure'))
        )
    
    )
    return frame 

def repeat_multi_probe_areas(frame: polars._typing.FrameType) -> polars._typing.FrameType:
    """"If an area is recorded on multiple probes, transform the dataframe so it has rows for each
    probe and a row for both probes combined ('electrode_group_names': List[String])"""
    duplicates =  (
        frame
        .clone()
        .with_columns(
             pl.col('electrode_group_name').unique().over('session_id', 'structure', mapping_strategy='join').alias('electrode_group_names')
         )
        .filter(
             pl.col('electrode_group_names').list.len().ge(2)
         )
    )
    return (
        pl.concat(
            [
                frame.with_columns(pl.col('electrode_group_name').cast(pl.List(pl.String)).alias('electrode_group_names')),
                duplicates,
            ],
        )
        .drop('electrode_group_name')
    )

def decode_context_with_linear_shift(
    session_ids: str | Iterable[str],
    params: Params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    if params.filter_units_by_metrics==False:
        combinations_df = (
            utils.get_df('units', lazy=True)
            .drop_nulls('structure')
            .filter(
                pl.col('session_id').is_in(session_ids),
                params.units_query,
            )
            .pipe(group_structures, keep_originals=True)
            .pipe(repeat_multi_probe_areas)
            .filter(params.min_n_units_query)
            .select(params.units_group_by)
            .unique(params.units_group_by)
            .collect()
        )

    #option to apply filter by unit metrics
    elif params.filter_units_by_metrics==True:
        metrics_table_path=None
        for p in sorted(utils.get_data_root().iterdir(), reverse=True): # in case we have multiple assets attached, the latest will be used
            if p.is_dir() and p.name.startswith('dynamicrouting_single_unit'):
                for f in p.iterdir():
                    if f.is_file() and f.name.endswith('.parquet'):
                        metrics_table_path=f
                        break
            if metrics_table_path is not None:
                break

        if metrics_table_path is not None:
            # get unit ids defined by filter
            # join with units lazyframe, then apply metrics filter, then get combinations
            metrics_table=(
                pl.scan_parquet(metrics_table_path)
                .join(
                    utils.get_df('units', lazy=True),
                    on='unit_id'
                    )
            )

            combinations_df=(
                metrics_table
                .drop_nulls('structure')
                .filter(
                    pl.col('session_id').is_in(session_ids),
                    params.units_query,
                    params.unit_metrics_filter,
                )
                .pipe(group_structures, keep_originals=True)
                .pipe(repeat_multi_probe_areas)
                .filter(params.min_n_units_query)
                .select(params.units_group_by)
                .unique(params.units_group_by)
                .collect()
            )


    if params.skip_existing and params.data_path.exists():
        existing = (
            pl.scan_parquet(params.data_path.as_posix().removesuffix('/') + '/')
            .filter(
                pl.col('unit_subsample_size').is_null() if params.unit_subsample_size is None else pl.col('unit_subsample_size').eq(params.unit_subsample_size),
                pl.col('unit_criteria') == params.unit_criteria,
            )
            .select(params.units_group_by)
            .unique(params.units_group_by)
            .collect()
            .to_dicts()
        )
    else:
        existing = []
    def is_row_in_existing(row):
        """Regular dict comparison doesn't work with list field?"""
        return any(
            x for x in existing 
            if all(x[k] == row[k] for k in ['session_id', 'structure'])
            and set(x['electrode_group_names']) == set(row['electrode_group_names'])
        )
        
    logger.info(f"Processing {len(combinations_df)} unique session/area/probe combinations")
    if params.use_process_pool:
        session_results: dict[str, list[cf.Future]] = {}
        future_to_session = {}
        lock = None # multiprocessing.Manager().Lock() # or None
        with cf.ProcessPoolExecutor(max_workers=params.max_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            for row in combinations_df.iter_rows(named=True):
                if params.skip_existing and is_row_in_existing(row):
                    logger.info(f"Skipping {row} - results already exist")
                    continue
                future = executor.submit(
                    wrap_decoder_helper,
                    params=params,
                    **row,
                    lock=lock,
                )
                session_results.setdefault(row['session_id'], []).append(future)
                future_to_session[future] = row['session_id']
                logger.debug(f"Submitted decoding to process pool for session {row['session_id']}, structure {row['structure']}")
                if params.test:
                    logger.info("Test mode: exiting after first session")
                    break
            for future in tqdm.tqdm(cf.as_completed(future_to_session), total=len(future_to_session), unit='structure', desc=f'Decoding'):
                session_id = future_to_session[future]
                if all(future.done() for future in session_results[session_id]):
                    logger.debug(f"Decoding completed for session {session_id}")
                    for f in session_results[session_id]:
                        try:
                            _ = f.result()
                        except Exception:
                            logger.exception(f'{session_id} | Failed:')
                    logger.info(f'{session_id} | Completed')

    else: # single-process mode
        for row in tqdm.tqdm(combinations_df.iter_rows(named=True), total=len(combinations_df), unit='row', desc=f'decoding'):
            if params.skip_existing and is_row_in_existing(row):
                logger.info(f"Skipping {row} - results already exist")
                continue
            try:
                wrap_decoder_helper(
                    params=params,
                    **row,
                )
            except NotEnoughBlocksError as exc:
                logger.warning(f'{row["session_id"]} | {exc!r}')
            except Exception:
                logger.exception(f'{row["session_id"]} | Failed:')
            if params.test:
                logger.info("Test mode: exiting after first session")
                break

def wrap_decoder_helper(
    params: Params,
    session_id: str,
    structure: str,
    electrode_group_names: Sequence[str],
    lock = None,
) -> None:
    logger.debug(f"Getting units and trials for {session_id} {structure}")
    results = []

    all_trials = (
        utils.get_df('trials', lazy=True)
        .filter(
            pl.col('session_id') == session_id,
        ).with_columns( #make new columns for is_response_or_reward and response_or_reward_time
            pl.min_horizontal('response_time','reward_time').alias('response_or_reward_time'),
            (pl.col('response_time').is_not_null() | pl.col('reward_time').is_not_null()).alias('is_response_or_reward')
        ).filter(
            params.trials_filter,
            # obs_intervals may affect number of trials available
        )
        .sort('trial_index')
        .collect()
    )

    if params.test_on_spontaneous:
        #make random seed from the session id for consistency
        random_seed = int(session_id.replace('_','').replace('-',''))
        spont_flag=True
        try:
            spont_trials = pl.from_pandas(data_utils.generate_spontaneous_trials_table(session_id,distribution='DR',random_seed=random_seed))
            interval_bin_size = params.spike_count_interval_configs[0].bin_size
            spike_counts_spont_df = (
                utils.get_per_trial_spike_times(
                    intervals={
                        'n_spikes_window': (
                            pl.col('start_time') + 0, 
                            pl.col('start_time') + interval_bin_size,
                        ),
                    },
                    trials_frame=spont_trials,
                    as_counts=True,
                    unit_ids=(
                        utils.get_df('units', lazy=True)
                        .pipe(group_structures)
                        .filter(
                            params.units_query,
                            pl.col('session_id') == session_id,
                            pl.col('structure') == structure,
                            pl.col('electrode_group_name').is_in(electrode_group_names),
                        )
                        .select('unit_id')
                        .collect()
                        ['unit_id']
                        .unique()
                    ),
                )
                .filter(
                    pl.col('n_spikes_window').is_not_null(),
                    # only keep observed trials
                )
                .sort('trial_index', 'unit_id') 
            )
        except:
            logger.info(f"No spontaneous epoch for {session_id}; skipping session")
            spont_flag=False
            spont_data=None
            return
        
    else:
        spont_flag=False
        spont_data = None

    if params.test_across_context:
        train_test_split_labels=[
            'train_vis_test_aud',
            'train_vis_test_vis',
            'train_aud_test_vis',
            'train_aud_test_aud'
        ]
    else:
        train_test_split_labels= [None]

    if params.label_to_decode == 'context_appropriate_for_response':
        all_trials=(
            all_trials.filter(
                pl.col('is_target'),
                pl.col('is_hit').eq(False)
            ).with_columns(
                (((pl.col("is_vis_target")==True) & (pl.col("is_response")==True)) |
                ((pl.col("is_aud_target")==True) & (pl.col("is_response")==False))
                ).alias("is_vis_appropriate_response"),
            ).with_columns(
                (pl.when(pl.col("is_vis_appropriate_response")==True)
                    .then(pl.lit("vis"))
                    .otherwise(pl.lit("aud"))
                ).alias("context_appropriate_for_response")
            )
        )
        #if only one block, this method won't work so cancel for this structure/session
        if all_trials.n_unique('block_index') != 6:
            raise NotEnoughBlocksError(f'Expecting 6 blocks for context_appropriate_for_response analysis: {session_id} has {all_trials.n_unique("block_index")} blocks of observed ephys data')


    # select unit ids for resampling here - keep consistent across time bins
    resample_unit_ids=[]
    
    if params.filter_units_by_metrics==False:
        unique_unit_ids=(
            utils.get_df('units', lazy=True)
            .pipe(group_structures)
            .filter(
                params.units_query,
                pl.col('session_id') == session_id,
                pl.col('structure') == structure,
                pl.col('electrode_group_name').is_in(electrode_group_names),
            )
            .select('unit_id')
            .sort('unit_id')
            .collect()
            ['unit_id']
            .unique()
        )

    # option to filter by separate unit metrics table (unit IDs must match!)
    # unit metrics data asset folder must contain a single .parquet file!
    elif params.filter_units_by_metrics==True:
        metrics_table_path=None
        for p in sorted(utils.get_data_root().iterdir(), reverse=True): # in case we have multiple assets attached, the latest will be used
            if p.is_dir() and p.name.startswith('dynamicrouting_single_unit'):
                for f in p.iterdir():
                    if f.is_file() and f.name.endswith('.parquet'):
                        metrics_table_path=f
                        break
            if metrics_table_path is not None:
                break

        if metrics_table_path is not None:
            # get unit ids defined by filter
            metrics_table=(
                pl.scan_parquet(metrics_table_path)
                .join(
                    utils.get_df('units', lazy=True),
                    on='unit_id'
                    )
            )

            unique_unit_ids=(
                metrics_table
                .pipe(group_structures)
                .filter(
                    params.units_query,
                    params.unit_metrics_filter,
                    pl.col('session_id') == session_id,
                    pl.col('structure') == structure,
                    pl.col('electrode_group_name').is_in(electrode_group_names),
                )
                .select('unit_id')
                .sort('unit_id')
                .collect()
                ['unit_id']
                .unique()
            )


    n_units_to_use = params.unit_subsample_size or len(unique_unit_ids) # if unit_subsample_size is None, use all available        
    unit_idx = list(range(0, len(unique_unit_ids)))

    for repeat_idx in range(params.n_repeats):
        sel_unit_idx = random.sample(unit_idx, n_units_to_use)
        resample_unit_ids.append(unique_unit_ids[sel_unit_idx])
    resample_unit_ids=np.array(resample_unit_ids)
        

    for interval_config in params.spike_count_interval_configs:
        for start, stop in interval_config.intervals:
            
            #option to use cumulative spike counts
            #change start to equal the start of the first interval
            if params.use_cumulative_spike_counts and params.sliding_window_size is not None:
                 logger.exception(f'cumulative_spike_counts and sliding_window_size are incompatible, select only one to use')

            if params.use_cumulative_spike_counts:
                start_original=np.copy(start)
                start=interval_config.intervals[0][0]
            
            elif params.sliding_window_size is not None:
                start_original=np.copy(start)
                stop_original=np.copy(stop)
                start=(start_original+stop_original)/2-(params.sliding_window_size/2)
                stop=(start_original+stop_original)/2+(params.sliding_window_size/2)
                #start=stop-params.sliding_window_size
                

            #option to subtract trialwise baseline, defined as 500ms before event (stimulus)
            if params.baseline_subtraction:
                spike_counts_df = (
                    utils.get_per_trial_spike_times(
                        intervals={
                            'n_spikes_baseline': (
                                pl.col(interval_config.event_column_name) + -0.5, 
                                pl.col(interval_config.event_column_name) + 0
                            ),
                            'n_spikes_window': (
                                pl.col(interval_config.event_column_name) + start, 
                                pl.col(interval_config.event_column_name) + stop,
                            ),
                        },
                        trials_frame=all_trials,
                        as_counts=True,
                        unit_ids=(
                            utils.get_df('units', lazy=True)
                            .pipe(group_structures)
                            .filter(
                                params.units_query,
                                pl.col('session_id') == session_id,
                                pl.col('structure') == structure,
                                pl.col('electrode_group_name').is_in(electrode_group_names),
                            )
                            .select('unit_id')
                            .collect()
                            ['unit_id']
                            .unique()
                        ),
                    )
                    .with_columns(
                        pl.col('n_spikes_baseline')*2*(stop-start),
                        #get firing rate, then extrapolate spike counts to the bin size used
                    )
                    .with_columns(
                        pl.col('n_spikes_window').sub(pl.col('n_spikes_baseline'))
                    )
                    .filter(
                        pl.col('n_spikes_window').is_not_null(),
                        # only keep observed trials
                    )
                    .sort('trial_index', 'unit_id') 
                )

            else:

                spike_counts_df = (
                    utils.get_per_trial_spike_times(
                        intervals={
                            'n_spikes_window': (
                                pl.col(interval_config.event_column_name) + start, 
                                pl.col(interval_config.event_column_name) + stop,
                            ),
                        },
                        trials_frame=all_trials,
                        as_counts=True,
                        unit_ids=(
                            utils.get_df('units', lazy=True)
                            .pipe(group_structures)
                            .filter(
                                params.units_query,
                                pl.col('session_id') == session_id,
                                pl.col('structure') == structure,
                                pl.col('electrode_group_name').is_in(electrode_group_names),
                            )
                            .select('unit_id')
                            .collect()
                            ['unit_id']
                            .unique()
                        ),
                    )
                    .filter(
                        pl.col('n_spikes_window').is_not_null(),
                        # only keep observed trials
                    )
                    .sort('trial_index', 'unit_id') 
                )
                # len == n_units x n_trials, with spike counts in a column
                # sequence of unit_ids is used later: don't re-sort!

            
            logger.debug(f"Got spike counts: {spike_counts_df.shape} rows")

            if params.label_to_decode=='rewarded_modality':
                trials = (
                    all_trials
                    .filter(
                        pl.col('session_id') == session_id,
                        pl.col('trial_index').is_in(spike_counts_df['trial_index'].unique()),
                        # obs_intervals may affect number of trials available
                    )
                    .sort('trial_index')
                    .select(params.label_to_decode, 'start_time', 'trial_index', 'block_index', 'session_id')
                )
            else:
                trials = (
                    all_trials
                    .filter(
                        pl.col('session_id') == session_id,
                        pl.col('trial_index').is_in(spike_counts_df['trial_index'].unique()),
                        # obs_intervals may affect number of trials available
                    )
                    .sort('trial_index')
                    .select(params.label_to_decode, 'start_time', 'trial_index', 'block_index', 'session_id', 'rewarded_modality', 'is_vis_stim', 'is_aud_stim', 'is_correct')
                )

            if (
                trials['block_index'].n_unique() == 1
                and not (
                    utils.get_df('session')
                    .filter(
                        pl.col('session_id') == trials['session_id'][0],
                        pl.col('keywords').list.contains('templeton'),
                    )
                ).is_empty()
            ):
                logger.info(f'Adding dummy context labels for Templeton session {session_id}')
                trials = (
                    trials
                    .with_columns(
                        pl.col('start_time').sub(pl.col('start_time').min().over('session_id')).truediv(10*60).floor().clip(0, 5).alias('block_index')
                        # short 7th block will sometimes be present: merge into 6th with clip
                    )
                    .with_columns(
                        pl.when(pl.col('block_index').mod(2).eq(random.choice([0, 1])))
                        .then(pl.lit('vis'))
                        .otherwise(pl.lit('aud'))
                        .alias('rewarded_modality')
                    )
                    .sort('trial_index')
                )
            if trials.n_unique('block_index') != params.n_blocks_expected:
                raise NotEnoughBlocksError(f'Expecting {params.n_blocks_expected} blocks: {session_id} {structure} has {trials.n_unique("block_index")} blocks of observed ephys data')
            logger.debug(f"Got {len(trials)} trials")

            label_to_decode = trials[params.label_to_decode].to_numpy().squeeze()

            if params.linear_shift:
                max_neg_shift = math.ceil(len(trials.filter(pl.col('block_index')==0))/2)
                max_pos_shift = math.floor(len(trials.filter(pl.col('block_index')==5))/2)
            else:
                max_neg_shift = 0
                max_pos_shift = 1
            shifts = tuple(range(-max_neg_shift, max_pos_shift + 1))
            logger.debug(f"Using shifts from {shifts[0]} to {shifts[-1]}")

            for repeat_idx in tqdm.tqdm(range(params.n_repeats), total=params.n_repeats, unit='repeat', desc=f'repeating {structure} | {session_id}'):
                
                for train_test_split_label in train_test_split_labels: #loop through custom train-test labels

                    if train_test_split_label is None:
                        train_test_split_input=None

                    filtered_unit_df = spike_counts_df.filter(pl.col('unit_id').is_in(resample_unit_ids[repeat_idx]))

                    spike_counts_array = (
                        filtered_unit_df
                        .select('n_spikes_window')
                        .to_numpy()
                        .squeeze()
                        .reshape(filtered_unit_df.n_unique('trial_index'), filtered_unit_df.n_unique('unit_id'))
                    )
                    logger.debug(f"Reshaped spike counts array: {spike_counts_array.shape}")
                    
                    unit_ids = filtered_unit_df['unit_id'].unique(maintain_order=True).to_list()

                    logger.debug(f"Repeat {repeat_idx}: selected {len(sel_unit_idx)} units")

                    if spont_flag:
                        filtered_spont_unit_df=spike_counts_spont_df.filter(pl.col('unit_id').is_in(resample_unit_ids[repeat_idx]))
                        spont_data = (
                            filtered_spont_unit_df
                            .select('n_spikes_window')
                            .to_numpy()
                            .squeeze()
                            .reshape(filtered_spont_unit_df.n_unique('trial_index'), filtered_spont_unit_df.n_unique('unit_id'))
                        )
                    
                    for shift in (*shifts, None): # None will be a special case using all trials, with no shift
                        
                        is_all_trials = shift is None
                        if not is_all_trials:

                            if params.linear_shift==0:
                                continue
                            labels = label_to_decode[max_neg_shift: -max_pos_shift]
                            if params.crossval=='blockwise' or params.crossval=='leave_2_blocks_out':
                                crossval_index=trials['block_index'].to_numpy().squeeze()[max_neg_shift: -max_pos_shift]
                            else:
                                crossval_index=None
                            first_trial_index = max_neg_shift + shift
                            last_trial_index = len(trials) - max_pos_shift + shift
                            logger.debug(f"Shift {shift}: using trials {first_trial_index} to {last_trial_index} out of {len(trials)}")
                            assert first_trial_index >= 0, f"{first_trial_index=}"
                            assert last_trial_index > first_trial_index, f"{last_trial_index=}, {first_trial_index=}"
                            assert last_trial_index <= spike_counts_array.shape[0], f"{last_trial_index=}, {spike_counts_array.shape[0]=}"
                            data = spike_counts_array[first_trial_index: last_trial_index, :]
                        else:
                            labels = label_to_decode
                            crossval_index=trials['block_index'].to_numpy().squeeze()
                            data = spike_counts_array[:, :]

                            if params.test_across_context:
                                rng = np.random.default_rng()

                                #temporary adjustment depending on variable being decoded:
                                if params.label_to_decode=='is_response':
                                    vis_query_string='rewarded_modality=="vis" and is_vis_stim==True and is_correct==True'
                                    aud_query_string='rewarded_modality=="aud" and is_aud_stim==True and is_correct==True'
                                elif params.label_to_decode=='stim_name':
                                    vis_query_string='rewarded_modality=="vis"'
                                    aud_query_string='rewarded_modality=="aud"'

                                if train_test_split_label is not None:
                                    # label_to_decode = trials[params.label_to_decode].to_numpy().squeeze()
                                    params.crossval='custom'

                                    #divide vis trials into 2 train sets & aud trials into 2 test sets
                                    vis_context_trial_index=trials.to_pandas().reset_index().query(vis_query_string).index.values
                                    #permute vis_context_trial_index
                                    vis_context_trial_index_permuted = rng.permutation(vis_context_trial_index)
                                    half_len_vis_trials=np.round(len(vis_context_trial_index_permuted)/2).astype(int)
                                    #get folds
                                    vis_fold_1=vis_context_trial_index_permuted[:half_len_vis_trials]
                                    vis_fold_2=vis_context_trial_index_permuted[half_len_vis_trials:]

                                    aud_context_trial_index=trials.to_pandas().reset_index().query(aud_query_string).index.values
                                    #permute vis_context_trial_index
                                    aud_context_trial_index_permuted = rng.permutation(aud_context_trial_index)
                                    half_len_aud_trials=np.round(len(aud_context_trial_index_permuted)/2).astype(int)
                                    #get folds
                                    aud_fold_1=aud_context_trial_index_permuted[:half_len_aud_trials]
                                    aud_fold_2=aud_context_trial_index_permuted[half_len_aud_trials:]

                                    if train_test_split_label=='train_vis_test_aud':
                                        train=[vis_fold_1,vis_fold_1,vis_fold_2,vis_fold_2]
                                        test=[aud_fold_1,aud_fold_2,aud_fold_1,aud_fold_2]
                                    elif train_test_split_label=='train_vis_test_vis':
                                        train=[vis_fold_1,vis_fold_2]
                                        test=[vis_fold_2,vis_fold_1]
                                    elif train_test_split_label=='train_aud_test_vis':
                                        train=[aud_fold_1,aud_fold_1,aud_fold_2,aud_fold_2]
                                        test=[vis_fold_1,vis_fold_2,vis_fold_1,vis_fold_2]
                                    elif train_test_split_label=='train_aud_test_aud':
                                        train=[aud_fold_1,aud_fold_2]
                                        test=[aud_fold_2,aud_fold_1]
                                    else:
                                        train=[]
                                        test=[]
                                        train_test_split_label=None
                                        train_test_split_input=None

                                    train_test_split_input=zip(train,test)


                        assert data.shape == (len(labels), len(unit_ids)), f"{data.shape=}, {len(labels)=}, {len(sel_unit_idx)=}"
                        logger.debug(f"Shift {shift}: using data shape {data.shape} with {len(labels)} labels")

                        _result = decoder_helper(
                            data,
                            labels,
                            decoder_type=params.decoder_type,
                            crossval=params.crossval,
                            crossval_index=crossval_index,
                            labels_as_index=params.labels_as_index,
                            train_test_split_input=train_test_split_input,
                            train_test_split_label=train_test_split_label,
                            regularization=params.regularization,
                            penalty=params.penalty,
                            solver=params.solver,
                            n_jobs=None,
                            other_data=spont_data,
                            scaler=params.scaler,
                        )
                        result = {}
                        result['balanced_accuracy_test'] = _result['balanced_accuracy_test'].item()
                        result['balanced_accuracy_train'] = _result['balanced_accuracy_train'].item()
                        result['time_aligned_to'] = interval_config.event_column_name
                        result['bin_size'] = interval_config.bin_size
                        result['sliding_window_size'] = params.sliding_window_size
                        if params.use_cumulative_spike_counts:
                            result['bin_center'] = stop
                        elif params.sliding_window_size is not None:
                            result['bin_center'] = (start_original + stop_original) / 2
                            #result['bin_center'] = stop
                        else:
                            result['bin_center'] = (start + stop) / 2
                        result['shift_idx'] = shift
                        result['repeat_idx'] = repeat_idx
                        result['labels'] = _result['labels'].tolist()
                        result['train_test_split_label'] = train_test_split_label

                        train_set_indices=[]
                        for idx, trial_list in enumerate(_result['train_trials']):
                            train_set_indices.append(np.ones(len(trial_list))*idx)
                        result['train_set_indices'] = np.hstack(train_set_indices).astype('int').tolist()
                        result['train_trials'] = np.hstack(_result['train_trials']).tolist()
                        
                        test_set_indices=[]
                        for idx, trial_list in enumerate(_result['test_trials']):
                            test_set_indices.append(np.ones(len(trial_list))*idx)
                        result['test_set_indices'] = np.hstack(test_set_indices).astype('int').tolist()
                        result['test_trials'] = np.hstack(_result['test_trials']).tolist()

                        result['balanced_accuracy_test_all'] = _result['balanced_accuracy_test_all'].tolist()
                        
                        if shift in (0, None):
                            if params.label_to_decode in ["is_response", "is_target", "is_rewarded"]:
                                result['decision_function'] = _result['decision_function'].tolist()
                                result['decision_function_all'] = _result['decision_function_all'].tolist()
                                result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == True)[0][0]].tolist()
                                result['predict_proba_all_trials'] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == True)[0][0]].tolist()
                            elif params.label_to_decode=="stim_name":
                                #if decoding only 2 stimuli
                                if len(_result['label_names'])==2: 
                                    result['decision_function'] = _result['decision_function'].tolist()
                                    result['decision_function_all'] = _result['decision_function_all'].tolist()
                                    if 'vis1' in _result['label_names']:
                                        temp_target_label='vis1'
                                    elif 'sound1' in _result['label_names']:
                                        temp_target_label='sound1'
                                    result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == temp_target_label)[0][0]].tolist()
                                    result['predict_proba_all_trials'] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == temp_target_label)[0][0]].tolist()
                                #if decoding all 4 stimuli
                                elif len(_result['label_names'])==4:
                                    predict_proba_multiclass=np.full((len(labels),4),np.nan)
                                    predict_proba_all_trials_multiclass=np.full((len(labels),4),np.nan)
                                    stim_order=['sound1','sound2','vis1','vis2']
                                    for ss,stim_label in enumerate(stim_order):
                                        result['decision_function_'+stim_label] = _result['decision_function'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()
                                        result['decision_function_all_'+stim_label] = _result['decision_function_all'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()
                                        result['predict_proba_'+stim_label] = _result['predict_proba'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()
                                        result['predict_proba_all_trials_'+stim_label] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()

                            elif params.label_to_decode in ["rewarded_modality","context_appropriate_for_response"]:
                                result['decision_function'] = _result['decision_function'].tolist()
                                result['decision_function_all'] = _result['decision_function_all'].tolist()
                                result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
                                result['predict_proba_all_trials'] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
                            else:
                                logger.exception(f'{session_id} | Failed: decoding unknown column')

                            if params.test_on_spontaneous: #only save spontaneous results columns if they were computed
                                if spont_data is not None: #save predictions about spont data plus relevant trial info
                                    result['predict_proba_spont'] = _result['predict_proba_other'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
                                    result['decision_function_spont'] = _result['decision_function_other'].tolist()
                                    result['pred_label_spont'] = _result['label_names'][_result['pred_label_other']].tolist()
                                    result['spont_trial_times'] = spont_trials['start_time'].to_list()
                                    result['spont_epoch_name'] = spont_trials['epoch_name'].to_list()
                                    result['spont_trial_is_rewarded'] = spont_trials['is_rewarded'].to_list()
                                else: 
                                    result['predict_proba_spont'] = []
                                    result['decision_function_spont'] = []
                                    result['pred_label_spont'] = []
                                    result['spont_trial_times'] = []
                                    result['spont_epoch_name'] = []
                                    result['spont_trial_is_rewarded'] = []
                        else:
                            # don't save probabilities from shifts which we won't use 
                            result['predict_proba'] = None 
                            result['predict_proba_all_trials'] = None
                            result['decision_function'] = None
                            result['decision_function_all'] = None 
                            
                        if is_all_trials:
                            result['trial_indices'] = trials['trial_index'].to_list()
                        elif shift in (0, None):
                            result['trial_indices'] = trials['trial_index'].to_list()[first_trial_index: last_trial_index]
                        else:
                            # don't save trial indices for all shifts
                            result['trial_indices'] = None 
                            
                        result['unit_ids'] = unit_ids
                        # result['coefs'] = _result['coefs'][0].tolist()
                        result['coefs'] = np.nanmean(np.vstack(_result['coefs_all']),axis=0).tolist()
                        if params.save_all_coefs:
                            coef_crossval_indices=[]
                            all_coefs=[]
                            for idx, coef_list in enumerate(_result['coefs_all']):
                                coef_crossval_indices.append(np.ones(len(coef_list[0]))*idx)
                                all_coefs.append(coef_list[0])
                            result['coef_crossval_indices'] = np.hstack(coef_crossval_indices).astype('int').tolist()
                            result['coefs_all'] = np.hstack(all_coefs).tolist()
                        result['is_all_trials'] = is_all_trials
                        results.append(result)
                        if params.test:
                            break
                    if params.test:
                        break
                if params.test:
                    break
            if params.test:
                logger.info(f"Test mode: exiting after first bin in relative to {interval_config.event_column_name}")
                break
        if params.test:
            logger.info("Test mode: exiting after first event intervals config")
            break
        
    with lock or contextlib.nullcontext():
        logger.info('Writing data')
        (
            pl.DataFrame(results)
            .with_columns(
                pl.lit(session_id).alias('session_id'),
                pl.lit(structure).alias('structure'),
                pl.lit(sorted(electrode_group_names)).alias('electrode_group_names'),
                pl.lit(params.unit_subsample_size).alias('unit_subsample_size').cast(pl.UInt16),
                pl.lit(params.unit_criteria).alias('unit_criteria'),
            )
            .cast(
                {
                    'shift_idx': pl.Int16,
                    'repeat_idx': pl.UInt16,
                    'time_aligned_to': pl.Enum([c.event_column_name for c in params.spike_count_interval_configs]),
                    'trial_indices': pl.List(pl.UInt16),
                    'predict_proba': pl.List(pl.Float64),
                    'coefs': pl.List(pl.Float64),
                }
            )
            .write_parquet(
                (params.data_path / f"{uuid.uuid4()}.parquet").as_posix(),
                compression_level=18,
                statistics='full',    
            )
            # .write_delta(params.data_path.as_posix(), mode='append')
        )
    logger.info(f"Completed decoding for session {session_id}, structure {structure}")
    # return results