import logging
import typing
from typing import Literal

import lazynwb
import npc_lims
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from tqdm import tqdm

import dynamic_routing_analysis.datacube_utils as datacube_utils

# Define master kernel list
master_kernels_list = {
    'intercept': {'function_call': 'intercept', 'type': 'discrete', 'length': 0, 'offset': 0,
                    'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'constant value'},
    'vis1': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'target stim in rewarded context'},
    'sound1': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'target stim in non-rewarded context'},
    'vis2': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'non-target stim in vis context'},
    'sound2': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'non-target stim in vis context'},
    'vis1_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'target stim in rewarded context'},
    'sound1_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'target stim in non-rewarded context'},
    'vis2_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'non-target stim in vis context'},
    'sound2_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'non-target stim in vis context'},
    'vis1_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'target stim in non-rewarded context'},
    'sound1_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'target stim in rewarded context'},
    'vis2_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'non-target stim in aud context'},
    'sound2_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 0, 'orthogonalize': None,
                    'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'non-target stim in aud context'},
    'nose': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False,
                'text': 'Z-scored Euclidean displacement of nose movements'},
    'ears': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False,
                'text': 'Z-scored Euclidean displacement of ear movements'},
    'jaw': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
            'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False,
            'text': 'Z-scored Euclidean displacement of jaw movements'},
    'whisker_pad': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                    'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False,
                    'text': 'Z-scored Euclidean displacement of whisker pad movements'},
    'licks': {'function_call': 'licks', 'type': 'discrete', 'length': 1, 'offset': -0.5, 'orthogonalize': None,
                'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'lick responses'},
    'running': {'function_call': 'running', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'Z-scored running speed'},
    'pupil': {'function_call': 'pupil', 'type': 'continuous', 'length': 1, 'offset': -0.5, 'orthogonalize': None,
                'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'Z-scored pupil diameter'},
    'hit': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5, 'orthogonalize': None,
            'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'lick to GO trial'},
    'miss': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5, 'orthogonalize': None,
                'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'no lick to GO trial'},
    'correct_reject': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5,
                        'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False,
                        'text': 'no lick to NO-GO trial'},
    'false_alarm': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5,
                    'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'lick to NO-GO trial'},
    'context': {'function_call': 'context', 'type': 'discrete', 'length': 0, 'offset': 0, 'orthogonalize': None,
                'num_weights': None, 'shuffle': False, 'shift': False, 'text': 'block-wise context'},
    'session_time': {'function_call': 'session_time', 'type': 'continuous', 'length': 0, 'offset': 0,
                        'orthogonalize': None, 'num_weights': None, 'shuffle': False, 'shift': False,
                        'text': 'z-scored time in session'}
}

logger = logging.getLogger(__name__) # debug < info < warning < error
pd.set_option('display.max_columns', None)
np.random.seed(0)


class RunParams:
    def __init__(self, session_id):
        self.run_params = {
            "session_id": session_id,
            "project": "DynamicRouting",
            "cache_version": '0.0.265',
            "time_of_interest": 'quiescent',
            "spontaneous_duration": 2 * 60,  # in seconds
            "input_variables": None,
            "input_offsets": True,
            "input_window_lengths": None,  # offset

            "drop_variables": None,
            "linear_shift_variables": None,
            "linear_shift_by": 0.1, # in seconds
            "unit_inclusion_criteria": {'isi_violations': 0.1,
                                        'presence_ratio': 0.99,
                                        'amplitude_cutoff': 0.1,
                                        'firing_rate': 1},
            "run_on_qc_units": True,
            "unit_ids_to_use": None,
            "spike_bin_width": 0.1,
            "smooth_spikes_half_gaussian": False,
            "half_gaussian_std_dev": 0.05,
            "areas_to_include": None,
            "areas_to_exclude": None,
            "orthogonalize_against_context": ['facial_features'],
            "quiescent_start_time": -1.5,
            "quiescent_stop_time": 0,
            "trial_start_time": -2,
            "trial_stop_time": 3,
            "intercept": True,
            "model_label": 'fullmodel'
        }

    def update_metric(self, key, value):
        """Update or add a parameter in the run_params dictionary."""
        if key not in self.run_params:
            logger.warning(f"{key} is not a valid key. Adding new parameter '{key}' with value {value}")
        if key == 'unit_inclusion_criteria':
            for criteria in value.keys():
                if criteria in self.run_params[key]:
                    self.run_params[key][criteria] = value[criteria]
                else:
                    logger.warning(f"{criteria} is not a valid key for unit_inclusion_criteria. Skipping.")
        else:
            self.run_params[key] = value

    def update_multiple_metrics(self, updates: dict):
        """Update multiple parameters at once."""
        for key, value in updates.items():
            self.update_metric(key, value)

    def get_params(self):
        """Retrieve the run_params dictionary."""
        self.define_kernels()
        return self.run_params

    def validate_params(self):
        """Validation logic to ensure parameters are consistent."""

        if self.run_params["time_of_interest"] not in ['trial', 'full_trial', 'spontaneous_trial',
                                                        'quiescent', 'spontaneous_quiescent',
                                                        'full_spontaneous',
                                                        'full']:
            raise ValueError(f"Invalid time_of_interest: {self.run_params['time_of_interest']}")

        if self.run_params["spike_bin_width"] <= 0:
            raise ValueError(f"Invalid spike_bin_width: {self.run_params['spike_bin_width']}")


def define_kernels(run_params):
    '''
        Returns kernel info for input variables
    '''

    # Define categories for input variables
    categories = {
        'stimulus_context': ['vis1_vis', 'sound1_vis', 'vis2_vis', 'sound2_vis', 'vis1_aud', 'sound1_aud', 'vis2_aud',
                    'sound2_aud'],
        'stimulus': ['vis1', 'sound1', 'vis2', 'sound2'],
        'movements': ['ears', 'nose', 'jaw', 'whisker_pad', 'running', 'pupil', 'licks'],
        'movements_no_licks': ['ears', 'nose', 'jaw', 'whisker_pad', 'running', 'pupil'],
        'choice': ['hit', 'miss', 'correct_reject', 'false_alarm'],
        'facial_features': ['ears', 'nose', 'jaw', 'whisker_pad']
    }

    # Initialize selected keys list
    selected_keys = []

    # Determine selected input variables based on run_params
    time_of_interest = run_params.get('time_of_interest', '')
    input_variables = run_params.get('input_variables', [])

    # Choose input variables based on 'time_of_interest'
    if not input_variables:
        if 'trial' in time_of_interest or time_of_interest == 'full':
            selected_keys = categories['stimulus'] + categories['movements'] + categories['choice'] + ['context',
                                                                                                    'session_time']
        elif 'quiescent' in time_of_interest:
            selected_keys = categories['movements_no_licks'] + ['context', 'session_time']
        elif 'spontaneous' in time_of_interest:
            selected_keys = categories['movements_no_licks'] + ['session_time']
    else:
        # Extend selected_keys with input variables
        for input_variable in input_variables:
            selected_keys.extend(categories.get(input_variable, [input_variable]))

    # Add intercept if required
    if run_params.get('intercept', False) and 'intercept' not in selected_keys:
        selected_keys.append('intercept')

    # Log error if no input variables are selected
    if not selected_keys:
        raise ValueError("No input variables selected!") # raise value error .

    # remove drop variables if any
    drop_keys = run_params.get('drop_variables', [])
    if drop_keys and run_params['model_label'] != 'fullmodel':
        for drop_key in drop_keys:
            sub_keys = categories.get(drop_key, [drop_key])
            for sub_key in sub_keys:
                logger.info(f': dropping {sub_key}')
                selected_keys.remove(sub_key)

    shift_keys = run_params.get('linear_shift_variables', [])
    if shift_keys and run_params['model_label'] != 'fullmodel':
        for shift_key in shift_keys:
            sub_keys = categories.get(shift_key, [shift_key])
            for sub_key in sub_keys:
                master_kernels_list[sub_key]['shift'] = True

    # Build kernels dictionary based on selected keys
    kernels = {key: master_kernels_list[key] for key in selected_keys}

    if 'intercept' in selected_keys:
        selected_keys.remove('intercept')
    run_params['input_variables'] = selected_keys

    # Update kernel lengths based on run_params
    if not run_params.get('input_offsets'):
        for key, kernel in kernels.items():
            kernel['length'] = 0
            kernel['offset'] = 0
    else:
        input_window_lengths = run_params.get('input_window_lengths', {})
        if input_window_lengths:
            input_window_lengths_updated = {}
            for super_key in input_window_lengths.keys():
                for sub_key in categories.get(super_key, [super_key]):
                    input_window_lengths_updated[sub_key] = input_window_lengths[super_key]

            for key, length in input_window_lengths_updated.items():
                if key in kernels:
                    if kernels[key]['length'] == np.abs(kernels[key]['offset']):
                        kernels[key]['length'] = length
                        kernels[key]['offset'] = np.sign(kernels[key]['offset'])*length
                    else:
                        kernels[key]['length'] = length
                        kernels[key]['offset'] = np.sign(kernels[key]['offset'])*length/2
                else:
                    raise KeyError(f"Key {key} not found in kernels.")

    # Update orthogonalization keys
    input_ortho_keys = run_params.get('orthogonalize_against_context', [])
    if input_ortho_keys:
        ortho_keys = []
        for input_variable in input_ortho_keys:
            ortho_keys.extend(categories.get(input_variable, [input_variable]))
        for key in ortho_keys:
            if key in kernels:
                kernels[key]['orthogonalize'] = True

    # update which context kernel to use based on project
    if run_params['project'].lower() == 'templeton' and 'context' in run_params['input_variables']:
        kernels['context']['function_call'] = 'context_templeton'

    run_params['kernels'] = kernels

    return run_params


def _create_behavior_info(trials, performance, epochs):
    dprimes = performance.cross_modality_dprime.values
    return {
        'trials': trials,
        'dprime': dprimes,
        'epoch_info': epochs,
    }

def get_session_data_from_session_obj(session):
    """Fetch data from DynamicRoutingSession."""
    behavior_info = _create_behavior_info(
        trials=session.trials[:],
        performance=session.intervals['performance'][:],
        epochs=session.epochs[:],
    )
    return session.units[:], behavior_info

def get_session_data(session):
    return get_session_data_from_session_obj(session)


def get_session_data_from_datacube(
    session_id,
    lazy: bool = False,
    get_df_kwargs: dict | None = None,
    scan_nwb_kwargs: dict | None = None,
) -> tuple[pl.LazyFrame, dict[str, pd.DataFrame]]:
    nwb_path = datacube_utils.get_nwb_paths(session_id)
    behavior_info = _create_behavior_info(
        trials=lazynwb.get_df(nwb_path, '/intervals/trials', exact_path=True, **(get_df_kwargs or {})),
        performance=lazynwb.get_df(nwb_path, '/intervals/performance', exact_path=True, **(get_df_kwargs or {})),
        epochs=lazynwb.get_df(nwb_path, '/intervals/epochs', exact_path=True, **(get_df_kwargs or {})),
    )
    if lazy:
        return lazynwb.scan_nwb(nwb_path, '/units', **(scan_nwb_kwargs or {})), behavior_info
    else:
        return (
            lazynwb.get_df(
                nwb_path, "/units", exact_path=True, as_polars=True, **(get_df_kwargs or {})
            ),
            behavior_info,
        )


def get_session_data_from_cache(session_id, version='0.0.260'):

    '''
    :param session_id: ecephys session_id
    :param version: cache version
    :return: session object (if found), units_table, trials table,
                epoch information and session performance
    '''

    # to get current cache version
    # npc_lims.get_current_cache_version()
    try:
        # Attempt to load data from cached files
        behavior_info = _create_behavior_info(
            trials=pd.read_parquet(
                npc_lims.get_cache_path('trials', session_id, version=version)
            ),
            performance=pd.read_parquet(
                npc_lims.get_cache_path('performance', session_id, version=version)
            ),
            epochs=pd.read_parquet(
                npc_lims.get_cache_path('epochs', session_id, version=version)
            ),
        )
        units_table_path = npc_lims.get_cache_path("units", session_id, version=version)
        schema = pl.scan_parquet(units_table_path).collect_schema()
        units_table = pd.read_parquet(
            units_table_path, columns=[c for c in schema if c not in ['waveform_mean', 'waveform_sd',]]
        )
        return units_table, behavior_info

    except FileNotFoundError:
        # Attempt to load data from DynamicRoutingSession as a fallback
        logger.warning(f"File not found for session_id {session_id}. Attempting fallback.")
        return get_session_data_from_session_obj(session_id)

    except Exception as e:
        raise FileNotFoundError(f"Unexpected error occurred: {e}")


def setup_units_table(run_params, units_table):
    '''
        Returns the units_table with the column indicating QC
        Filters the table for area specific runs
    '''

    units_table['good_unit'] = (units_table['isi_violations_ratio'] < run_params['unit_inclusion_criteria'][
        'isi_violations']) & \
                               (units_table['presence_ratio'] > run_params['unit_inclusion_criteria'][
                                   'presence_ratio']) & \
                               (units_table['amplitude_cutoff'] < run_params['unit_inclusion_criteria'][
                                   'amplitude_cutoff']) & \
                               (units_table['firing_rate'] > run_params['unit_inclusion_criteria']['firing_rate'])

    if run_params['run_on_qc_units']:
        units_table = units_table[units_table.good_unit]

    unit_ids_to_use = run_params.get('unit_ids_to_use', [])
    if unit_ids_to_use:
        units_table = units_table[units_table.unit_id.isin(unit_ids_to_use)]

    areas_to_include = run_params.get('areas_to_include', [])
    if areas_to_include:
        units_table = units_table[units_table.structure.isin(areas_to_include)]

    areas_to_exclude = run_params.get('areas_to_exclude', [])
    if areas_to_exclude:
        units_table = units_table[~units_table.structure.isin(areas_to_exclude)]

    return units_table


def setup_trials_table(run_params, trials_table):
    '''
       Returns trials table excluding aborted trials if running encoding on quiescent period
    '''

    # TO-DO: find out how? Find out what else to include in the trials table.

    return trials_table


def get_spont_times(run_params, behavior_info):
    '''
    Returns timestamps for spontaneous period based on time of interest.
    '''
    def pick_values(start, stop, N, L):
        iti = 5  # inter-trial interval
        arr = np.arange(start, stop - L, iti + L)  # Ensure end range does not exceed the stop
        if len(arr) < N:  # Handle edge case where N exceeds possible choices
            logger.warning("Not enough intervals to pick from. Reducing number of snippets.")
            N = len(arr)
        picked_vals = np.zeros((N, 2))
        picked_vals[:, 0] = np.sort(np.random.choice(arr, size=N, replace=False))
        picked_vals[:, 1] = picked_vals[:, 0] + L
        return picked_vals

    epoch = behavior_info['epoch_info']
    if 'Spontaneous' not in epoch.script_name.values:
        logger.warning("No spontaneous activity recorded for this session.")
        return np.empty((0, 2))  # Return an empty array if no spontaneous data exists

    start_times = epoch[epoch.script_name == 'Spontaneous'].start_time.values
    stop_times = epoch[epoch.script_name == 'Spontaneous'].stop_time.values
    num_snippets = 0
    L = 0

    if 'full' in run_params['time_of_interest'] or run_params['time_of_interest'] == 'spontaneous':
        return np.column_stack((start_times, stop_times))

    elif 'trial' in run_params['time_of_interest']:
        T = run_params['spontaneous_duration']
        L = run_params['trial_stop_time'] - run_params['trial_start_time']
        num_snippets = int(T // L)

    elif 'quiescent' in run_params['time_of_interest']:
        T = run_params['spontaneous_duration']
        L = run_params['quiescent_stop_time'] - run_params['quiescent_start_time']
        num_snippets = int(T // L)

    intervals = []
    for i in range(len(start_times)):
        snippets_per_epoch = num_snippets // len(start_times)
        intervals.append(
            pick_values(start_times[i], stop_times[i], snippets_per_epoch, L)
        )

    return np.vstack(intervals)


def establish_timebins(run_params, fit, behavior_info):
    '''
    Returns the actual timestamps for each time bin that will be used in the regression model
    '''

    bin_starts = []
    epoch_trace = []
    if 'spontaneous' in run_params['time_of_interest'] or run_params['time_of_interest'] == 'full':
        spont_times = get_spont_times(run_params, behavior_info)
        for n in range(spont_times.shape[0]):
            bin_edges = np.arange(spont_times[n, 0], spont_times[n, 1], run_params['spike_bin_width'])
            bin_starts.append(bin_edges[:-1])
            epoch_trace.append([f'spontaneous{n}'] * len(bin_edges[:-1]))

    if 'full' in run_params['time_of_interest']:
        if 'trial' in run_params['time_of_interest'] or run_params['time_of_interest'] == 'full':
            start = behavior_info['trials'].start_time.values
            stop = np.append(start[1:], behavior_info['trials'].stop_time.values[-1])
            for n in range(len(behavior_info['trials'])):
                bin_edges = np.arange(start[n], stop[n], run_params['spike_bin_width'])
                bin_starts.append(bin_edges[:-1])
                epoch_trace.append([f'trial{n}'] * len(bin_edges[:-1]))

    elif 'trial' in run_params['time_of_interest']:
        start = behavior_info['trials'].stim_start_time.values + run_params['trial_start_time']
        stop = behavior_info['trials'].stim_start_time.values + run_params['trial_stop_time']
        for n in range(len(behavior_info['trials'])):
            bin_edges = np.arange(start[n], stop[n], run_params['spike_bin_width'])
            bin_starts.append(bin_edges[:-1])
            epoch_trace.append([f'trial{n}'] * len(bin_edges[:-1]))

    if 'quiescent' in run_params['time_of_interest']:
        start = behavior_info['trials'].stim_start_time.values + run_params['quiescent_start_time']
        stop = behavior_info['trials'].stim_start_time.values + run_params['quiescent_stop_time']
        for n in range(len(behavior_info['trials'])):
            bin_edges = np.arange(start[n], stop[n], run_params['spike_bin_width'])
            bin_starts.append(bin_edges[:-1])
            epoch_trace.append([f'trial{n}'] * len(bin_edges[:-1]))

    epoch_trace = np.concatenate(epoch_trace)
    bin_starts = np.concatenate(bin_starts)

    sorted_indices = np.argsort(bin_starts)
    bin_starts = bin_starts[sorted_indices]
    epoch_trace = epoch_trace[sorted_indices]

    bin_ends = bin_starts + run_params['spike_bin_width']
    timebins = np.vstack([bin_starts, bin_ends]).T

    fit['spike_bin_width'] = run_params['spike_bin_width']
    fit['timebins'] = timebins
    fit['bin_centers'] = bin_starts + run_params['spike_bin_width'] / 2
    fit['epoch_trace'] = epoch_trace


    # Extend time bins to include trace around existing time bins for time-embedding, to create a fuller trace.
    scale_factor = int(1 / run_params['spike_bin_width'])
    result = next((x for x in range(2 * scale_factor, 6 * scale_factor) if x % 1 == 0), None)
    r = result / scale_factor if result is not None else 0

    bin_starts_all = []
    epoch_trace_all = []
    for epoch in np.unique(epoch_trace):
        # Extend start by `r`
        bins = np.arange(bin_starts[epoch_trace == epoch][0] - r,
                         bin_ends[epoch_trace == epoch][-1] + r,
                         run_params['spike_bin_width'])
        bin_starts_all.append(bins)
        epoch_trace_all.append([epoch]*len(bins))

    bin_starts_all = np.concatenate(bin_starts_all, axis=0)
    epoch_trace_all = np.concatenate(epoch_trace_all)

    sorted_indices = np.argsort(bin_starts_all)
    bin_starts_all = bin_starts_all[sorted_indices]
    epoch_trace_all = epoch_trace_all[sorted_indices]

    ind = np.where(np.diff(bin_starts_all) < fit['spike_bin_width'])[0]
    bin_starts_all = np.delete(bin_starts_all, ind)
    epoch_trace_all = np.delete(epoch_trace_all, ind)

    bin_ends_all = bin_starts_all + run_params['spike_bin_width']
    timebins_all = np.vstack([bin_starts_all, bin_ends_all]).T

    if run_params['input_offsets'] and 'full' not in run_params['time_of_interest']:
        fit['timebins_all'] = timebins_all
        fit['bin_centers_all'] = bin_starts_all + run_params['spike_bin_width'] / 2
        fit['epoch_trace_all'] = epoch_trace_all
        precision = 5
        rounded_times = np.round(timebins[:, 0], precision)
        fit['mask'] = np.array([index for index, value in enumerate(timebins_all[:, 0]) if np.round(value, precision) in rounded_times])
    else:
        fit['timebins_all'] = timebins
        fit['bin_centers_all'] = bin_starts + run_params['spike_bin_width'] / 2
        fit['epoch_trace_all'] = epoch_trace
        fit['mask'] = np.arange(timebins.shape[0])


    assert len(fit['mask']) == timebins.shape[0], (
        f"Incorrect masking, length of mask ({len(fit['mask'])}) != "
        f"length of timebins ({timebins.shape[0]})."
    )
    # potentially a precision problem

    return fit


def construct_half_gaussian_kernel(std_dev, spike_bin_width):
    '''
    Returns a half-gaussian kernel
    '''
    x = np.arange(-3 * std_dev, 3 * std_dev, spike_bin_width)
    kernel = np.exp(-x ** 2 / (2 * std_dev ** 2))
    kernel[:len(kernel) // 2] = 0
    kernel = kernel / np.sum(kernel)
    return kernel


def half_gaussian_smoothing(spike_counts, kernel):
    '''
    Returns a smoothed version of spike_counts using a half-gaussian kernel
    '''
    return np.convolve(spike_counts, kernel, mode='same')


def get_spike_counts(spike_times, timebins):
    # old version ------------------------------------------------------ #
    # '''
    #     spike_times, a list of spike times, sorted
    #     timebins, numpy array of bins X start/stop
    #         timebins[i,0] is the start of bin i
    #         timbins[i,1] is the end of bin i
    # '''

    # counts = np.zeros([np.shape(timebins)[0]])
    # spike_pointer = 0
    # bin_pointer = 0
    # while (spike_pointer < len(spike_times)) & (bin_pointer < np.shape(timebins)[0]):
    #     if spike_times[spike_pointer] < timebins[bin_pointer, 0]:
    #         # This spike happens before the time bin, advance spike
    #         spike_pointer += 1
    #     elif spike_times[spike_pointer] >= timebins[bin_pointer, 1]:
    #         # This spike happens after the time bin, advance time bin
    #         bin_pointer += 1
    #     else:
    #         counts[bin_pointer] += 1
    #         spike_pointer += 1

    # return counts
    # ------------------------------------------------------------------ #
    return [b-a for a, b in np.searchsorted(spike_times, timebins)]


def process_spikes(units_table, run_params, fit):
    '''
        Returns a  dictionary including spike counts and unit-specific information.
    '''

    spikes = np.zeros((fit['timebins'].shape[0], len(units_table)))

    if run_params["smooth_spikes_half_gaussian"]:
        kernel = construct_half_gaussian_kernel(run_params["half_gaussian_std_dev"], run_params["spike_bin_width"])


    for uu, (_, unit) in tqdm(enumerate(units_table.iterrows()), total=len(units_table), desc='getting spike counts'):
        spikes[:, uu] = get_spike_counts(np.array(unit['spike_times']), fit['timebins'])
        if run_params["smooth_spikes_half_gaussian"]:
            spikes[:, uu] = half_gaussian_smoothing(spikes[:, uu], kernel)


    spike_count_arr = {
        'spike_counts': spikes,
        'bin_centers': fit['bin_centers'],
        'unit_id': units_table.unit_id.values,
    }
    fit['spike_count_arr'] = spike_count_arr

    # Check to make sure there are no NaNs in the fit_trace
    assert np.isnan(fit['spike_count_arr']['spike_counts']).sum() == 0, "Have NaNs in spike_count_arr"

    return fit


def extract_unit_data(run_params, units_table, behavior_info):
    '''
        Creates the fit dictionary
        establishes time bins
        processes spike times into spike counts for each time bin
    '''

    fit = dict()
    fit = establish_timebins(run_params, fit, behavior_info)
    fit = process_spikes(units_table, run_params, fit)

    return fit


def add_kernels(design, run_params, session, fit, behavior_info):
    '''
        Iterates through the kernels in run_params['kernels'] and adds
        each to the design matrix
        Each kernel must have fields:
            offset:
            length:

        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model
    '''

    fit['failed_kernels'] = set()
    fit['kernel_error_dict'] = dict()

    if session is None:
        import npc_sessions
        session = npc_sessions.DynamicRoutingSession(run_params["session_id"])

    for kernel_name in run_params['kernels']:
        if 'num_weights' not in run_params['kernels'][kernel_name]:
            run_params['kernels'][kernel_name]['num_weights'] = None
        design, fit = add_kernel_by_label(kernel_name, design, run_params, session, fit, behavior_info)

    return design, fit


def add_kernel_by_label(kernel_name, design, run_params, session, fit, behavior_info):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name     <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the session object for this experiment
        fit             the fit object for this model
    '''

    logger.info('    Adding kernel: ' + kernel_name)

    try:
        kernel_function = globals().get(run_params['kernels'][kernel_name]['function_call'])
        if not callable(kernel_function):
            raise ValueError(f"Invalid kernel name: {kernel_name}")
        input_x = kernel_function(kernel_name, session, fit, behavior_info)

        if run_params['kernels'][kernel_name]['type'] == 'continuous':
            input_x = standardize_inputs(input_x)

        if run_params['kernels'][kernel_name]['orthogonalize']:
            context_kernel = context('context', session, fit, behavior_info) \
                if 'context' not in design.events.keys() else design.events['context']
            input_x = orthogonalize_this_kernel(input_x, context_kernel)
            input_x = standardize_inputs(input_x)

    except Exception as e:
        logger.warning(f"Exception: {e}")
        logger.warning('Attempting to continue without this kernel.')

        fit['failed_kernels'].add(kernel_name)
        fit['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel',
            'kernel_name': kernel_name,
            'exception': e.args[0],
        }
        return design, fit
    else:
        design.add_kernel(
            input_x,
            run_params['kernels'][kernel_name]['length'],
            kernel_name,
            offset=run_params['kernels'][kernel_name]['offset'],
            num_weights=run_params['kernels'][kernel_name]['num_weights']
        )
    return design, fit


def intercept(kernel_name, session, fit, behavior_info):
    return np.ones(len(fit['bin_centers_all']))


def context(kernel_name, session, fit, behavior_info):
    this_kernel = np.zeros(len(fit['bin_centers_all']))
    epoch_trace = fit['epoch_trace_all']

    for n, epoch in enumerate(epoch_trace):
        if 'trial' in epoch:
            trial_no = int(''.join(filter(str.isdigit, epoch)))
            this_kernel[n] = 1 if behavior_info['trials'].loc[trial_no, 'is_vis_rewarded'] else -1

    return this_kernel


def context_templeton(kernel_name, session, fit, behavior_info):
    this_kernel = np.zeros(len(fit['bin_centers_all']))
    epoch_trace = fit['epoch_trace_all']
    trial_indexes = [n for n, epoch in enumerate(epoch_trace) if 'trial' in epoch]

    # find index corresponding to passage of every 10 minutes from fit['bin_centers_all']
    trial_times = fit['bin_centers_all'][trial_indexes]
    block_length = 10 * 60  # 10 minutes in seconds
    switch_times = trial_times[0] + np.arange(0, block_length*6, block_length)
    signed_context = np.random.choice([-1, 1], size=1)[0]
    for i in range(1, len(switch_times)):
        this_kernel[np.where((fit['bin_centers_all'] >= switch_times[i-1]) & (fit['bin_centers_all'] < switch_times[i]))] = signed_context
        signed_context = -signed_context
    # Handle the last segment
    this_kernel[np.where((fit['bin_centers_all'] >= switch_times[-1]) & (fit['bin_centers_all'] <= trial_times[-1]))] = signed_context

    return this_kernel


@typing.overload
def _datacube_data(session_id: str, internal_path: str, is_timeseries: Literal[False]) -> pd.DataFrame:
    ...

@typing.overload
def _datacube_data(session_id: str, internal_path: str, is_timeseries: Literal[True]) -> lazynwb.TimeSeries:
    ...

def _datacube_data(session_id: str, internal_path: str, is_timeseries: bool = False) -> pd.DataFrame | lazynwb.TimeSeries:
    if is_timeseries:
        return lazynwb.get_timeseries(datacube_utils.get_nwb_paths(session_id), internal_path, exact_path=True, match_all=False)
    else:
        return lazynwb.get_df(datacube_utils.get_nwb_paths(session_id), internal_path, exact_path=True)


def pupil(kernel_name, session, fit, behavior_info):
    def process_pupil_data(df, behavior_info):
        for pos, row in behavior_info['epoch_info'].iterrows():
            # Select rows within the current epoch
            epoch_mask = (df.timestamps >= row.start_time) & (df.timestamps < row.stop_time)
            epoch_df = df.loc[epoch_mask]

            # Compute the threshold for the current epoch
            threshold = np.nanmean(epoch_df['pupil_area']) + 3 * np.nanstd(epoch_df['pupil_area'])

            # Apply threshold and set outliers to NaN within the epoch
            df.loc[epoch_mask & (df['pupil_area'] > threshold), 'pupil_area'] = np.nan
        df['pupil_area'] = df['pupil_area'].interpolate(method='linear')
        df['pupil_area'] = df['pupil_area'].ffill()
        df['pupil_area'] = df['pupil_area'].bfill()
        return df

    if isinstance(session, str) and datacube_utils.is_datacube_available():
        df = process_pupil_data(_datacube_data(session, '/processing/behavior/eye_tracking'), behavior_info)
    else:
        df = process_pupil_data(session.processing['behavior']['eye_tracking'][:], behavior_info)
    this_kernel = bin_timeseries(df.pupil_area.values, df.timestamps.values, fit['timebins_all'])
    this_kernel = pd.Series(this_kernel).ffill().bfill().to_numpy()
    if np.isnan(this_kernel).all():
        raise ValueError(f"The trace is all nans for {kernel_name}")
    return this_kernel


def running(kernel_name, session, fit, behavior_info):
    if isinstance(session, str) and datacube_utils.is_datacube_available():
        timeseries = _datacube_data(session, '/processing/behavior/running_speed', is_timeseries=True)
    else:
        timeseries = session.processing['behavior']['running_speed']
    this_kernel = bin_timeseries(timeseries.data[:], timeseries.timestamps[:], fit['timebins_all'])
    this_kernel = pd.Series(this_kernel).ffill().bfill().to_numpy()
    if np.isnan(this_kernel).all():
        raise ValueError(f"The trace is all nans for {kernel_name}")
    return this_kernel


def licks(kernel_name, session, fit, behavior_info):
    if isinstance(session, str) and datacube_utils.is_datacube_available():
        timeseries = _datacube_data(session, '/processing/behavior/licks', is_timeseries=True)
    else:
        timeseries = session.processing['behavior']['licks']
    lick_times = timeseries.timestamps[:]
    lick_duration = timeseries.data[:]
    lick_duration_threshold = 0.5
    lick_times = lick_times[lick_duration < lick_duration_threshold]

    # Extract the bin edges
    bin_starts, bin_stops = fit['timebins_all'][:, 0], fit['timebins_all'][:, 1]

    # Check if any lick times are within each bin
    in_bin = (lick_times[:, None] >= bin_starts) & (lick_times[:, None] < bin_stops)
    this_kernel = np.any(in_bin, axis=0).astype(int)

    if np.isnan(this_kernel).all():
        raise ValueError(f"The trace is all nans for {kernel_name}")
    return this_kernel


def facial_features(kernel_name, session, fit, behavior_info):
    def eu_dist_for_LP(x_c, y_c):
        return np.sqrt(x_c ** 2 + y_c ** 2)

    def part_info_LP(part_name, df):
        confidence = df[part_name + '_likelihood'].values.astype('float')
        temp_norm = df[part_name + '_temporal_norm'].values.astype('float')
        x_c = df[part_name + '_x'].values.astype('float')
        y_c = df[part_name + '_y'].values.astype('float')
        xy = eu_dist_for_LP(x_c, y_c)

        xy[(confidence < 0.98) | (temp_norm > np.nanmean(temp_norm) + 3 * np.nanstd(temp_norm))] = np.nan
        xy = pd.Series(xy).interpolate(limit_direction='both').to_numpy()
        return xy, confidence

    map_names = {'ears': 'ear_base_l', 'jaw': 'jaw', 'nose': 'nose_tip', 'whisker_pad': 'whisker_pad_l_side'}
    if isinstance(session, str) and datacube_utils.is_datacube_available():
        try:
            df = _datacube_data(session, '/processing/behavior/lp_side_camera')
        except KeyError:
            raise IndexError(f'{session} is not a session with video.')
    else:
        try:
            df = session.processing['behavior']['lp_side_camera'][:]
        except IndexError:
            raise IndexError(f'{session.id} is not a session with video.')
    timestamps = df['timestamps'].values.astype('float')
    lp_part_name = map_names[kernel_name]
    part_xy, confidence = part_info_LP(lp_part_name, df)
    this_kernel = bin_timeseries(part_xy, timestamps, fit['timebins_all'])
    this_kernel = pd.Series(this_kernel).ffill().bfill().to_numpy()
    if np.isnan(this_kernel).all():
        raise ValueError(f"The trace is all nans for {kernel_name}")
    return this_kernel


def choice(kernel_name, session, fit, behavior_info):
    bin_starts, bin_stops = fit['timebins_all'][:, 0], fit['timebins_all'][:, 1]
    if behavior_info['trials']['is_' + kernel_name].any():
        choice_times = behavior_info['trials'][behavior_info['trials']['is_' + kernel_name]].stim_start_time.values
        in_bin = (choice_times[:, None] >= bin_starts) & (choice_times[:, None] < bin_stops)
        this_kernel = np.any(in_bin, axis=0).astype(int)
    else:
        raise ValueError(f"No trials with is_{kernel_name}")
    return this_kernel


def stimulus(kernel_name, session, fit, behavior_info):
    if '_' in kernel_name:
        stim_name, rewarded_modality = kernel_name.split('_')
        filtered_trials = behavior_info['trials'][
            (behavior_info['trials'].stim_name == stim_name) & (behavior_info['trials'].rewarded_modality == rewarded_modality)]
    else:
        stim_name = kernel_name
        rewarded_modality = 'all'
        filtered_trials = behavior_info['trials'][behavior_info['trials'].stim_name == stim_name]
    bin_starts, bin_stops = fit['timebins_all'][:, 0], fit['timebins_all'][:, 1]

    if not filtered_trials.empty:
        stim_times = filtered_trials.stim_start_time.values
        in_bin = (stim_times[:, None] >= bin_starts) & (stim_times[:, None] < bin_stops)
        this_kernel = np.any(in_bin, axis=0).astype(int)
        # Ensure no consecutive 1s
        for i in range(1, len(this_kernel)):
            if this_kernel[i] == 1 and this_kernel[i - 1] == 1:
                this_kernel[i] = 0
    else:
        raise ValueError(f"No trials presented with {stim_name} stimulus in {rewarded_modality} context")
    return this_kernel


def session_time(kernel_name, session, fit, behavior_info):
    return fit['bin_centers_all']


def toeplitz_this_kernel(input_x, kernel_length_samples, offset_samples):
    '''
    Build a toeplitz matrix aligned to events.

    Args:
        events (np.array of 1/0): Array with 1 if the event happened at that time, and 0 otherwise.
        kernel_length_samples (int): How many kernel parameters
    Returns
        np.array, size(len(events), kernel_length_samples) of 1/0
    '''

    total_len = len(input_x)
    events = np.concatenate([input_x, np.zeros(kernel_length_samples)])
    arrays_list = [events]
    for i in range(kernel_length_samples - 1):
        arrays_list.append(np.roll(events, i + 1))
    this_kernel = np.vstack(arrays_list)

    # pad with zeros, roll offset_samples, and truncate to length
    if offset_samples < 0:
        this_kernel = np.concatenate([np.zeros((this_kernel.shape[0], np.abs(offset_samples))), this_kernel], axis=1)
        this_kernel = np.roll(this_kernel, offset_samples)[:, np.abs(offset_samples):]
    elif offset_samples > 0:
        this_kernel = np.concatenate([this_kernel, np.zeros((this_kernel.shape[0], offset_samples))], axis=1)
        this_kernel = np.roll(this_kernel, offset_samples)[:, :-offset_samples]
    return this_kernel[:, :total_len]


class DesignMatrix:
    def __init__(self, fit):
        '''
        A toeplitz-matrix builder for running regression with multiple temporal kernels.

        Args
            fit_dict, a dictionary with:
                event_timestamps: The actual timestamps for each time bin that will be used in the regression model.
        '''
        self.X = None
        self.kernel_dict = {}
        self.running_stop = 0
        self.events = {'timestamps': fit['bin_centers']}
        self.spike_bin_width = fit['spike_bin_width']
        self.epochs = fit['epoch_trace']
        self.mask = fit['mask']

    def make_labels(self, label, num_weights, offset, length):
        base = [label] * num_weights
        numbers = [str(x) for x in np.array(range(0, length)) + offset]
        return [x[0] + '_' + x[1] for x in zip(base, numbers)]

    def add_kernel(self, input_x, kernel_length, label, offset=0, num_weights=None):
        '''
        Add a temporal kernel.

        Args:
            input_x (np.array): input data points
            kernel_length (int): length of the kernel (in SECONDS).
            label (string): Name of the kernel.
            offset (int) :offset relative to the events. Negative offsets cause the kernel
                          to overhang before the event (in SECONDS)
        '''

        # Enforce unique labels
        if label in self.kernel_dict.keys():
            raise ValueError('Labels must be unique')

        self.events[label] = input_x[self.mask]

        # CONVERT kernel_length to kernel_length_samples
        if num_weights is None:
            if kernel_length == 0:
                kernel_length_samples = 1
            else:
                kernel_length_samples = int(np.ceil((1 / self.spike_bin_width) * kernel_length))
        else:
            # Some kernels are hard-coded by number of weights
            kernel_length_samples = num_weights

        # CONVERT offset to offset_samples
        offset_samples = int(np.floor((1 / self.spike_bin_width) * offset))

        this_kernel = toeplitz_this_kernel(input_x, kernel_length_samples, offset_samples)

        # keep only the relevant trace for prediction
        this_kernel = this_kernel[:, self.mask]

        self.kernel_dict[label] = {
            'kernel': this_kernel,
            'kernel_length_samples': kernel_length_samples,
            'offset_samples': offset_samples,
            'kernel_length_seconds': kernel_length,
            'offset_seconds': offset,
            'ind_start': self.running_stop,
            'ind_stop': self.running_stop + kernel_length_samples
        }
        self.running_stop += kernel_length_samples

    def get_X(self, kernels=None):
        '''
        Get the design matrix.
        Args:
            kernels (optional list of kernel string names): which kernels to include (for model selection)
        Returns:
            X (xr.array): The design matrix
        '''
        if kernels is None:
            kernels = self.kernel_dict.keys()

        kernels_to_use = []
        param_labels = []
        for kernel_name in kernels:
            kernels_to_use.append(self.kernel_dict[kernel_name]['kernel'])
            param_labels.append(self.make_labels(kernel_name,
                                                 np.shape(self.kernel_dict[kernel_name]['kernel'])[0],
                                                 self.kernel_dict[kernel_name]['offset_samples'],
                                                 self.kernel_dict[kernel_name]['kernel_length_samples']))

        X = np.vstack(kernels_to_use)
        x_labels = np.hstack(param_labels)

        assert np.shape(X)[0] == np.shape(x_labels)[0], 'Weight Matrix must have the same length as the weight labels'

        X_array = xr.DataArray(
            X,
            dims=('weights', 'timestamps'),
            coords={'weights': x_labels,
                    'timestamps': self.events['timestamps']}
        )
        self.X = X_array.T
        return X_array.T


def bin_timeseries(x, x_timestamps, timebins_all):
    # Remove NaN values from x_timestamps and corresponding x values
    valid_indices = ~np.isnan(x_timestamps)
    x_timestamps = x_timestamps[valid_indices]
    x = x[valid_indices]

    Tm = timebins_all.shape[0]
    start_indices = np.searchsorted(x_timestamps, timebins_all[:, 0], side='left')
    end_indices = np.searchsorted(x_timestamps, timebins_all[:, 1], side='right')

    binned = np.full(Tm, np.nan)

    for t_i in range(Tm):
        indices = slice(start_indices[t_i], end_indices[t_i])
        if indices.start < indices.stop:  # Check if the slice is non-empty
            binned[t_i] = np.nanmean(x[indices])
        elif t_i > 0:  # Propagate previous value
            binned[t_i] = binned[t_i - 1]
    return binned


def orthogonalize_this_kernel(this_kernel, y):
    mat_to_ortho = np.concatenate((y.reshape(-1, 1), this_kernel.reshape(-1, 1)), axis=1)
    logger.info('                 : ' + 'orthogonalizing against context')
    Q, R = np.linalg.qr(mat_to_ortho)
    return Q[:, 1]


def standardize_inputs(timeseries, mean_center=True, unit_variance=True, max_value=None):
    '''
        Performs three different input standarizations to the timeseries
        if mean_center, the timeseries is adjusted to have 0-mean. This can be performed with unit_variance.
        if unit_variance, the timeseries is adjusted to have unit variance. This can be performed with mean_center.
        if max_value is given, then the timeseries is normalized by max_value. This cannot be performed with mean_center and unit_variance.
    '''
    if (max_value is not None) & (mean_center or unit_variance):
        raise Exception(
            'Cannot perform max_value standardization and mean_center or unit_variance standardizations together.')

    if mean_center:
        logger.info('                 : ' + 'mean centering')
        timeseries = timeseries - np.mean(timeseries)  # mean center
    if unit_variance:
        logger.info('                 : ' + 'standardized to unit variance')
        timeseries = timeseries / np.std(timeseries)
    if max_value is not None:
        logger.info('                 : ' + 'normalized by max value: ' + str(max_value))
        timeseries = timeseries / max_value

    return timeseries
