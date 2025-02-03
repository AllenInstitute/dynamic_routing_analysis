import logging
import time

import npc_lims
import npc_sessions
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__) # debug < info < warning < error
pd.set_option('display.max_columns', None)
np.random.seed(0)


class RunParams:
    def __init__(self, session_id):
        self.run_params = {
            "session_id": session_id,
            "time_of_interest": 'quiescent',
            "spontaneous_duration": 2 * 60,  # in seconds
            "input_variables": None,
            "input_offsets": True,
            "input_window_lengths": None,  # offset
            "drop_variables": None,
            "unit_inclusion_criteria": {'isi_violations': 0.1,
                                        'presence_ratio': 0.99,
                                        'amplitude_cutoff': 0.1,
                                        'firing_rate': 1},
            "run_on_qc_units": True,
            "spike_bin_width": 0.025,
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

    def define_kernels(self):
        '''
            Returns kernel info for input variables
        '''

        # Define master kernel list
        master_kernels_list = {
            'intercept': {'function_call': 'intercept', 'type': 'discrete', 'length': 0, 'offset': 0,
                        'orthogonalize': None, 'num_weights': None, 'dropout': True, 'text': 'constant value'},
            'vis1_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'target stim in rewarded context'},
            'sound1_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'target stim in non-rewarded context'},
            'vis2_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'non-target stim in vis context'},
            'sound2_vis': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'non-target stim in vis context'},
            'vis1_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'target stim in non-rewarded context'},
            'sound1_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'target stim in rewarded context'},
            'vis2_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'non-target stim in aud context'},
            'sound2_aud': {'function_call': 'stimulus', 'type': 'discrete', 'length': 1, 'offset': 1, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'non-target stim in aud context'},
            'nose': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                    'orthogonalize': None, 'num_weights': None, 'dropout': True,
                    'text': 'Z-scored Euclidean displacement of nose movements'},
            'ears': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                    'orthogonalize': None, 'num_weights': None, 'dropout': True,
                    'text': 'Z-scored Euclidean displacement of ear movements'},
            'jaw': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                    'orthogonalize': None, 'num_weights': None, 'dropout': True,
                    'text': 'Z-scored Euclidean displacement of jaw movements'},
            'whisker_pad': {'function_call': 'facial_features', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                            'orthogonalize': None, 'num_weights': None, 'dropout': True,
                            'text': 'Z-scored Euclidean displacement of whisker pad movements'},
            'licks': {'function_call': 'licks', 'type': 'discrete', 'length': 1, 'offset': -0.5, 'orthogonalize': None,
                    'num_weights': None, 'dropout': True, 'text': 'lick responses'},
            'running': {'function_call': 'running', 'type': 'continuous', 'length': 1, 'offset': -0.5,
                        'orthogonalize': None, 'num_weights': None, 'dropout': True, 'text': 'Z-scored running speed'},
            'pupil': {'function_call': 'pupil', 'type': 'continuous', 'length': 1, 'offset': -0.5, 'orthogonalize': None,
                    'num_weights': None, 'dropout': True, 'text': 'Z-scored pupil diameter'},
            'hit': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5, 'orthogonalize': None,
                    'num_weights': None, 'dropout': True, 'text': 'lick to GO trial'},
            'miss': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5, 'orthogonalize': None,
                    'num_weights': None, 'dropout': True, 'text': 'no lick to GO trial'},
            'correct_reject': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5,
                            'orthogonalize': None, 'num_weights': None, 'dropout': True,
                            'text': 'no lick to NO-GO trial'},
            'false_alarm': {'function_call': 'choice', 'type': 'discrete', 'length': 3, 'offset': -1.5,
                            'orthogonalize': None, 'num_weights': None, 'dropout': True, 'text': 'lick to NO-GO trial'},
            'context': {'function_call': 'context', 'type': 'discrete', 'length': 0, 'offset': 0, 'orthogonalize': None,
                        'num_weights': None, 'dropout': True, 'text': 'block-wise context'},
            'session_time': {'function_call': 'session_time', 'type': 'continuous', 'length': 0, 'offset': 0,
                            'orthogonalize': None, 'num_weights': None, 'dropout': True,
                            'text': 'z-scored time in session'}
        }

        # Define categories for input variables
        categories = {
            'stimulus': ['vis1_vis', 'sound1_vis', 'vis2_vis', 'sound2_vis', 'vis1_aud', 'sound1_aud', 'vis2_aud',
                        'sound2_aud'],
            'movements': ['ears', 'nose', 'jaw', 'whisker_pad', 'running', 'pupil', 'licks'],
            'movements_no_licks': ['ears', 'nose', 'jaw', 'whisker_pad', 'running', 'pupil'],
            'choice': ['hit', 'miss', 'correct_reject', 'false_alarm'],
            'facial_features': ['ears', 'nose', 'jaw', 'whisker_pad']
        }

        # Initialize selected keys list
        selected_keys = []

        # Determine selected input variables based on run_params
        time_of_interest = self.run_params.get('time_of_interest', '')
        input_variables = self.run_params.get('input_variables', [])

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
        if self.run_params.get('intercept', False) and 'intercept' not in selected_keys:
            selected_keys.append('intercept')

        # Log error if no input variables are selected
        if not selected_keys:
            raise ValueError("No input variables selected!") # raise value error .

        # remove drop variables if any
        drop_keys = self.run_params.get('drop_variables', [])
        if drop_keys and self.run_params['model_label'] != 'fullmodel':
            for drop_key in drop_keys:
                sub_keys = categories.get(drop_key, [drop_key])
                for sub_key in sub_keys:
                    logger.info(get_timestamp() + f': dropping {sub_key}')
                    selected_keys.remove(sub_key)

        # Build kernels dictionary based on selected keys
        kernels = {key: master_kernels_list[key] for key in selected_keys}

        # Update kernel lengths based on run_params
        if not self.run_params.get('input_offsets'):
            for key, kernel in kernels.items():
                kernel['length'] = 0
                kernel['offset'] = 0
        else:
            input_window_lengths = self.run_params.get('input_window_lengths', {})
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
        input_ortho_keys = self.run_params.get('orthogonalize_against_context', [])
        if input_ortho_keys:
            ortho_keys = []
            for input_variable in input_ortho_keys:
                ortho_keys.extend(categories.get(input_variable, [input_variable]))
            for key in ortho_keys:
                if key in kernels:
                    kernels[key]['orthogonalize'] = True

        self.run_params['kernels'] = kernels


def get_session_data(session):
    """Fetch data from DynamicRoutingSession if files are not found."""
    try:
        trials = session.trials[:]
        dprimes = np.array(session.performance.cross_modal_dprime[:])
        epoch = session.epochs[:]
        behavior_info = {'trials': trials,
                         'dprime': dprimes,
                         'is_good_behavior': np.count_nonzero(dprimes >= 1) >= 4,
                         'epoch_info': epoch}
        units_table = session.units[:]
        return session, units_table, behavior_info, None
    except Exception as e:
        raise FileNotFoundError(f"Failed to load data from DynamicRoutingSession: {e}")


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
        trials = pd.read_parquet(
            npc_lims.get_cache_path('trials', session_id, version=version)
        )
        dprimes = np.array(pd.read_parquet(
            npc_lims.get_cache_path('performance', session_id, version=version)
        ).cross_modal_dprime.values)
        epoch = pd.read_parquet(
            npc_lims.get_cache_path('epochs', session_id, version=version)
        )
        behavior_info = {'trials': trials,
                         'dprime': dprimes,
                         'is_good_behavior': np.count_nonzero(dprimes >= 1) >= 4,
                         'epoch_info': epoch}

        units_table = pd.read_parquet(
            npc_lims.get_cache_path('units', session_id, version=version)
        )
        return None, units_table, behavior_info, None

    except FileNotFoundError:
        # Attempt to load data from DynamicRoutingSession as a fallback
        logger.warning(f"File not found for session_id {session_id}. Attempting fallback.")
        return get_session_data(session_id)

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
    r = result / scale_factor if result is not None else None

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

    bin_ends_all = bin_starts_all + run_params['spike_bin_width']
    timebins_all = np.vstack([bin_starts_all, bin_ends_all]).T

    fit['timebins_all'] = timebins_all
    fit['bin_centers_all'] = bin_starts_all + run_params['spike_bin_width'] / 2
    fit['epoch_trace_all'] = epoch_trace_all
    precision = 5
    rounded_times = np.round(timebins[:, 0], precision)
    fit['mask'] = np.array([index for index, value in enumerate(timebins_all[:, 0]) if np.round(value, precision) in rounded_times])

    assert len(fit['mask']) == timebins.shape[0], 'Incorrect masking, recheck timebins.'
    # potentially a precision problem

    return fit


def get_spike_counts(spike_times, timebins):
    '''
        spike_times, a list of spike times, sorted
        timebins, numpy array of bins X start/stop
            timebins[i,0] is the start of bin i
            timbins[i,1] is the end of bin i
    '''

    counts = np.zeros([np.shape(timebins)[0]])
    spike_pointer = 0
    bin_pointer = 0
    while (spike_pointer < len(spike_times)) & (bin_pointer < np.shape(timebins)[0]):
        if spike_times[spike_pointer] < timebins[bin_pointer, 0]:
            # This spike happens before the time bin, advance spike
            spike_pointer += 1
        elif spike_times[spike_pointer] >= timebins[bin_pointer, 1]:
            # This spike happens after the time bin, advance time bin
            bin_pointer += 1
        else:
            counts[bin_pointer] += 1
            spike_pointer += 1

    return counts


def process_spikes(units_table, run_params, fit):
    '''
        Returns a  dictionary including spike counts and unit-specific information.
    '''

    # identifies good units
    units_table = setup_units_table(run_params, units_table)

    spikes = np.zeros((fit['timebins'].shape[0], len(units_table)))

    for uu, (_, unit) in tqdm(enumerate(units_table.iterrows()), total=len(units_table), desc='getting spike counts'):
        spikes[:, uu] = get_spike_counts(np.array(unit['spike_times']), fit['timebins'])

    spike_count_arr = {
        'spike_counts': spikes,
        'bin_centers': fit['bin_centers'],
        'unit_id': units_table.unit_id.values,
        'structure': units_table.structure.values,
        'location': units_table.location.values,
        'quality':  units_table.good_unit.values,
        'firing_rate': units_table.firing_rate.values
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

    logger.info(get_timestamp() + '    Adding kernel: ' + kernel_name)

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
        logger.warning(get_timestamp() + f"Exception: {e}")
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
            this_kernel[n] = 1 if behavior_info['trials'].loc[trial_no, 'is_vis_context'] else -1

    return this_kernel


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
        return df

    df = process_pupil_data(session._eye_tracking.to_dataframe(), behavior_info)
    this_kernel = bin_timeseries(df.pupil_area.values, df.timestamps.values, fit['timebins_all'])
    if np.isnan(this_kernel).all():
        raise ValueError(f"The trace is all nans for {kernel_name}")
    return this_kernel


def running(kernel_name, session, fit, behavior_info):
    this_kernel = bin_timeseries(session._running_speed.data, session._running_speed.timestamps, fit['timebins_all'])
    if np.isnan(this_kernel).all():
        raise ValueError(f"The trace is all nans for {kernel_name}")
    return this_kernel


def licks(kernel_name, session, fit, behavior_info):
    lick_times = session._all_licks[0].timestamps

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
    try:
        df = session._lp[0][:]
    except IndexError:
        raise IndexError(f'{session.id} is not a session with video.')
    timestamps = df['timestamps'].values.astype('float')
    lp_part_name = map_names[kernel_name]
    part_xy, confidence = part_info_LP(lp_part_name, df)
    this_kernel = bin_timeseries(part_xy, timestamps, fit['timebins_all'])

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
    stim_name, context_name = kernel_name.split('_')
    bin_starts, bin_stops = fit['timebins_all'][:, 0], fit['timebins_all'][:, 1]
    filtered_trials = behavior_info['trials'][
        (behavior_info['trials'].stim_name == stim_name) & (behavior_info['trials'].context_name == context_name)]
    if not filtered_trials.empty:
        stim_times = filtered_trials.stim_start_time.values
        in_bin = (stim_times[:, None] >= bin_starts) & (stim_times[:, None] < bin_stops)
        this_kernel = np.any(in_bin, axis=0).astype(int)
    else:
        raise ValueError(f"No trials presented with {stim_name} stimulus in {context_name} context")
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

        if np.abs(offset_samples) > 0:
            this_kernel = toeplitz_this_kernel(input_x, kernel_length_samples, offset_samples)
        else:
            this_kernel = input_x.reshape(1, -1)

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


def get_timestamp():
    t = time.localtime()
    return time.strftime('%Y-%m-%d: %H:%M:%S') + ' '


def orthogonalize_this_kernel(this_kernel, y):
    mat_to_ortho = np.concatenate((y.reshape(-1, 1), this_kernel.reshape(-1, 1)), axis=1)
    logger.info(get_timestamp() + '                 : ' + 'othogonalizing against context')
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
        logger.info(get_timestamp() + '                 : ' + 'mean centering')
        timeseries = timeseries - np.mean(timeseries)  # mean center
    if unit_variance:
        logger.info(get_timestamp() + '                 : ' + 'standardized to unit variance')
        timeseries = timeseries / np.std(timeseries)
    if max_value is not None:
        logger.info(get_timestamp() + '                 : ' + 'normalized by max value: ' + str(max_value))
        timeseries = timeseries / max_value

    return timeseries
