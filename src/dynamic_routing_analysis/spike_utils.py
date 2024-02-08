import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import xarray as xr

#functions for making 3d trial-aligned tensor

def makePSTH(spike_times, event_times, time_before, time_after, bin_size):
    
    #spike_times: array of spike times
    #event_time: time of event to create PSTH around
    #time_before: time before event to include in PSTH
    #time_after: time after event to include in PSTH
    #bin_size: size of each bin in seconds
    #returns: event-aligned spikes (time x trials), bin_edges

    bins = np.arange(-time_before, time_after, bin_size)
    event_aligned_spikes=[]

    for event_time in event_times:
        sel_spike_times = spike_times[(spike_times >= event_time - time_before) & (spike_times < event_time + time_after)]-event_time
        spike_histogram, bin_edges = np.histogram(sel_spike_times, bins = bins)

        event_aligned_spikes.append(spike_histogram)

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    return np.vstack(event_aligned_spikes).T, bin_centers


def make_neuron_time_trials_tensor(units, trials, time_before, time_after, bin_size, event_name='stim_start_time'):
    
    #units: units to include in tensor
    #trials: trials to include in tensor
    #time_before: time before event to include in PSTH
    #time_after: time after event to include in PSTH
    #bin_size: size of each bin in seconds
    #event_name: optional input to specify column to use to align events
    #returns: 3d tensor of shape (units, time, trials)

    unit_count = len(units[:])
    trial_count = len(trials[:])
    bins = np.arange(-time_before, time_after, bin_size)

    tensor = np.zeros((unit_count, len(bins)-1, trial_count))

    for uu, (_, unit) in enumerate(units[:].iterrows()):
        spike_times = np.array(unit['spike_times'])
        event_times = trials[:][event_name]
        event_aligned_spikes, bin_centers = makePSTH(spike_times, event_times, time_before, time_after, bin_size)
        tensor[uu,:,:] = event_aligned_spikes/bin_size

    trial_da = xr.DataArray(tensor, dims=("unit_id", "time", "trials"), 
                            coords={
                                "unit_id": units[:]['unit_id'].values,
                                "time": bin_centers,
                                "trials": trials[:].index.values
                                })

    return trial_da


def make_timebins_table(trials, bin_size):

    start_time = trials[:]['start_time'].iloc[0]
    end_time = trials[:]['stop_time'].iloc[-1]

    bins = np.arange(start_time, end_time, bin_size)

    bin_centers = (bins[:-1] + bins[1:])/2

    timebins_table={
        'bin_start':bins[:-1],
        'bin_end':bins[1:],
        'bin_center':bin_centers,
        'stim_start':np.zeros(len(bin_centers),dtype=bool),
        'stim_stop':np.zeros(len(bin_centers),dtype=bool),
        'reward':np.zeros(len(bin_centers),dtype=bool),
        'is_vis_context':np.zeros(len(bin_centers),dtype=bool),
        'is_aud_context':np.zeros(len(bin_centers),dtype=bool),
        'is_vis_stim':np.zeros(len(bin_centers),dtype=bool),
        'is_aud_stim':np.zeros(len(bin_centers),dtype=bool),
        'is_vis_target':np.zeros(len(bin_centers),dtype=bool),
        'is_aud_target':np.zeros(len(bin_centers),dtype=bool),
        'is_catch':np.zeros(len(bin_centers),dtype=bool),
        'is_vis_nontarget':np.zeros(len(bin_centers),dtype=bool),
        'is_aud_nontarget':np.zeros(len(bin_centers),dtype=bool),
        'is_context_switch':np.zeros(len(bin_centers),dtype=bool),
    }

    #set context
    context_switches=trials[:].query('is_context_switch')

    context_switch_trial_index=np.hstack([0,context_switches['trial_index'].values,len(trials[:])-1])

    for ii,switch_trial_num in enumerate(context_switch_trial_index[:-1]):
        block_start=trials.loc[switch_trial_num]['start_time']
        block_end=trials.loc[context_switch_trial_index[ii+1]]['stop_time']

        if trials['is_vis_context'].loc[switch_trial_num]:
            timebins_table['is_vis_context'][(timebins_table['bin_start']>=block_start) & 
                                            (timebins_table['bin_start']<block_end)]=True
            timebins_table['is_aud_context'][(timebins_table['bin_start']>=block_start) & 
                                            (timebins_table['bin_start']<block_end)]=False
        elif trials['is_aud_context'].loc[switch_trial_num]:
            timebins_table['is_aud_context'][(timebins_table['bin_start']>=block_start) & 
                                            (timebins_table['bin_start']<block_end)]=True
            timebins_table['is_vis_context'][(timebins_table['bin_start']>=block_start) & 
                                            (timebins_table['bin_start']<block_end)]=False
        if ii > 0:    
            timebins_table['is_context_switch'][(timebins_table['bin_start']>=block_start) &
                                                (timebins_table['bin_start']<block_start+bin_size)]=True


    #set reward
    reward_trials=trials[:].query('is_rewarded')

    for rr,reward_trial in reward_trials.iterrows():
        timebins_table['reward'][(timebins_table['bin_start']>=reward_trial['reward_time']) & 
                                (timebins_table['bin_start']<reward_trial['reward_time']+bin_size)]=True
        
        
    #set stimuli
    for tt,trial in trials[:].iterrows():
        if trial['is_vis_stim']:
            timebins_table['is_vis_stim'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                        (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
            if trial['is_target']:
                timebins_table['is_vis_target'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                                (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
            elif trial['is_nontarget']:
                timebins_table['is_vis_nontarget'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                                (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
        elif trial['is_aud_stim']:
            timebins_table['is_aud_stim'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                        (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
            if trial['is_target']:
                timebins_table['is_aud_target'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                                (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
            elif trial['is_nontarget']:
                timebins_table['is_aud_nontarget'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                                (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
        elif trial['is_catch']:
            timebins_table['is_catch'][(timebins_table['bin_start']>=trial['stim_start_time']) & 
                                        (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
            
        timebins_table['stim_start'][(timebins_table['bin_start']>=trial['stim_start_time']) &
                                    (timebins_table['bin_start']<trial['stim_start_time']+bin_size)]=True
        timebins_table['stim_stop'][(timebins_table['bin_start']>=trial['stim_stop_time']) &
                                    (timebins_table['bin_start']<trial['stim_stop_time']+bin_size)]=True


    return pd.DataFrame.from_dict(timebins_table),bins



def make_neuron_timebins_matrix(units, trials, bin_size, generate_context_labels=False):
    
    #units: units table to include in matrix
    #trials: trials table to create timebins table
    #bin_size: size of each bin in seconds
    
    # generate 10-minute blocks of context labels
    if generate_context_labels:
        start_time=trials[:]['start_time'].iloc[0]
        block_context_names=np.array(['vis','aud'])
        context=np.full(len(trials), fill_value='nan')
        block_nums=np.full(len(trials), fill_value=np.nan)

        # make "real" subdivided blocks
        if np.random.choice(block_context_names,1)=='vis':
            block_context_index=[0,1]*3
        #elif np.random.choice(block_context_names,1)=='aud': #sometimes this if & elif aren't reached, IDK why
        else:
            block_context_index=[1,0]*3
        block_contexts=block_context_names[block_context_index]
        for block in range(0,6):
            block_start_time=start_time+block*600
            block_trials=trials.query('start_time>=@block_start_time').index
            context[block_trials]=block_contexts[block]
            block_nums[block_trials]=block
        trials['context_name']=context
        context_switches=np.where(np.diff(trials['context_name'].values=='vis'))[0]+1
        trials['is_context_switch']=np.full(len(trials), fill_value=False)
        trials['is_context_switch'].iloc[context_switches]=True
        trials['is_vis_context']=context=='vis'
        trials['is_aud_context']=context=='aud'

    timebins_table,bins = make_timebins_table(trials, bin_size)

    unit_count = len(units[:])
    timebin_count = len(timebins_table)

    matrix = np.zeros((unit_count, timebin_count))

    for uu, unit in units[:].iterrows():
        spike_times = np.array(unit['spike_times'])
        # event_times = timebins_table['bin_start']
        # event_aligned_spikes, bin_centers = makePSTH(spike_times, event_times, time_before, time_after, bin_size)
        event_aligned_spikes, bin_edges = np.histogram(spike_times, bins=bins)
        matrix[uu,:] = event_aligned_spikes/bin_size

    timebin_da = xr.DataArray(matrix, dims=("unit_id", "timebin"), 
                            coords={
                                "unit_id": units[:]['unit_id'].values,
                                "timebin": timebins_table.index.values,
                                })

    return timebin_da, timebins_table


def get_structure_probe(session):
    
    units=pd.read_parquet(
                npc_lims.get_cache_path('units',session.id,version='v0.0.173')
            )

    unique_areas=units[:]['structure'].unique()

    structure_probe=np.full(len(units[:]),'',dtype=object)

    for aa in unique_areas:
        unique_probes=units[:].query('structure==@aa')['group_name'].unique()

        if len(unique_probes)>1:
            for up in unique_probes:
                unit_idx=units[:].query('structure==@aa and group_name==@up').index.values
                structure_probe[unit_idx]=aa+'_'+up
        elif len(unique_probes)==1:
            unit_idx=units[:].query('structure==@aa').index.values
            structure_probe[unit_idx]=aa
        else:
            print('no units in '+aa)

    structure_probe=pd.DataFrame({
        'structure_probe':structure_probe,
        'unit_id':units[:]['unit_id']},index=units[:].index.values)

    return structure_probe
