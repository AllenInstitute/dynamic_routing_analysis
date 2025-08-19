import os

import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import fdrcorrection

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
    
    #units: dataframe of units to include in tensor - must contain spike_times, unit_id columns
    #trials: dataframe of trials to include in tensor - must contain event_name column
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
    #makes a dataframe of timebins across a session along with relevant events
    #trials: dataframe of trials to create timebins table
    #bin_size: size of each bin in seconds
    #returns: timebins dataframe, bins

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
        'response':np.zeros(len(bin_centers),dtype=bool),
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
        'trial_index':np.full(len(bin_centers),np.nan),
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
        
    #set response
    response_trials=trials[:].query('is_response')

    for rr,response_trial in response_trials.iterrows():
        timebins_table['response'][(timebins_table['bin_start']>=response_trial['response_time']) & 
                                 (timebins_table['bin_start']<response_trial['response_time']+bin_size)]=True
        
        
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
        
        timebins_table['trial_index'][(timebins_table['bin_start']>=trial['start_time']) &
                                    (timebins_table['bin_start']<trial['stop_time'])]=tt


    return pd.DataFrame.from_dict(timebins_table),bins



def make_neuron_timebins_matrix(units, trials, bin_size, generate_context_labels=False):
    
    #units: units table to include in matrix
    #trials: trials table to create timebins table
    #bin_size: size of each bin in seconds
    #generate_context_labels: whether to generate context labels for each trial (i.e. if no context label present)
    
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
        trials['rewarded_modality']=context
        context_switches=np.where(np.diff(trials['rewarded_modality'].values=='vis'))[0]+1
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


def get_structure_probe(units):

    #appends probe to structure name if multiple probes in same structure i.e. "MOs_probeA"
    #returns new dataframe with structure_probe and unit_id columns 

    unique_areas=units[:]['structure'].unique()

    structure_probe=pd.DataFrame({
        'structure_probe':np.full(len(units[:]),'',dtype=object),
        'unit_id':units[:]['unit_id']
    })

    for aa in unique_areas:
        unique_probes=units[:].query('structure==@aa')['electrode_group_name'].unique()

        if len(unique_probes)>1:
            for up in unique_probes:
                unit_ids=units[:].query('structure==@aa and electrode_group_name==@up')['unit_id'].values
                structure_probe.loc[structure_probe['unit_id'].isin(unit_ids),'structure_probe']=aa+'_'+up
        elif len(unique_probes)==1:
            unit_ids=units[:].query('structure==@aa')['unit_id'].values
            structure_probe.loc[structure_probe['unit_id'].isin(unit_ids),'structure_probe']=aa
        else:
            print('no units in '+aa)

    return structure_probe

def compute_lick_modulation(trials, units, session_info, save_path=None, test=True):

    #computes lick modulation index, zscore, p-value, sign, and ROC AUC for each unit
    #saves or returns dataframe with lick modulation metrics and unit_id, session_id, and project
    #trials: dataframe of trials
    #units: dataframe of units
    #session_info: session_info object i.e. from npc_lims.get_session_info()
    #save_path: path to save lick modulation dataframe - if None, returns dataframe

    if test:
        units=units.head(10)

    lick_modulation={
        'unit_id':units['unit_id'].values.tolist(),
        'session_id':[str(session_info.id),]*len(units),
        'project':[str(session_info.project),]*len(units),
    }

    #make data array first
    time_before = 0.5
    time_after = 0.6
    binsize = 0.1
    trial_da = make_neuron_time_trials_tensor(units, trials, time_before, time_after, binsize)
                                                                              
    if "Templeton" in session_info.project:
        lick_trials = trials.query('is_response==True')
        non_lick_trials = trials.query('is_response==False')
        baseline_trials = trials

    elif "DynamicRouting" in session_info.project:
        lick_trials = trials.query('(stim_name=="vis1" and rewarded_modality=="aud" and is_response==True) or \
                                (stim_name=="sound1" and rewarded_modality=="vis" and is_response==True)')
        non_lick_trials = trials.query('(stim_name=="vis1" and rewarded_modality=="aud" and is_response==False) or \
                                        (stim_name=="sound1" and rewarded_modality=="vis" and is_response==False)')
        baseline_trials = trials.query('(stim_name=="vis1" and rewarded_modality=="aud") or \
                                        (stim_name=="sound1" and rewarded_modality=="vis")')
    else:
        print('incompatible project: ',session_info.project,'; skipping')
        return


    #lick modulation
    lick_frs_by_trial = trial_da.sel(time=slice(0.2,0.5),trials=lick_trials.index).mean(dim='time',skipna=True)
    non_lick_frs_by_trial = trial_da.sel(time=slice(0.2,0.5),trials=non_lick_trials.index).mean(dim='time',skipna=True)
    
    baseline_frs_by_trial = trial_da.sel(time=slice(-0.5,-0.2),trials=baseline_trials.index).mean(dim='time',skipna=True)

    lick_frs_by_trial_zscore = (lick_frs_by_trial.mean(dim='trials',skipna=True)-non_lick_frs_by_trial.mean(dim='trials',skipna=True)
                                )/(baseline_frs_by_trial.std(dim='trials',skipna=True))
    
    lick_frs_by_trial_zscore[np.isinf(lick_frs_by_trial_zscore)]=np.nan
    lick_modulation['lick_modulation_zscore'] = lick_frs_by_trial_zscore.values.tolist()

    lick_modulation_index=(lick_frs_by_trial.mean(dim='trials',skipna=True)-non_lick_frs_by_trial.mean(dim='trials',skipna=True)
                           )/(lick_frs_by_trial.mean(dim='trials',skipna=True)+non_lick_frs_by_trial.mean(dim='trials',skipna=True))
    lick_modulation['lick_modulation_index'] = lick_modulation_index
    
    pval = st.mannwhitneyu(lick_frs_by_trial.values.T, non_lick_frs_by_trial.values.T, nan_policy='omit')[1]
    lick_modulation['lick_modulation_p_value'] = pval

    stim_mod_sign=np.sign(lick_frs_by_trial.mean(dim='trials',skipna=True).values-non_lick_frs_by_trial.mean(dim='trials',skipna=True).values)
    lick_modulation['lick_modulation_sign'] = stim_mod_sign

    #ROC AUC - needs to loop through each unit
    lick_roc_auc=[]
    for uu,unit in units.iterrows():
        unit_lick_frs_by_trial = lick_frs_by_trial.sel(unit_id=unit['unit_id'])
        unit_non_lick_frs_by_trial = non_lick_frs_by_trial.sel(unit_id=unit['unit_id'])
        binary_label = np.concatenate([np.ones(unit_lick_frs_by_trial.size),np.zeros(unit_non_lick_frs_by_trial.size)])
        binary_score = np.concatenate([unit_lick_frs_by_trial.values,unit_non_lick_frs_by_trial.values])
        lick_roc_auc.append(roc_auc_score(binary_label, binary_score))

    lick_modulation['lick_modulation_roc_auc'] = lick_roc_auc

    lick_modulation = pd.DataFrame(lick_modulation)

    if save_path==None:
        return lick_modulation
    else:
        lick_modulation.to_parquet(os.path.join(save_path,session_info.id+'_lick_modulation.parquet'))


def compute_stim_context_modulation(trials, units, session_info, save_path=None, test=False, exclude_instruction_trials=True):

    #computes stimulus and context modulation indices, zscores, p-values, signs, and ROC AUCs for each unit
    #saves or returns new unit dataframe with modulation metrics and without spikes and waveforms
    #trials: dataframe of trials
    #units: dataframe of units
    #session_info: session_info object i.e. from npc_lims.get_session_info()
    #save_path: path to save stim_context_modulation dataframe - if None, returns unit dataframe

    if test:
        units=units.head(10)

    stim_context_modulation = {
        'unit_id':units['unit_id'].values.tolist(),
        'session_id':[str(session_info.id),]*len(units),
        'project':[str(session_info.project),]*len(units),
    }
  
    contexts=trials['rewarded_modality'].unique()

    #make fake context labels for Templeton (if not already present)
    if 'Templeton' in session_info.project:
        contexts = ['aud','vis']

        start_time=trials['start_time'].iloc[0]
        fake_context=np.full(len(trials), fill_value='nan')
        fake_block_nums=np.full(len(trials), fill_value=np.nan)

        if np.random.choice(contexts,1)=='vis':
            block_contexts=['vis','aud','vis','aud','vis','aud']
        else:
            block_contexts=['aud','vis','aud','vis','aud','vis']

        trials['true_block_index']=trials['block_index']
        trials['true_rewarded_modality']=trials['rewarded_modality']

        for block in range(0,6):
            block_start_time=start_time+block*10*60
            block_end_time=start_time+(block+1)*10*60
            block_trials=trials.query('start_time>=@block_start_time').index
            fake_context[block_trials]=block_contexts[block]
            fake_block_nums[block_trials]=block
        
        trials['rewarded_modality']=fake_context
        trials['block_index']=fake_block_nums
        trials['is_vis_context']=trials['rewarded_modality']=='vis'
        trials['is_aud_context']=trials['rewarded_modality']=='aud'

    #make data array first
    time_before = 0.1
    time_after = 0.31
    binsize = 0.025
    trial_da = make_neuron_time_trials_tensor(units, trials, time_before, time_after, binsize)

    #remove instruction trials after making data array to ensure correct indexing
    if exclude_instruction_trials==True:
        trials = trials.query('is_instruction==False')

    #get linear shifts
    #find middle 4 block labels
    first_block=trials.query('block_index==0').index.values
    middle_of_first=first_block[np.round(len(first_block)/2).astype('int')]

    last_block=trials.query('block_index==5').index.values
    middle_of_last=last_block[np.round(len(last_block)/2).astype('int')]

    middle_4_block_trials=trials.loc[middle_of_first:middle_of_last]
    middle_4_blocks=middle_4_block_trials.index.values

    #find the number of trials to shift by, from -1 to +1 block
    negative_shift=middle_4_blocks.min()
    positive_shift=trials.index.max()-middle_4_blocks.max()
    shifts=np.arange(-negative_shift,positive_shift+1)

    #find baseline frs across all trials
    baseline_frs = trial_da.sel(time=slice(-0.1,0)).mean(dim='time')

    vis_baseline_frs = baseline_frs.sel(trials=trials.query('rewarded_modality=="vis"').index)
    aud_baseline_frs = baseline_frs.sel(trials=trials.query('rewarded_modality=="aud"').index)

    pval = st.mannwhitneyu(vis_baseline_frs.values.T, aud_baseline_frs.values.T, nan_policy='omit')[1]
    stim_context_modulation['baseline_context_modulation_p_value'] = pval

    vis_baseline_mean_frs = vis_baseline_frs.mean(dim='trials',skipna=True)
    aud_baseline_mean_frs = aud_baseline_frs.mean(dim='trials',skipna=True)

    baseline_modulation_zscore=(vis_baseline_mean_frs-aud_baseline_mean_frs)/(baseline_frs.std(dim='trials',skipna=True))
    stim_context_modulation['baseline_context_modulation_zscore'] = baseline_modulation_zscore

    baseline_modulation_index=(vis_baseline_mean_frs-aud_baseline_mean_frs)/(vis_baseline_mean_frs+aud_baseline_mean_frs)
    stim_context_modulation['baseline_context_modulation_index'] = baseline_modulation_index

    baseline_mod_sign=np.sign(vis_baseline_mean_frs-aud_baseline_mean_frs)
    stim_context_modulation['baseline_context_modulation_sign'] = baseline_mod_sign

    stim_context_modulation['baseline_context_modulation_raw'] = (vis_baseline_mean_frs-aud_baseline_mean_frs)
    
    #TODO: re-enable linear shifted baseline context metrics
    # #linear shifted baseline context
    # temp_baseline_context_diff=[]
    # for sh,shift in enumerate(shifts):
    #     # labels = middle_4_block_trials['rewarded_modality'].values
    #     input_data = baseline_frs.sel(trials=middle_4_blocks+shift)
    #     temp_baseline_context_diff.append(input_data.sel(trials=middle_4_block_trials.query('rewarded_modality=="vis"').index).mean()-
    #                                       input_data.sel(trials=middle_4_block_trials.query('rewarded_modality=="aud"').index).mean())
    # temp_baseline_context_diff=np.asarray(temp_baseline_context_diff)
    
    # true_value = temp_baseline_context_diff[shifts==0][0]
    # null_median = np.median(temp_baseline_context_diff[shifts!=0])
    # null_mean = np.mean(temp_baseline_context_diff[shifts!=0])
    # null_std = np.std(temp_baseline_context_diff[shifts!=0])
    # pval_higher = np.mean(temp_baseline_context_diff[shifts!=0]>=true_value)
    # pval_lower = np.mean(temp_baseline_context_diff[shifts!=0]<=true_value)

    # stim_context_modulation['linear_shift_baseline_context_true_value'].append(true_value)
    # stim_context_modulation['linear_shift_baseline_context_null_median'].append(null_median)
    # stim_context_modulation['linear_shift_baseline_context_null_mean'].append(null_mean)
    # stim_context_modulation['linear_shift_baseline_context_null_std'].append(null_std)
    # stim_context_modulation['linear_shift_baseline_context_p_value_higher'].append(pval_higher)
    # stim_context_modulation['linear_shift_baseline_context_p_value_lower'].append(pval_lower)

    #loop through each unit to compute AUC metrics
    baseline_context_auc=[]
    vis_discrim_auc=[]
    aud_discrim_auc=[]
    target_discrim_auc=[]
    nontarget_discrim_auc=[]
    vis_vs_aud_auc=[]

    vis1_trials = trials.query('stim_name=="vis1"')
    vis2_trials = trials.query('stim_name=="vis2"')
    aud1_trials = trials.query('stim_name=="sound1"')
    aud2_trials = trials.query('stim_name=="sound2"')

    for uu,unit in units.iterrows():
        
        #baseline context auc
        binary_label=trials['rewarded_modality'].values=='vis'
        baseline_context_auc.append(roc_auc_score(binary_label,baseline_frs.sel(unit_id=unit['unit_id']).values))

        #cross stimulus discrimination
        #vis1 vs. vis2
        vis1_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=vis1_trials.index).mean(dim='time',skipna=True)
        vis2_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=vis2_trials.index).mean(dim='time',skipna=True)
        vis1_and_vis2_frs=np.concatenate([vis1_frs_by_trial.values,vis2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis1_frs_by_trial)),np.zeros(len(vis2_frs_by_trial))])
        vis_discrim_auc.append(roc_auc_score(binary_label,vis1_and_vis2_frs))

        #aud1 vs. aud2
        aud1_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=aud1_trials.index).mean(dim='time',skipna=True)
        aud2_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=aud2_trials.index).mean(dim='time',skipna=True)
        aud1_and_aud2_frs=np.concatenate([aud1_frs_by_trial.values,aud2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(aud1_frs_by_trial)),np.zeros(len(aud2_frs_by_trial))])
        aud_discrim_auc.append(roc_auc_score(binary_label,aud1_and_aud2_frs))

        #targets: vis1 vs sound1
        vis1_vs_aud1_frs=np.concatenate([vis1_frs_by_trial.values,aud1_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis1_frs_by_trial)),np.zeros(len(aud1_frs_by_trial))])
        target_discrim_auc.append(roc_auc_score(binary_label,vis1_vs_aud1_frs))

        #nontargets: vis2 vs sound2
        vis2_vs_aud2_frs=np.concatenate([vis2_frs_by_trial.values,aud2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis2_frs_by_trial)),np.zeros(len(aud2_frs_by_trial))])
        nontarget_discrim_auc.append(roc_auc_score(binary_label,vis2_vs_aud2_frs))

        #vis vs. aud
        vis_and_aud_frs=np.concatenate([vis1_frs_by_trial.values,vis2_frs_by_trial.values,
                                        aud1_frs_by_trial.values,aud2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis1_frs_by_trial)+len(vis2_frs_by_trial)),
                                    np.zeros(len(aud1_frs_by_trial)+len(aud2_frs_by_trial))])
        vis_vs_aud_auc.append(roc_auc_score(binary_label,vis_and_aud_frs))
        

    stim_context_modulation['baseline_context_roc_auc'] = baseline_context_auc
    stim_context_modulation['vis_discrim_roc_auc'] = vis_discrim_auc
    stim_context_modulation['aud_discrim_roc_auc'] = aud_discrim_auc
    stim_context_modulation['target_discrim_roc_auc'] = target_discrim_auc
    stim_context_modulation['nontarget_discrim_roc_auc'] = nontarget_discrim_auc
    stim_context_modulation['vis_vs_aud'] = vis_vs_aud_auc

    #HIT/CR/FA - currently only makes sense for DR experiments
    behav_time_windows_start=[0,0.1,0.2]
    behav_time_windows_end=[0.1,0.2,0.3]
    behav_time_window_labels=['early','mid','late']

    if 'DynamicRouting' in session_info.project:
        cr_trials=trials.query('is_response==False and is_correct==True and is_target==True')
        fa_trials=trials.query('is_response==True and is_correct==False and is_target==True')
        hit_trials=trials.query('is_response==True and is_correct==True and is_target==True')
  
        for tw,time_window in enumerate(behav_time_window_labels):
            cr_vs_fa_auc = []
            hit_vs_cr_auc = []
            hit_vs_fa_auc = []

            for uu, unit in units.iterrows():
                cr_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(behav_time_windows_start[tw],behav_time_windows_end[tw]),trials=cr_trials.index).mean(dim='time',skipna=True)
                fa_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(behav_time_windows_start[tw],behav_time_windows_end[tw]),trials=fa_trials.index).mean(dim='time',skipna=True)
                hit_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(behav_time_windows_start[tw],behav_time_windows_end[tw]),trials=hit_trials.index).mean(dim='time',skipna=True)

                #cr vs. fa
                cr_and_fa_frs=np.concatenate([cr_frs_by_trial.values,fa_frs_by_trial.values])
                binary_label=np.concatenate([np.ones(len(cr_frs_by_trial)),np.zeros(len(fa_frs_by_trial))])
                cr_vs_fa_auc.append(roc_auc_score(binary_label,cr_and_fa_frs))
                
                #hit vs. cr
                hit_and_cr_frs=np.concatenate([hit_frs_by_trial.values,cr_frs_by_trial.values])
                binary_label=np.concatenate([np.ones(len(hit_frs_by_trial)),np.zeros(len(cr_frs_by_trial))])
                hit_vs_cr_auc.append(roc_auc_score(binary_label,hit_and_cr_frs))

                #hit vs. fa
                hit_and_fa_frs=np.concatenate([hit_frs_by_trial.values,fa_frs_by_trial.values])
                binary_label=np.concatenate([np.ones(len(hit_frs_by_trial)),np.zeros(len(fa_frs_by_trial))])
                hit_vs_fa_auc.append(roc_auc_score(binary_label,hit_and_fa_frs))

            stim_context_modulation['cr_vs_fa_'+time_window+'_roc_auc'] = cr_vs_fa_auc
            stim_context_modulation['hit_vs_cr_'+time_window+'_roc_auc'] = hit_vs_cr_auc
            stim_context_modulation['hit_vs_fa_'+time_window+'_roc_auc'] = hit_vs_fa_auc

    else:
        for tw,time_window in enumerate(behav_time_window_labels):
            stim_context_modulation['cr_vs_fa_'+time_window+'_roc_auc'] = np.nan
            stim_context_modulation['hit_vs_cr_'+time_window+'_roc_auc'] = np.nan
            stim_context_modulation['hit_vs_fa_'+time_window+'_roc_auc'] = np.nan

    #loop through stimuli
    for ss in trials['stim_name'].unique():
        if ss=='catch':
            same_context=contexts[0]
            other_context=contexts[1]
        elif 'sound' in ss:
            same_context='aud'
            other_context='vis'
        elif 'vis' in ss:
            same_context='vis'
            other_context='aud'

        #stimulus modulation
        stim_trials = trials.query('stim_name==@ss')
        stim_frs_by_trial = trial_da.sel(time=slice(0,0.1),trials=stim_trials.index).mean(dim='time',skipna=True)
        stim_baseline_frs_by_trial = baseline_frs.sel(trials=stim_trials.index)
        stim_frs_by_trial_zscore = (stim_frs_by_trial.mean(dim='trials',skipna=True)-stim_baseline_frs_by_trial.mean(dim='trials',skipna=True)
                                    )/(stim_baseline_frs_by_trial.std(dim='trials',skipna=True))
        stim_context_modulation[ss+'_stimulus_modulation_zscore']=stim_frs_by_trial_zscore
        stimulus_modulation_index=(stim_frs_by_trial.mean(dim='trials',skipna=True)-stim_baseline_frs_by_trial.mean(dim='trials',skipna=True)
                                   )/(stim_frs_by_trial.mean(dim='trials',skipna=True)+stim_baseline_frs_by_trial.mean(dim='trials',skipna=True))
        stim_context_modulation[ss+'_stimulus_modulation_index']=stimulus_modulation_index

        pval = st.wilcoxon(stim_frs_by_trial.values.T, stim_baseline_frs_by_trial.values.T, nan_policy='omit',zero_method='zsplit')[1]
        stim_context_modulation[ss+'_stimulus_modulation_p_value']=pval
        stim_mod_sign=np.sign(stim_frs_by_trial.mean(dim='trials',skipna=True)-stim_baseline_frs_by_trial.mean(dim='trials',skipna=True))
        stim_context_modulation[ss+'_stimulus_modulation_sign']=stim_mod_sign

        #stimulus late modulation
        stim_late_frs_by_trial = trial_da.sel(time=slice(0.1,0.2),trials=stim_trials.index).mean(dim='time',skipna=True)
        stim_late_frs_by_trial_zscore = (stim_late_frs_by_trial.mean(dim='trials',skipna=True)-stim_baseline_frs_by_trial.mean(dim='trials',skipna=True)
                                        )/(stim_baseline_frs_by_trial.std(dim='trials',skipna=True))
        stim_context_modulation[ss+'_stimulus_late_modulation_zscore'] = stim_late_frs_by_trial_zscore
        stimulus_late_modulation_index=(stim_late_frs_by_trial.mean(dim='trials',skipna=True)-stim_baseline_frs_by_trial.mean(dim='trials',skipna=True)
                                       )/(stim_late_frs_by_trial.mean(dim='trials',skipna=True)+stim_baseline_frs_by_trial.mean(dim='trials',skipna=True))
        stim_context_modulation[ss+'_stimulus_late_modulation_index'] = stimulus_late_modulation_index

        pval = st.wilcoxon(stim_late_frs_by_trial.values.T, stim_baseline_frs_by_trial.values.T, nan_policy='omit',zero_method='zsplit')[1]
        stim_context_modulation[ss+'_stimulus_late_modulation_p_value'] = pval
        stim_late_mod_sign=np.sign(stim_late_frs_by_trial.mean(dim='trials',skipna=True)-stim_baseline_frs_by_trial.mean(dim='trials',skipna=True))
        stim_context_modulation[ss+'_stimulus_late_modulation_sign'] = stim_late_mod_sign

        #latency
        stim_latency = np.abs(trial_da).sel(time=slice(0,0.3),trials=stim_trials.index).mean(dim='trials',skipna=True).idxmax(dim='time').values
        stim_context_modulation[ss+'_stimulus_latency'] = np.nanmean(stim_latency)

        #find zscore of max fr per unit
        stim_max_frs = trial_da.sel(time=slice(0,0.3),trials=stim_trials.index).max(dim='time',skipna=True).mean(dim='trials',skipna=True)
        stim_max_frs_zscore = (stim_max_frs - stim_baseline_frs_by_trial.mean(dim='trials',skipna=True)) / stim_baseline_frs_by_trial.std(dim='trials',skipna=True)
        stim_context_modulation[ss+'_stimulus_max_fr_zscore'] = stim_max_frs_zscore

        #find stim trials in same vs. other context
        same_context_trials = trials.query('rewarded_modality==@same_context and stim_name==@ss')
        other_context_trials = trials.query('rewarded_modality==@other_context and stim_name==@ss')

        same_context_baseline_frs = baseline_frs.sel(trials=same_context_trials.index)
        other_context_baseline_frs = baseline_frs.sel(trials=other_context_trials.index)

        #find raw frs during stim (first 100ms)
        same_context_frs_by_trial = trial_da.sel(time=slice(0,0.1),trials=same_context_trials.index).mean(dim='time',skipna=True)
        other_context_frs_by_trial = trial_da.sel(time=slice(0,0.1),trials=other_context_trials.index).mean(dim='time',skipna=True)

        pval = st.mannwhitneyu(same_context_frs_by_trial.values.T, other_context_frs_by_trial.values.T, nan_policy='omit')[1]
        stim_context_modulation[ss+'_context_modulation_p_value'] = pval

        same_context_frs = same_context_frs_by_trial.mean(dim='trials',skipna=True).values
        other_context_frs = other_context_frs_by_trial.mean(dim='trials',skipna=True).values

        context_modulation_zscore=(same_context_frs-other_context_frs)/(stim_baseline_frs_by_trial.std(dim='trials',skipna=True).values)
        stim_context_modulation[ss+'_context_modulation_zscore'] = context_modulation_zscore

        # stim context modulation sign
        context_mod_sign=np.sign(np.nanmean(same_context_frs-other_context_frs))
        stim_context_modulation[ss+'_context_modulation_sign'] = context_mod_sign

        stim_context_modulation[ss+'_context_modulation_raw'] = (same_context_frs - other_context_frs)

        #find evoked frs during stim (first 100ms)
        same_context_evoked_frs_by_trial = (trial_da.sel(time=slice(0,0.1),trials=same_context_trials.index).mean(dim=['time'],skipna=True)-same_context_baseline_frs)
        other_context_evoked_frs_by_trial = (trial_da.sel(time=slice(0,0.1),trials=other_context_trials.index).mean(dim=['time'],skipna=True)-other_context_baseline_frs)

        pval = st.mannwhitneyu(same_context_evoked_frs_by_trial.values.T, other_context_evoked_frs_by_trial.values.T, nan_policy='omit')[1]
        stim_context_modulation[ss+'_evoked_context_modulation_p_value'] = pval

        same_context_evoked_frs = same_context_evoked_frs_by_trial.mean(dim='trials',skipna=True).values
        other_context_evoked_frs = other_context_evoked_frs_by_trial.mean(dim='trials',skipna=True).values

        context_modulation_evoked_zscore=(same_context_evoked_frs-other_context_evoked_frs)/(stim_baseline_frs_by_trial.std(dim='trials',skipna=True).values)
        stim_context_modulation[ss+'_evoked_context_modulation_zscore'] = context_modulation_evoked_zscore

        # evoked stim context modulation sign
        context_mod_evoked_sign=np.sign(same_context_evoked_frs-other_context_evoked_frs)
        stim_context_modulation[ss+'_evoked_context_modulation_sign'] = context_mod_evoked_sign

        stim_context_modulation[ss+'_evoked_context_modulation_raw'] = (same_context_evoked_frs - other_context_evoked_frs)
        
        # #negative numbers can make index behave weirdly, so subtract the minimum from both
        # if same_context_evoked_frs<0 or other_context_evoked_frs<0:
        #     same_context_evoked_frs = same_context_evoked_frs - np.min([same_context_evoked_frs,other_context_evoked_frs])
        #     other_context_evoked_frs = other_context_evoked_frs - np.min([same_context_evoked_frs,other_context_evoked_frs])

        #compute metrics
        raw_fr_metric=(same_context_frs-other_context_frs)/(same_context_frs+other_context_frs)
        stim_context_modulation[ss+'_context_modulation_index'] = raw_fr_metric

        evoked_fr_metric=(same_context_evoked_frs-other_context_evoked_frs)/(same_context_evoked_frs+other_context_evoked_frs)
        stim_context_modulation[ss+'_evoked_context_modulation_index'] = evoked_fr_metric

        #late stimulus context modulation
        same_context_late_frs_by_trial = trial_da.sel(time=slice(0.1,0.2),trials=same_context_trials.index).mean(dim='time',skipna=True)
        other_context_late_frs_by_trial = trial_da.sel(time=slice(0.1,0.2),trials=other_context_trials.index).mean(dim='time',skipna=True)
        pval = st.mannwhitneyu(same_context_late_frs_by_trial.values.T, other_context_late_frs_by_trial.values.T, nan_policy='omit')[1]
        stim_context_modulation[ss+'_stimulus_late_context_modulation_p_value'] = pval
        same_context_late_frs = same_context_late_frs_by_trial.mean(dim='trials',skipna=True).values
        other_context_late_frs = other_context_late_frs_by_trial.mean(dim='trials',skipna=True).values
        stim_late_context_modulation_zscore=(same_context_late_frs-other_context_late_frs)/(stim_baseline_frs_by_trial.std(dim='trials',skipna=True).values)
        stim_context_modulation[ss+'_stimulus_late_context_modulation_zscore'] = stim_late_context_modulation_zscore

        # stim late context modulation sign
        context_mod_late_sign=np.sign(same_context_late_frs-other_context_late_frs)
        stim_context_modulation[ss+'_stimulus_late_context_modulation_sign'] = context_mod_late_sign
        stim_context_modulation[ss+'_stimulus_late_context_modulation_raw'] = (same_context_late_frs - other_context_late_frs)
        stim_late_context_modulation_metric=(same_context_late_frs-other_context_late_frs)/(same_context_late_frs+other_context_late_frs)
        stim_context_modulation[ss+'_stimulus_late_context_modulation_index'] = stim_late_context_modulation_metric

        # evoked late stimulus context modulation
        same_context_late_evoked_frs_by_trial = (trial_da.sel(time=slice(0.1,0.2),trials=same_context_trials.index).mean(dim=['time'],skipna=True)-same_context_baseline_frs)
        other_context_late_evoked_frs_by_trial = (trial_da.sel(time=slice(0.1,0.2),trials=other_context_trials.index).mean(dim=['time'],skipna=True)-other_context_baseline_frs)
        pval = st.mannwhitneyu(same_context_late_evoked_frs_by_trial.values.T, other_context_late_evoked_frs_by_trial.values.T, nan_policy='omit')[1]
        stim_context_modulation[ss+'_evoked_stimulus_late_context_modulation_p_value'] = pval
        same_context_late_evoked_frs = same_context_late_evoked_frs_by_trial.mean(dim='trials',skipna=True).values
        other_context_late_evoked_frs = other_context_late_evoked_frs_by_trial.mean(dim='trials',skipna=True).values
        context_modulation_evoked_late_zscore=(same_context_late_evoked_frs-other_context_late_evoked_frs)/(stim_baseline_frs_by_trial.std(dim='trials',skipna=True).values)
        stim_context_modulation[ss+'_evoked_stimulus_late_context_modulation_zscore'] = context_modulation_evoked_late_zscore

        # evoked late context modulation sign
        context_modulation_evoked_late_sign=np.sign(same_context_late_evoked_frs-other_context_late_evoked_frs)
        stim_context_modulation[ss+'_evoked_stimulus_late_context_modulation_sign'] = context_modulation_evoked_late_sign
        stim_context_modulation[ss+'_evoked_stimulus_late_context_modulation_raw'] = (same_context_late_evoked_frs - other_context_late_evoked_frs)
        stim_late_context_modulation_evoked_metric=(same_context_late_evoked_frs-other_context_late_evoked_frs)/(same_context_late_evoked_frs+other_context_late_evoked_frs)
        stim_context_modulation[ss+'_evoked_stimulus_late_context_modulation_index'] = stim_late_context_modulation_evoked_metric


        #AUC calculation
        stim_auc=[]
        stim_late_auc=[]
        stim_context_auc=[]
        evoked_stim_context_auc=[]
        stim_late_context_auc=[]
        stim_late_evoked_context_auc=[]
        
        for uu,unit in units.iterrows():
            unit_stim_frs_by_trial = stim_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_stim_late_frs_by_trial = stim_late_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_stim_baseline_frs_by_trial = stim_baseline_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_same_context_frs_by_trial = same_context_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_other_context_frs_by_trial = other_context_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_same_context_evoked_frs_by_trial = same_context_evoked_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_other_context_evoked_frs_by_trial = other_context_evoked_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_same_context_late_frs_by_trial = stim_late_same_context_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_other_context_late_frs_by_trial = stim_late_other_context_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_same_context_late_evoked_frs_by_trial = stim_late_same_context_evoked_frs_by_trial.sel(unit_id=unit['unit_id'])
            unit_other_context_late_evoked_frs_by_trial = stim_late_other_context_evoked_frs_by_trial.sel(unit_id=unit['unit_id'])

            #auc for stimulus frs
            stim_and_baseline_frs=np.concatenate([unit_stim_frs_by_trial.values,unit_stim_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(unit_stim_frs_by_trial)),np.zeros(len(unit_stim_baseline_frs_by_trial))])
            stim_auc.append(roc_auc_score(binary_label,stim_and_baseline_frs))
            
            #auc for stimulus late frs
            stim_late_and_baseline_frs=np.concatenate([unit_stim_late_frs_by_trial.values,unit_stim_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(unit_stim_late_frs_by_trial)),np.zeros(len(unit_stim_baseline_frs_by_trial))])
            stim_late_auc.append(roc_auc_score(binary_label,stim_late_and_baseline_frs))
            
            # stim context modulation auc
            binary_label=np.concatenate([np.ones(len(unit_same_context_frs_by_trial.values)),np.zeros(len(unit_other_context_frs_by_trial.values))])
            stim_context_auc.append(roc_auc_score(binary_label,np.concatenate([unit_same_context_frs_by_trial.values,unit_other_context_frs_by_trial.values])))

            # evoked stim context modulation auc
            binary_label=np.concatenate([np.ones(len(unit_same_context_evoked_frs_by_trial.values)),np.zeros(len(unit_other_context_evoked_frs_by_trial.values))])
            evoked_stim_context_auc.append(roc_auc_score(binary_label,np.concatenate([unit_same_context_evoked_frs_by_trial.values,unit_other_context_evoked_frs_by_trial.values])))

            #stimulus late context modulation auc
            binary_label=np.concatenate([np.ones(len(unit_same_context_late_frs_by_trial.values)),np.zeros(len(unit_other_context_late_frs_by_trial.values))])
            stim_late_context_auc.append(roc_auc_score(binary_label,np.concatenate([unit_same_context_late_frs_by_trial.values,unit_other_context_late_frs_by_trial.values])))

            # evoked late stimulus context modulation auc
            binary_label=np.concatenate([np.ones(len(unit_same_context_late_evoked_frs_by_trial.values)),np.zeros(len(unit_other_context_late_evoked_frs_by_trial.values))])
            stim_late_evoked_context_auc.append(roc_auc_score(binary_label,np.concatenate([unit_same_context_late_evoked_frs_by_trial.values,unit_other_context_late_evoked_frs_by_trial.values])))


        stim_context_modulation[ss+'_stimulus_modulation_roc_auc']=stim_auc
        stim_context_modulation[ss+'_stimulus_late_modulation_roc_auc']=stim_late_auc
        stim_context_modulation[ss+'_context_modulation_roc_auc']=stim_context_auc
        stim_context_modulation[ss+'_evoked_context_modulation_roc_auc']=evoked_stim_context_auc
        stim_context_modulation[ss+'_stimulus_late_context_modulation_roc_auc']=stim_late_context_auc
        stim_context_modulation[ss+'_evoked_stimulus_late_context_modulation_roc_auc']=stim_late_evoked_context_auc

    stim_context_modulation = pd.DataFrame(stim_context_modulation)

    unit_metric_merge=units.drop(columns=['spike_times','waveform_sd','waveform_mean','obs_intervals'], errors='ignore'
                                 ).merge(stim_context_modulation,on=['unit_id']).sort_values(by='unit_id').reset_index()

    if save_path==None:
        return unit_metric_merge
    else:
        unit_metric_merge.to_parquet(os.path.join(save_path,session_info.id+'_stim_context_modulation.parquet'))


#calculate metrics for channel alignment
def compute_metrics_for_alignment(trials, units, session_info, save_path):

    #computes metrics for alignment of units to annotations
    #saves metrics dataframe for each probe

    alignment_metrics = {
        'unit_id':[],
        'session_id':[],
        'experiment_day':[],
        'project':[],
        'probe':[],
        'peak_channel':[],
        'lick_modulation_roc_auc':[],
        'vis_discrim_roc_auc':[],
        'aud_discrim_roc_auc':[],
        'any_vis_roc_auc':[],
        'any_aud_roc_auc':[],
        'firing_rate':[],
        'peak_to_valley':[],
        'peak_trough_ratio':[],
        'repolarization_slope':[],
        'recovery_slope':[],
        'spread':[],
        'velocity_above':[],
        'velocity_below':[],
        'snr':[],

        'amplitude_cutoff':[], 
        'amplitude_cv_median':[], 
        'amplitude_cv_range':[],
        'amplitude_median':[], 
        'drift_ptp':[], 
        'drift_std':[], 
        'drift_mad':[],
        'firing_range':[], 
        'isi_violations_ratio':[],
        'isi_violations_count':[], 
        'presence_ratio':[],
        'rp_contamination':[], 
        'rp_violations':[], 
        'sliding_rp_violation':[],
        'sync_spike_2':[], 
        'sync_spike_4':[], 
        'sync_spike_8':[], 
        'd_prime':[],
        'isolation_distance':[], 
        'l_ratio':[], 
        'silhouette':[], 
        'nn_hit_rate':[],
        'nn_miss_rate':[], 
        'exp_decay':[], 
        'half_width':[], 
        'num_negative_peaks':[],
        'num_positive_peaks':[],
    }

    if trials is not None:
        contexts=trials['rewarded_modality'].unique()

        if 'Templeton' in session_info.project:
            contexts = ['aud','vis']

            start_time=trials['start_time'].iloc[0]
            fake_context=np.full(len(trials), fill_value='nan')
            fake_block_nums=np.full(len(trials), fill_value=np.nan)

            if np.random.choice(contexts,1)=='vis':
                block_contexts=['vis','aud','vis','aud','vis','aud']
            else:
                block_contexts=['aud','vis','aud','vis','aud','vis']

            trials['true_block_index']=trials['block_index']
            trials['true_rewarded_modality']=trials['rewarded_modality']

            for block in range(0,6):
                block_start_time=start_time+block*10*60
                block_end_time=start_time+(block+1)*10*60
                block_trials=trials.query('start_time>=@block_start_time').index
                fake_context[block_trials]=block_contexts[block]
                fake_block_nums[block_trials]=block
            
            trials['rewarded_modality']=fake_context
            trials['block_index']=fake_block_nums
            trials['is_vis_context']=trials['rewarded_modality']=='vis'
            trials['is_aud_context']=trials['rewarded_modality']=='aud'

        #make data array first
        time_before = 0.5
        time_after = 0.5
        binsize = 0.025
        trial_da = make_neuron_time_trials_tensor(units, trials, time_before, time_after, binsize)

    #for each unit
    for uu,unit in units.iterrows():
        
        alignment_metrics['unit_id'].append(unit['unit_id'])
        alignment_metrics['session_id'].append(str(session_info.id))
        alignment_metrics['project'].append(str(session_info.project))
        alignment_metrics['experiment_day'].append(str(session_info.experiment_day))
        alignment_metrics['probe'].append(str(unit['electrode_group_name']))
        alignment_metrics['peak_channel'].append(unit['peak_channel'])

        alignment_metrics['firing_rate'].append(unit['firing_rate'])
        alignment_metrics['peak_to_valley'].append(unit['peak_to_valley'])
        alignment_metrics['peak_trough_ratio'].append(unit['peak_trough_ratio'])
        alignment_metrics['repolarization_slope'].append(unit['repolarization_slope'])
        alignment_metrics['recovery_slope'].append(unit['recovery_slope'])

        alignment_metrics['spread'].append(unit['spread'])
        alignment_metrics['velocity_above'].append(unit['velocity_above'])
        alignment_metrics['velocity_below'].append(unit['velocity_below'])
        alignment_metrics['snr'].append(unit['spread'])

        alignment_metrics['amplitude_cutoff'].append(unit['amplitude_cutoff'])
        alignment_metrics['amplitude_cv_median'].append(unit['amplitude_cv_median'])
        alignment_metrics['amplitude_cv_range'].append(unit['amplitude_cv_range'])
        alignment_metrics['amplitude_median'].append(unit['amplitude_median'])
        alignment_metrics['drift_ptp'].append(unit['drift_ptp'])
        alignment_metrics['drift_std'].append(unit['drift_std'])
        alignment_metrics['drift_mad'].append(unit['drift_mad'])
        alignment_metrics['firing_range'].append(unit['firing_range'])
        alignment_metrics['isi_violations_ratio'].append(unit['isi_violations_ratio'])
        alignment_metrics['isi_violations_count'].append(unit['isi_violations_count'])
        alignment_metrics['presence_ratio'].append(unit['presence_ratio'])
        alignment_metrics['rp_contamination'].append(unit['rp_contamination'])
        alignment_metrics['rp_violations'].append(unit['rp_violations'])
        alignment_metrics['sliding_rp_violation'].append(unit['sliding_rp_violation'])
        alignment_metrics['sync_spike_2'].append(unit['sync_spike_2'])
        alignment_metrics['sync_spike_4'].append(unit['sync_spike_4'])
        alignment_metrics['sync_spike_8'].append(unit['sync_spike_8'])
        alignment_metrics['d_prime'].append(unit['d_prime'])
        alignment_metrics['isolation_distance'].append(unit['isolation_distance'])
        alignment_metrics['l_ratio'].append(unit['l_ratio'])
        alignment_metrics['silhouette'].append(unit['silhouette'])
        alignment_metrics['nn_hit_rate'].append(unit['nn_hit_rate'])
        alignment_metrics['nn_miss_rate'].append(unit['nn_miss_rate'])
        alignment_metrics['exp_decay'].append(unit['exp_decay'])
        alignment_metrics['half_width'].append(unit['half_width'])
        alignment_metrics['num_negative_peaks'].append(unit['num_negative_peaks'])
        alignment_metrics['num_positive_peaks'].append(unit['num_positive_peaks'])
        
        #if surface channel, don't try to calculate metrics
        if unit['peak_channel']>383:
            #append nans for all metrics
            alignment_metrics['any_vis_roc_auc'].append(np.nan)
            alignment_metrics['any_aud_roc_auc'].append(np.nan)
            alignment_metrics['vis_discrim_roc_auc'].append(np.nan)
            alignment_metrics['aud_discrim_roc_auc'].append(np.nan)
            alignment_metrics['lick_modulation_roc_auc'].append(np.nan)
            continue
        
        if trials is not None:
            #find baseline frs across all trials
            baseline_frs = trial_da.sel(unit_id=unit['unit_id'],time=slice(-0.1,0)).mean(dim='time')

            all_stim_frs_by_trial = {}
            #loop through stimuli
            for ss in trials['stim_name'].unique():

                #stimulus modulation
                if "Templeton" in session_info.project:
                    stim_trials = trials.query('stim_name==@ss')
                else:
                    stim_trials = trials.query('stim_name==@ss and is_response==False') #remove response trials to minimize contamination
                stim_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=stim_trials.index).mean(dim='time',skipna=True)

                all_stim_frs_by_trial[ss]=stim_frs_by_trial

            if "Templeton" in session_info.project:
                any_vis_trials = trials.query('stim_name.str.contains("vis")')
            else:
                any_vis_trials = trials.query('stim_name.str.contains("vis") and is_response==False')
            any_vis_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=any_vis_trials.index).mean(dim='time',skipna=True)
            any_vis_baseline_frs_by_trial = baseline_frs.sel(trials=any_vis_trials.index)
            any_vis_and_baseline_frs=np.concatenate([any_vis_frs_by_trial.values,any_vis_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(any_vis_frs_by_trial)),np.zeros(len(any_vis_baseline_frs_by_trial))])
            if len(np.unique(binary_label))>1:
                any_vis_context_auc=roc_auc_score(binary_label,any_vis_and_baseline_frs)
            else:
                any_vis_context_auc=np.nan
            alignment_metrics['any_vis_roc_auc'].append(any_vis_context_auc)

            if "Templeton" in session_info.project:
                any_aud_trials = trials.query('stim_name.str.contains("sound")')
            else:
                any_aud_trials = trials.query('stim_name.str.contains("sound") and is_response==False')
            any_aud_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=any_aud_trials.index).mean(dim='time',skipna=True)
            any_aud_baseline_frs_by_trial = baseline_frs.sel(trials=any_aud_trials.index)
            any_aud_and_baseline_frs=np.concatenate([any_aud_frs_by_trial.values,any_aud_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(any_aud_frs_by_trial)),np.zeros(len(any_aud_baseline_frs_by_trial))])
            if len(np.unique(binary_label))>1:
                any_aud_context_auc=roc_auc_score(binary_label,any_aud_and_baseline_frs)
            else:
                any_aud_context_auc=np.nan
            alignment_metrics['any_aud_roc_auc'].append(any_aud_context_auc)

            #same modality stimulus discrimination
            #vis1 vs. vis2
            vis1_and_vis2_frs=np.concatenate([all_stim_frs_by_trial['vis1'].values,all_stim_frs_by_trial['vis2'].values])
            binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis1'])),np.zeros(len(all_stim_frs_by_trial['vis2']))])
            if len(np.unique(binary_label))>1:
                vis_discrim_auc=roc_auc_score(binary_label,vis1_and_vis2_frs)
            else:
                vis_discrim_auc=np.nan
            alignment_metrics['vis_discrim_roc_auc'].append(vis_discrim_auc)

            #aud1 vs. aud2
            aud1_and_aud2_frs=np.concatenate([all_stim_frs_by_trial['sound1'].values,all_stim_frs_by_trial['sound2'].values])
            binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['sound1'])),np.zeros(len(all_stim_frs_by_trial['sound2']))])
            if len(np.unique(binary_label))>1:
                aud_discrim_auc=roc_auc_score(binary_label,aud1_and_aud2_frs)
            else:
                aud_discrim_auc=np.nan
            alignment_metrics['aud_discrim_roc_auc'].append(aud_discrim_auc)

            # #targets: vis1 vs sound1
            # vis1_vs_aud1_frs=np.concatenate([all_stim_frs_by_trial['vis1'].values,all_stim_frs_by_trial['sound1'].values])
            # binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis1'])),np.zeros(len(all_stim_frs_by_trial['sound1']))])
            # target_discrim_auc=roc_auc_score(binary_label,vis1_vs_aud1_frs)
            # alignment_metrics['target_discrim_roc_auc'].append(target_discrim_auc)

            # #nontargets: vis2 vs sound2
            # vis2_vs_aud2_frs=np.concatenate([all_stim_frs_by_trial['vis2'].values,all_stim_frs_by_trial['sound2'].values])
            # binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis2'])),np.zeros(len(all_stim_frs_by_trial['sound2']))])
            # nontarget_discrim_auc=roc_auc_score(binary_label,vis2_vs_aud2_frs)
            # alignment_metrics['nontarget_discrim_roc_auc'].append(nontarget_discrim_auc)

            # #vis vs. aud
            # vis_and_aud_frs=np.concatenate([all_stim_frs_by_trial['vis1'].values,all_stim_frs_by_trial['vis2'].values,
            #                                 all_stim_frs_by_trial['sound1'].values,all_stim_frs_by_trial['sound2'].values])
            # binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis1'])+len(all_stim_frs_by_trial['vis2'])),
            #                             np.zeros(len(all_stim_frs_by_trial['sound1'])+len(all_stim_frs_by_trial['sound2']))])
            # vis_vs_aud_auc=roc_auc_score(binary_label,vis_and_aud_frs)
            # alignment_metrics['vis_vs_aud_roc_auc'].append(vis_vs_aud_auc)

            #lick modulation
            if "DynamicRouting" in session_info.project:
                lick_trials = trials.query('(stim_name=="vis1" and rewarded_modality=="aud" and is_response==True) or \
                                        (stim_name=="sound1" and rewarded_modality=="vis" and is_response==True)')
                non_lick_trials = trials.query('(stim_name=="vis1" and rewarded_modality=="aud" and is_response==False) or \
                                                (stim_name=="sound1" and rewarded_modality=="vis" and is_response==False)')
            elif "Templeton" in session_info.project:
                lick_trials = trials.query('is_response==True')
                non_lick_trials = trials.query('is_response==False')

            lick_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.2,0.5),trials=lick_trials.index).mean(dim='time',skipna=True)
            non_lick_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.2,0.5),trials=non_lick_trials.index).mean(dim='time',skipna=True)

            #ROC AUC
            binary_label = np.concatenate([np.ones(lick_frs_by_trial.size),np.zeros(non_lick_frs_by_trial.size)])
            binary_score = np.concatenate([lick_frs_by_trial.values,non_lick_frs_by_trial.values])
            if len(np.unique(binary_label))>1:
                lick_roc_auc = roc_auc_score(binary_label, binary_score)
            else:
                lick_roc_auc = np.nan
            alignment_metrics['lick_modulation_roc_auc'].append(lick_roc_auc)
        
        else:
            alignment_metrics['any_vis_roc_auc'].append(np.nan)
            alignment_metrics['any_aud_roc_auc'].append(np.nan)
            alignment_metrics['vis_discrim_roc_auc'].append(np.nan)
            alignment_metrics['aud_discrim_roc_auc'].append(np.nan)
            alignment_metrics['lick_modulation_roc_auc'].append(np.nan)
    
    alignment_metrics = pd.DataFrame(alignment_metrics)
    alignment_metrics['visual_response'] = np.abs(alignment_metrics['any_vis_roc_auc'] - 0.5)*2
    alignment_metrics['auditory_response'] = np.abs(alignment_metrics['any_aud_roc_auc'] - 0.5)*2
    alignment_metrics['visual_discrim'] = np.abs(alignment_metrics['vis_discrim_roc_auc'] - 0.5)*2
    alignment_metrics['auditory_discrim'] = np.abs(alignment_metrics['aud_discrim_roc_auc'] - 0.5)*2
    alignment_metrics['lick_modulation'] = np.abs(alignment_metrics['lick_modulation_roc_auc'] - 0.5)*2


    alignment_metrics.drop(columns=['any_vis_roc_auc','any_aud_roc_auc','vis_discrim_roc_auc','aud_discrim_roc_auc','lick_modulation_roc_auc'],inplace=True)

    probes=alignment_metrics['probe'].unique()
    for probe in probes:
        probe_units=alignment_metrics.query('probe==@probe')
        probe_units.to_csv(os.path.join(save_path,session_info.id+'_day_'+str(session_info.experiment_day)+'_'+probe+'_stim_modulation.csv'),index=False)

def get_all_performance():
    from npc_sessions import DynamicRoutingSession

    ephys_sessions = ephys_sessions=tuple(s for s in npc_lims.get_session_info(is_ephys=True))
    performance_dict={}

    for session_info in ephys_sessions[:]:
        
        try:
            try:
                performance = pd.read_parquet(
                                    npc_lims.get_cache_path('performance',session_info.id,version='any')
                                )
            except:
                session=DynamicRoutingSession(session_info.id)
                performance = session.performance[:]
        except:
            print(session_info.id,'failed to load performance, skipping session')
            continue

        performance_dict[session_info.id]=performance

    return performance_dict


def concat_single_unit_metrics_across_sessions(stim_context_loadpath,lick_loadpath,savepath,performance_loadpath=None):

    all_files = os.listdir(stim_context_loadpath)
    all_files = [f for f in all_files if f.endswith('stim_context_modulation.pkl')]
    for ff in all_files:
        if ff==all_files[0]:
            all_stim_context_data=pd.read_pickle(os.path.join(stim_context_loadpath,ff))
        else:
            all_stim_context_data=pd.concat([all_stim_context_data,pd.read_pickle(os.path.join(stim_context_loadpath,ff))],axis=0)
    #drop waveforms if they exist
    all_stim_context_data.drop(columns=['waveform_sd','waveform_mean'], inplace=True, errors='ignore')

    # load and concat all the lick-mod dataframes
    all_lick_files = os.listdir(lick_loadpath)
    all_lick_files = [f for f in all_lick_files if f.endswith('lick_modulation.pkl')]
    for ff in all_lick_files:
        if ff==all_lick_files[0]:
            all_lick_data=pd.read_pickle(os.path.join(lick_loadpath,ff))
        else:
            all_lick_data=pd.concat([all_lick_data,pd.read_pickle(os.path.join(lick_loadpath,ff))],axis=0)

    #concat stim & context with lick dataframe
    all_data=all_stim_context_data.merge(all_lick_data, on=['unit_id','project','session_id'], how='left')

    #get behavioral performance to append to table
    if performance_loadpath is not None:
        all_performance=pd.read_pickle(performance_loadpath)
    else:
        all_performance=get_all_performance()

    for ss in list(all_performance.keys()):
        if ss+'' in all_data['session_id'].values:
            all_data.loc[all_data['session_id']==ss+'', 'cross_modal_dprime']=all_performance[ss]['cross_modal_dprime'].mean()
            all_data.loc[all_data['session_id']==ss+'', 'n_good_blocks']=np.sum(all_performance[ss]['cross_modal_dprime']>=1.0)
        elif ss in all_data['session_id'].values:
            all_data.loc[all_data['session_id']==ss, 'cross_modal_dprime']=all_performance[ss]['cross_modal_dprime'].mean()
            all_data.loc[all_data['session_id']==ss, 'n_good_blocks']=np.sum(all_performance[ss]['cross_modal_dprime']>=1.0)

    all_data.to_pickle(os.path.join(savepath,"all_data_plus_performance.pkl"))


def calculate_single_unit_metric_adjusted_pvals(sel_units,sel_project):

    adj_pvals=pd.DataFrame({
        'unit_id':sel_units['unit_id'].values,
        'session_id':sel_units['session_id'].values,
        'structure':sel_units['structure'].values,
        'location':sel_units['location'].values,
        'peak_to_valley':sel_units['peak_to_valley'].values,
        'vis1':fdrcorrection(sel_units['vis1_stimulus_modulation_p_value'])[1],
        'vis2':fdrcorrection(sel_units['vis2_stimulus_modulation_p_value'])[1],
        'sound1':fdrcorrection(sel_units['sound1_stimulus_modulation_p_value'])[1],
        'sound2':fdrcorrection(sel_units['sound2_stimulus_modulation_p_value'])[1],
        'catch':fdrcorrection(sel_units['catch_stimulus_modulation_p_value'])[1],
        'vis1_late':fdrcorrection(sel_units['vis1_stimulus_late_modulation_p_value'])[1],
        'vis2_late':fdrcorrection(sel_units['vis2_stimulus_late_modulation_p_value'])[1],
        'sound1_late':fdrcorrection(sel_units['sound1_stimulus_late_modulation_p_value'])[1],
        'sound2_late':fdrcorrection(sel_units['sound2_stimulus_late_modulation_p_value'])[1],
        'catch_late':fdrcorrection(sel_units['catch_stimulus_late_modulation_p_value'])[1],
        'context':fdrcorrection(sel_units['baseline_context_modulation_p_value'])[1],
        # 'context_linear_shift':sel_units[['linear_shift_baseline_context_p_value_higher',
        #                                 'linear_shift_baseline_context_p_value_lower']].min(axis=1),
        # 'context_linear_shift_diff_from_null':((sel_units['linear_shift_baseline_context_true_value']-
        #                                     sel_units['linear_shift_baseline_context_null_median'])/
        #                                         sel_units['linear_shift_baseline_context_null_std']),

        'vis1_latency':sel_units['vis1_stimulus_latency'],
        'vis2_latency':sel_units['vis2_stimulus_latency'],
        'sound1_latency':sel_units['sound1_stimulus_latency'],
        'sound2_latency':sel_units['sound2_stimulus_latency'],
        'catch_latency':sel_units['catch_stimulus_latency'],

        'vis1_roc_auc':sel_units['vis1_stimulus_modulation_roc_auc'],
        'vis2_roc_auc':sel_units['vis2_stimulus_modulation_roc_auc'],
        'sound1_roc_auc':sel_units['sound1_stimulus_modulation_roc_auc'],
        'sound2_roc_auc':sel_units['sound2_stimulus_modulation_roc_auc'],
        'context_roc_auc':sel_units['baseline_context_roc_auc'],

        # 'lick_vis':fdrcorrection(sel_units['vis_lick_modulation_p_value'])[1],
        # 'lick_aud':fdrcorrection(sel_units['aud_lick_modulation_p_value'])[1],
        'lick':fdrcorrection(sel_units['lick_modulation_p_value'])[1],
        'lick_roc_auc':sel_units['lick_modulation_roc_auc'],

        'context_sign':sel_units['baseline_context_modulation_sign'],

        'vis1_context':fdrcorrection(sel_units['vis1_context_modulation_p_value'])[1],
        'vis2_context':fdrcorrection(sel_units['vis2_context_modulation_p_value'])[1],
        'sound1_context':fdrcorrection(sel_units['sound1_context_modulation_p_value'])[1],
        'sound2_context':fdrcorrection(sel_units['sound2_context_modulation_p_value'])[1],
        'catch_context':fdrcorrection(sel_units['catch_context_modulation_p_value'])[1],

        'vis1_context_roc_auc':sel_units['vis1_context_modulation_roc_auc'],
        'vis2_context_roc_auc':sel_units['vis2_context_modulation_roc_auc'],
        'sound1_context_roc_auc':sel_units['sound1_context_modulation_roc_auc'],
        'sound2_context_roc_auc':sel_units['sound2_context_modulation_roc_auc'],
        'catch_context_roc_auc':sel_units['catch_context_modulation_roc_auc'],

        # 'vis1_context_sign':sel_units['vis1_context_modulation_zscore'],
        # 'vis2_context_sign':sel_units['vis2_context_modulation_zscore'],
        # 'sound1_context_sign':sel_units['sound1_context_modulation_zscore'],
        # 'sound2_context_sign':sel_units['sound2_context_modulation_zscore'],
        'vis1_context_sign':sel_units['vis1_context_modulation_sign'],
        'vis2_context_sign':sel_units['vis2_context_modulation_sign'],
        'sound1_context_sign':sel_units['sound1_context_modulation_sign'],
        'sound2_context_sign':sel_units['sound2_context_modulation_sign'],

        'vis1_context_evoked':fdrcorrection(sel_units['vis1_evoked_context_modulation_p_value'])[1],
        'vis2_context_evoked':fdrcorrection(sel_units['vis2_evoked_context_modulation_p_value'])[1],
        'sound1_context_evoked':fdrcorrection(sel_units['sound1_evoked_context_modulation_p_value'])[1],
        'sound2_context_evoked':fdrcorrection(sel_units['sound2_evoked_context_modulation_p_value'])[1],
        'catch_context_evoked':fdrcorrection(sel_units['catch_evoked_context_modulation_p_value'])[1],

        'vis1_context_evoked_roc_auc':sel_units['vis1_evoked_context_modulation_roc_auc'],
        'vis2_context_evoked_roc_auc':sel_units['vis2_evoked_context_modulation_roc_auc'],
        'sound1_context_evoked_roc_auc':sel_units['sound1_evoked_context_modulation_roc_auc'],
        'sound2_context_evoked_roc_auc':sel_units['sound2_evoked_context_modulation_roc_auc'],
        'catch_context_evoked_roc_auc':sel_units['catch_evoked_context_modulation_roc_auc'],

        'ccf_ap':sel_units['ccf_ap'],
        'ccf_dv':sel_units['ccf_dv'],
        'ccf_ml':sel_units['ccf_ml'],
    })

    # if 'Templeton' in sel_project:
    #     adj_pvals['lick']=np.ones(len(adj_pvals))

    adj_pvals['any_stim']=adj_pvals[['vis1','vis2','sound1','sound2']].min(axis=1)

    return adj_pvals



def calculate_stimulus_modulation_by_area(sel_units,sel_project,plot_figures=False,savepath=None):
    # stimulus responsiveness by area

    area_number_responsive_to_stim={
            'area':[],
            'vis1':[],
            'vis2':[],
            'sound1':[],
            'sound2':[],
            'both_vis':[],
            'both_sound':[],
            'mixed':[],
            'none':[],
            'vis1_pos':[],
            'vis2_pos':[],
            'sound1_pos':[],
            'sound2_pos':[],
            'both_vis_pos':[],
            'both_sound_pos':[],
            'mixed_pos':[],
            'vis1_neg':[],
            'vis2_neg':[],
            'sound1_neg':[],
            'sound2_neg':[],
            'both_vis_neg':[],
            'both_sound_neg':[],
            'mixed_neg':[],
            'total_n':[],
            'n_sessions':[],
            'n_sessions_w_20_units':[],
            'n_sessions_w_15_units':[],
            'n_sessions_w_10_units':[],
    }

    for sel_area in sel_units['structure'].unique():

            area_units=sel_units.query('structure==@sel_area')

            n_sessions=len(area_units['session_id'].unique())

            for n_units in [20,15,10]:
                    n_sessions_w_units=area_units.groupby('session_id').filter(lambda x: len(x)>=n_units)['session_id'].unique()
                    area_number_responsive_to_stim['n_sessions_w_'+str(n_units)+'_units'].append(len(n_sessions_w_units))

            adj_pvals=pd.DataFrame({
            'unit_id':area_units['unit_id'],
            'vis1':fdrcorrection(area_units['vis1_stimulus_modulation_p_value'])[1],
            'vis2':fdrcorrection(area_units['vis2_stimulus_modulation_p_value'])[1],
            'sound1':fdrcorrection(area_units['sound1_stimulus_modulation_p_value'])[1],
            'sound2':fdrcorrection(area_units['sound2_stimulus_modulation_p_value'])[1],
            'vis1_sign':area_units['vis1_stimulus_modulation_sign'],
            'vis2_sign':area_units['vis2_stimulus_modulation_sign'],
            'sound1_sign':area_units['sound1_stimulus_modulation_sign'],
            'sound2_sign':area_units['sound2_stimulus_modulation_sign'],
            })

            #stimulus modulation across all units
            #each stim only
            vis1_stim_resp=adj_pvals.query('vis1<0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05')
            vis2_stim_resp=adj_pvals.query('vis2<0.05 and vis1>=0.05 and sound1>=0.05 and sound2>=0.05')
            sound1_stim_resp=adj_pvals.query('sound1<0.05 and sound2>=0.05 and vis1>=0.05 and vis2>=0.05')
            sound2_stim_resp=adj_pvals.query('sound2<0.05 and sound1>=0.05 and vis1>=0.05 and vis2>=0.05')

            #both vis
            both_vis_stim_resp=adj_pvals.query('vis1<0.05 and vis2<0.05 and sound1>=0.05 and sound2>=0.05')
            #both aud
            both_sound_stim_resp=adj_pvals.query('sound1<0.05 and sound2<0.05 and vis1>=0.05 and vis2>=0.05')

            #at least one vis and one aud
            mixed_stim_resp=adj_pvals.query('((vis1<0.05 or vis2<0.05) and (sound1<0.05 and sound2<0.05))')

            #none
            no_stim_resp=adj_pvals.query('vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05')

            area_number_responsive_to_stim['area'].append(sel_area)
            area_number_responsive_to_stim['vis1'].append(len(vis1_stim_resp))
            area_number_responsive_to_stim['vis2'].append(len(vis2_stim_resp))
            area_number_responsive_to_stim['sound1'].append(len(sound1_stim_resp))
            area_number_responsive_to_stim['sound2'].append(len(sound2_stim_resp))
            area_number_responsive_to_stim['both_vis'].append(len(both_vis_stim_resp))
            area_number_responsive_to_stim['both_sound'].append(len(both_sound_stim_resp))
            area_number_responsive_to_stim['mixed'].append(len(mixed_stim_resp))
            area_number_responsive_to_stim['none'].append(len(no_stim_resp))
            area_number_responsive_to_stim['total_n'].append(len(area_units))
            area_number_responsive_to_stim['n_sessions'].append(n_sessions)

            #positive vs. negative modulation
            #positive modulation
            vis1_pos_stim_resp=adj_pvals.query('vis1<0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and vis1_sign>0')
            vis2_pos_stim_resp=adj_pvals.query('vis2<0.05 and vis1>=0.05 and sound1>=0.05 and sound2>=0.05 and vis2_sign>0')
            sound1_pos_stim_resp=adj_pvals.query('sound1<0.05 and sound2>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1_sign>0')
            sound2_pos_stim_resp=adj_pvals.query('sound2<0.05 and sound1>=0.05 and vis1>=0.05 and vis2>=0.05 and sound2_sign>0')

            #both vis
            both_vis_pos_stim_resp=adj_pvals.query('vis1<0.05 and vis2<0.05 and sound1>=0.05 and sound2>=0.05 and vis1_sign>0 and vis2_sign>0')
            #both aud
            both_sound_pos_stim_resp=adj_pvals.query('sound1<0.05 and sound2<0.05 and vis1>=0.05 and vis2>=0.05 and sound1_sign>0 and sound2_sign>0')

            #at least one vis and one aud
            mixed_pos_stim_resp=adj_pvals.query('(((vis1<0.05 and vis1_sign>0) or (vis2<0.05 and vis2_sign>0)) and ((sound1<0.05 and sound1_sign>0) and (sound2<0.05 and sound2_sign>0)))')

            #negative modulation
            vis1_neg_stim_resp=adj_pvals.query('vis1<0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and vis1_sign<0')
            vis2_neg_stim_resp=adj_pvals.query('vis2<0.05 and vis1>=0.05 and sound1>=0.05 and sound2>=0.05 and vis2_sign<0')
            sound1_neg_stim_resp=adj_pvals.query('sound1<0.05 and sound2>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1_sign<0')
            sound2_neg_stim_resp=adj_pvals.query('sound2<0.05 and sound1>=0.05 and vis1>=0.05 and vis2>=0.05 and sound2_sign<0')

            #both vis
            both_vis_neg_stim_resp=adj_pvals.query('vis1<0.05 and vis2<0.05 and sound1>=0.05 and sound2>=0.05 and vis1_sign<0 and vis2_sign<0')
            #both aud
            both_sound_neg_stim_resp=adj_pvals.query('sound1<0.05 and sound2<0.05 and vis1>=0.05 and vis2>=0.05 and sound1_sign<0 and sound2_sign<0')

            #at least one vis and one aud
            mixed_neg_stim_resp=adj_pvals.query('(((vis1<0.05 and vis1_sign<0) or (vis2<0.05 and vis2_sign<0)) and ((sound1<0.05 and sound1_sign<0) and (sound2<0.05 and sound2_sign<0)))')

            area_number_responsive_to_stim['vis1_pos'].append(len(vis1_pos_stim_resp))
            area_number_responsive_to_stim['vis2_pos'].append(len(vis2_pos_stim_resp))
            area_number_responsive_to_stim['sound1_pos'].append(len(sound1_pos_stim_resp))
            area_number_responsive_to_stim['sound2_pos'].append(len(sound2_pos_stim_resp))
            area_number_responsive_to_stim['both_vis_pos'].append(len(both_vis_pos_stim_resp))
            area_number_responsive_to_stim['both_sound_pos'].append(len(both_sound_pos_stim_resp))
            area_number_responsive_to_stim['mixed_pos'].append(len(mixed_pos_stim_resp))

            area_number_responsive_to_stim['vis1_neg'].append(len(vis1_neg_stim_resp))
            area_number_responsive_to_stim['vis2_neg'].append(len(vis2_neg_stim_resp))
            area_number_responsive_to_stim['sound1_neg'].append(len(sound1_neg_stim_resp))
            area_number_responsive_to_stim['sound2_neg'].append(len(sound2_neg_stim_resp))
            area_number_responsive_to_stim['both_vis_neg'].append(len(both_vis_neg_stim_resp))
            area_number_responsive_to_stim['both_sound_neg'].append(len(both_sound_neg_stim_resp))
            area_number_responsive_to_stim['mixed_neg'].append(len(mixed_neg_stim_resp))

            labels=['vis1 only','vis2 only','both vis',
                    'sound1 only','sound2 only','both sound',
                    'mixed','none']
            
            sizes=[len(vis1_stim_resp),len(vis2_stim_resp),len(both_vis_stim_resp),
                    len(sound1_stim_resp),len(sound2_stim_resp),len(both_sound_stim_resp),
                    len(mixed_stim_resp),len(no_stim_resp)]
            
            if np.sum(sizes)>0 and plot_figures:
                    fig,ax=plt.subplots()
                    ax.pie(sizes,labels=labels,autopct='%1.1f%%')
                    ax.set_title('area='+sel_area+'; n_units='+str(len(area_units))+'; n_sessions='+str(n_sessions))
                    fig.suptitle('stimulus responsive units')
                    fig.tight_layout()

    area_number_responsive_to_stim=pd.DataFrame(area_number_responsive_to_stim)

    area_fraction_responsive_to_stim=area_number_responsive_to_stim.copy()

    for rr,row in area_fraction_responsive_to_stim.iterrows():
        if row['total_n']>0:
            area_fraction_responsive_to_stim.iloc[rr,1:-5]=row.iloc[1:-5]/row['total_n']

    if savepath is not None:
        if 'Templeton' in sel_project:
            temp_savepath=os.path.join(savepath,"stimulus_responsiveness_by_area_Templeton.csv")
        else:
            temp_savepath=os.path.join(savepath,"stimulus_responsiveness_by_area_DR.csv")
        area_fraction_responsive_to_stim.to_csv(temp_savepath)

    return area_fraction_responsive_to_stim



def compute_context_stim_lick_modulation_by_area(sel_units,sel_project,plot_figures=False,savepath=None):

# context modulation vs. stimulus modulation vs. lick modulation

    area_number_context_mod={
            'area':[],
            'any_stim':[],
            'only_stim':[],
            'any_context':[],
            'only_context':[],
            # 'any_context_linear_shift':[],
            # 'only_context_linear_shift':[],
            'any_lick':[],
            'only_lick':[],
            'stim_and_context':[],
            'lick_and_stim':[],
            'lick_and_context':[],
            'lick_and_stim_and_context':[],
            'none':[],
            'any_context_pos':[],
            'any_context_neg':[],
            'any_lick_pos':[],
            'any_lick_neg':[],
            'any_stim_pos':[],
            'any_stim_neg':[],
            'total_n':[],
            'n_sessions':[],
            'n_sessions_w_20_units':[],
            'n_sessions_w_15_units':[],
            'n_sessions_w_10_units':[],
    }

    for sel_area in sel_units['structure'].unique():

            area_units=sel_units.query('structure==@sel_area')
            
            n_sessions=len(area_units['session_id'].unique())

            for n_units in [20,15,10]:
                    n_sessions_w_units=area_units.groupby('session_id').filter(lambda x: len(x)>=n_units)['session_id'].unique()
                    area_number_context_mod['n_sessions_w_'+str(n_units)+'_units'].append(len(n_sessions_w_units))

            adj_pvals=pd.DataFrame({
            'unit_id':area_units['unit_id'],
            'vis1':fdrcorrection(area_units['vis1_stimulus_modulation_p_value'])[1],
            'vis2':fdrcorrection(area_units['vis2_stimulus_modulation_p_value'])[1],
            'sound1':fdrcorrection(area_units['sound1_stimulus_modulation_p_value'])[1],
            'sound2':fdrcorrection(area_units['sound2_stimulus_modulation_p_value'])[1],
            'context':fdrcorrection(area_units['baseline_context_modulation_p_value'])[1],
            'lick':fdrcorrection(area_units['lick_modulation_p_value'])[1],
            'vis1_sign':area_units['vis1_stimulus_modulation_sign'],
            'vis2_sign':area_units['vis2_stimulus_modulation_sign'],
            'sound1_sign':area_units['sound1_stimulus_modulation_sign'],
            'sound2_sign':area_units['sound2_stimulus_modulation_sign'],
            'context_sign':area_units['baseline_context_modulation_sign'],
            'lick_sign':area_units['lick_modulation_sign'],
            # 'context_linear_shift':area_units[['linear_shift_baseline_context_p_value_higher',
            #                                 'linear_shift_baseline_context_p_value_lower']].min(axis=1),
            })

            #lick modulation only
            only_lick_resp=adj_pvals.query('lick<0.05 and context>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05')
            #any lick modulation
            any_lick_resp=adj_pvals.query('lick<0.05')
            #lick and context
            lick_and_context_resp=adj_pvals.query('context<0.05 and lick<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05')
            #lick and stimulus
            lick_and_stim_resp=adj_pvals.query('lick<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and context>=0.05')
            #all three
            all_resp=adj_pvals.query('context<0.05 and lick<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05)')
            
            #stimulus modulation only
            only_stim_resp=adj_pvals.query('(vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and context>=0.05 and lick>=0.05')
            #any stim modulation
            any_stim_resp=adj_pvals.query('vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05')
        
            #context modulation only
            only_context_resp=adj_pvals.query('context<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05')
            #any context modulation
            any_context_resp=adj_pvals.query('context<0.05')

            # #linear-shifted conext modulation
            # only_context_linear_shift_resp=adj_pvals.query('context_linear_shift<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05')
            # #any context modulation
            # any_context_linear_shift_resp=adj_pvals.query('context_linear_shift<0.05')

            #stim and context modulation
            stim_and_context_resp=adj_pvals.query('context<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and lick>=0.05')
            #none
            no_resp=adj_pvals.query('context>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05')
            
            #pos vs. neg modulation
            #context
            any_context_pos=adj_pvals.query('context<0.05 and context_sign>0')
            any_context_neg=adj_pvals.query('context<0.05 and context_sign<0')
            #lick
            any_lick_pos=adj_pvals.query('lick<0.05 and lick_sign>0')
            any_lick_neg=adj_pvals.query('lick<0.05 and lick_sign<0')
            #stim
            any_stim_pos=adj_pvals.query('(vis1<0.05 and vis1_sign>0) or (vis2<0.05 and vis2_sign>0) or (sound1<0.05 and sound1_sign>0) or (sound2<0.05 and sound2_sign>0)')
            any_stim_neg=adj_pvals.query('(vis1<0.05 and vis1_sign<0) or (vis2<0.05 and vis2_sign<0) or (sound1<0.05 and sound1_sign<0) or (sound2<0.05 and sound2_sign<0)')

            area_number_context_mod['area'].append(sel_area)
            area_number_context_mod['only_stim'].append(len(only_stim_resp))
            area_number_context_mod['any_stim'].append(len(any_stim_resp))
            area_number_context_mod['only_context'].append(len(only_context_resp))
            area_number_context_mod['any_context'].append(len(any_context_resp))
            # area_number_context_mod['any_context_linear_shift'].append(len(any_context_linear_shift_resp))
            # area_number_context_mod['only_context_linear_shift'].append(len(only_context_linear_shift_resp))
            area_number_context_mod['only_lick'].append(len(only_lick_resp))
            area_number_context_mod['any_lick'].append(len(any_lick_resp))
            area_number_context_mod['stim_and_context'].append(len(stim_and_context_resp))
            area_number_context_mod['lick_and_stim'].append(len(lick_and_stim_resp))
            area_number_context_mod['lick_and_context'].append(len(lick_and_context_resp))
            area_number_context_mod['lick_and_stim_and_context'].append(len(all_resp))
            area_number_context_mod['none'].append(len(no_resp))
            area_number_context_mod['total_n'].append(len(area_units))
            area_number_context_mod['n_sessions'].append(n_sessions)

            area_number_context_mod['any_context_pos'].append(len(any_context_pos))
            area_number_context_mod['any_context_neg'].append(len(any_context_neg))
            area_number_context_mod['any_lick_pos'].append(len(any_lick_pos))
            area_number_context_mod['any_lick_neg'].append(len(any_lick_neg))
            area_number_context_mod['any_stim_pos'].append(len(any_stim_pos))
            area_number_context_mod['any_stim_neg'].append(len(any_stim_neg))

            # labels=['stimulus only','stimulus and context','context only','neither']
            # sizes=[len(any_stim_resp),len(stim_and_context_resp),len(context_resp),
            #         len(neither_stim_nor_context_resp)]
            
            labels=['stimulus only','stimulus and context','context only',
                    'context and lick','lick only', 'lick & stimulus & context',
                    'lick and stimulus',  'none']
            sizes=[len(only_stim_resp),len(stim_and_context_resp),len(only_context_resp),
                    len(lick_and_context_resp),len(only_lick_resp),len(all_resp),
                    len(lick_and_stim_resp), len(no_resp)]
            
            if np.sum(sizes)>0 and plot_figures:
                    fig,ax=plt.subplots()
                    ax.pie(sizes,labels=labels,autopct='%1.1f%%',
                    colors=['tab:blue', 'tab:orange', 'tab:green',
                            'tab:red' , 'tab:purple', 'tab:brown', 
                            'tab:pink', 'grey'])
                    ax.set_title('area='+sel_area+'; n_units='+str(len(area_units))+'; n_sessions='+str(n_sessions))

                    fig.tight_layout()

    area_number_context_mod=pd.DataFrame(area_number_context_mod)

    area_fraction_context_mod=area_number_context_mod.copy()

    for rr,row in area_fraction_context_mod.iterrows():
        if row['total_n']>0:
            area_fraction_context_mod.iloc[rr,1:-5]=row.iloc[1:-5]/row['total_n']

    if savepath is not None:
        if 'Templeton' in sel_project:
            temp_savepath=os.path.join(savepath,'area_fraction_context_stim_lick_mod_Templeton.csv')
        else:
            temp_savepath=os.path.join(savepath,'area_fraction_context_stim_lick_mod_DR.csv')
        area_fraction_context_mod.to_csv(temp_savepath)

    return area_fraction_context_mod


def calculate_context_mod_stim_resp_by_area(sel_units,sel_project,plot_figures=False,savepath=None):
    
        # context mod of stimulus responsiveness by area

        area_number_context_stim_mod={
                'area':[],
                'vis1':[],
                'vis2':[],
                'sound1':[],
                'sound2':[],
                'both_vis':[],
                'both_sound':[],
                'mixed':[],
                'none':[],
                'vis1_evoked':[],
                'vis2_evoked':[],
                'sound1_evoked':[],
                'sound2_evoked':[],
                'both_vis_evoked':[],
                'both_sound_evoked':[],
                'mixed_evoked':[],
                'none_evoked':[],
                'total_n':[],
                'n_sessions':[],
                'n_stim_responsive':[],
        }

        for sel_area in sel_units['structure'].unique():

                area_units=sel_units.query('structure==@sel_area')

                n_sessions=len(area_units['session_id'].unique())

                adj_pvals=pd.DataFrame({
                        'unit_id':area_units['unit_id'],
                        'vis1':fdrcorrection(area_units['vis1_stimulus_modulation_p_value'])[1],
                        'vis2':fdrcorrection(area_units['vis2_stimulus_modulation_p_value'])[1],
                        'sound1':fdrcorrection(area_units['sound1_stimulus_modulation_p_value'])[1],
                        'sound2':fdrcorrection(area_units['sound2_stimulus_modulation_p_value'])[1],
                        'vis1_context':fdrcorrection(area_units['vis1_context_modulation_p_value'])[1],
                        'vis2_context':fdrcorrection(area_units['vis2_context_modulation_p_value'])[1],
                        'sound1_context':fdrcorrection(area_units['sound1_context_modulation_p_value'])[1],
                        'sound2_context':fdrcorrection(area_units['sound2_context_modulation_p_value'])[1],
                        'vis1_evoked_context':fdrcorrection(area_units['vis1_evoked_context_modulation_p_value'])[1],
                        'vis2_evoked_context':fdrcorrection(area_units['vis2_evoked_context_modulation_p_value'])[1],
                        'sound1_evoked_context':fdrcorrection(area_units['sound1_evoked_context_modulation_p_value'])[1],
                        'sound2_evoked_context':fdrcorrection(area_units['sound2_evoked_context_modulation_p_value'])[1],
                })
                
                adj_pvals['any_stim']=adj_pvals[['vis1','vis2','sound1','sound2']].min(axis=1)
                
                #stimulus context modulation
                vis1_context_stim_mod=adj_pvals.query('vis1_context<0.05 and vis2_context>=0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05')
                vis2_context_stim_mod=adj_pvals.query('vis2_context<0.05 and vis1_context>=0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05')
                sound1_context_stim_mod=adj_pvals.query('sound1_context<0.05 and sound2_context>=0.05 and vis1_context>=0.05 and vis2_context>=0.05 and any_stim<0.05')
                sound2_context_stim_mod=adj_pvals.query('sound2_context<0.05 and sound1_context>=0.05 and vis1_context>=0.05 and vis2_context>=0.05 and any_stim<0.05')

                both_vis_context_stim_mod=adj_pvals.query('vis1_context<0.05 and vis2_context<0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05')
                both_aud_context_stim_mod=adj_pvals.query('sound1_context<0.05 and sound2_context<0.05 and vis1_context>=0.05 and vis2_context>=0.05 and any_stim<0.05')
                multi_modal_context_stim_mod=adj_pvals.query('((vis1_context<0.05 or vis2_context<0.05) and (sound1_context<0.05 or sound2_context<0.05)) and any_stim<0.05')

                no_context_stim_mod=adj_pvals.query('vis1_context>=0.05 and vis2_context>=0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05')
                
                #evoked stimulus context modulation
                vis1_context_evoked_stim_mod=adj_pvals.query('vis1_evoked_context<0.05 and vis2_evoked_context>=0.05 and sound1_evoked_context>=0.05 and sound2_evoked_context>=0.05 and any_stim<0.05')
                vis2_context_evoked_stim_mod=adj_pvals.query('vis2_evoked_context<0.05 and vis1_evoked_context>=0.05 and sound1_evoked_context>=0.05 and sound2_evoked_context>=0.05 and any_stim<0.05')
                sound1_context_evoked_stim_mod=adj_pvals.query('sound1_evoked_context<0.05 and sound2_evoked_context>=0.05 and vis1_evoked_context>=0.05 and vis2_evoked_context>=0.05 and any_stim<0.05')
                sound2_context_evoked_stim_mod=adj_pvals.query('sound2_evoked_context<0.05 and sound1_evoked_context>=0.05 and vis1_evoked_context>=0.05 and vis2_evoked_context>=0.05 and any_stim<0.05')

                both_vis_context_evoked_stim_mod=adj_pvals.query('vis1_evoked_context<0.05 and vis2_evoked_context<0.05 and sound1_evoked_context>=0.05 and sound2_evoked_context>=0.05 and any_stim<0.05')
                both_aud_context_evoked_stim_mod=adj_pvals.query('sound1_evoked_context<0.05 and sound2_evoked_context<0.05 and vis1_evoked_context>=0.05 and vis2_evoked_context>=0.05 and any_stim<0.05')
                multi_modal_context_evoked_stim_mod=adj_pvals.query('((vis1_evoked_context<0.05 or vis2_evoked_context<0.05) and (sound1_evoked_context<0.05 or sound2_evoked_context<0.05)) and any_stim<0.05')

                no_context_evoked_stim_mod=adj_pvals.query('vis1_evoked_context>=0.05 and vis2_evoked_context>=0.05 and sound1_evoked_context>=0.05 and sound2_evoked_context>=0.05 and any_stim<0.05')

                n_stim_resp_units=np.sum(adj_pvals['any_stim']<0.05)

                area_number_context_stim_mod['area'].append(sel_area)
                area_number_context_stim_mod['vis1'].append(len(vis1_context_stim_mod))
                area_number_context_stim_mod['vis2'].append(len(vis2_context_stim_mod))
                area_number_context_stim_mod['sound1'].append(len(sound1_context_stim_mod))
                area_number_context_stim_mod['sound2'].append(len(sound2_context_stim_mod))
                area_number_context_stim_mod['both_vis'].append(len(both_vis_context_stim_mod))
                area_number_context_stim_mod['both_sound'].append(len(both_aud_context_stim_mod))
                area_number_context_stim_mod['mixed'].append(len(multi_modal_context_stim_mod))
                area_number_context_stim_mod['none'].append(len(no_context_stim_mod))
                area_number_context_stim_mod['vis1_evoked'].append(len(vis1_context_evoked_stim_mod))
                area_number_context_stim_mod['vis2_evoked'].append(len(vis2_context_evoked_stim_mod))
                area_number_context_stim_mod['sound1_evoked'].append(len(sound1_context_evoked_stim_mod))
                area_number_context_stim_mod['sound2_evoked'].append(len(sound2_context_evoked_stim_mod))
                area_number_context_stim_mod['both_vis_evoked'].append(len(both_vis_context_evoked_stim_mod))
                area_number_context_stim_mod['both_sound_evoked'].append(len(both_aud_context_evoked_stim_mod))
                area_number_context_stim_mod['mixed_evoked'].append(len(multi_modal_context_evoked_stim_mod))
                area_number_context_stim_mod['none_evoked'].append(len(no_context_evoked_stim_mod))
                area_number_context_stim_mod['total_n'].append(len(adj_pvals))
                area_number_context_stim_mod['n_stim_responsive'].append(n_stim_resp_units)
                area_number_context_stim_mod['n_sessions'].append(n_sessions)

                labels=['vis1 only','vis2 only','both vis',
                        'sound1 only','sound2 only','both sound',
                        'mixed','none']
                
                sizes=[len(vis1_context_stim_mod),len(vis2_context_stim_mod),len(both_vis_context_stim_mod),
                        len(sound1_context_stim_mod),len(sound2_context_stim_mod),len(both_aud_context_stim_mod),
                        len(multi_modal_context_stim_mod),len(no_context_stim_mod)]
                
                if np.sum(sizes)>0 and plot_figures:
                        fig,ax=plt.subplots()
                        ax.pie(sizes,labels=labels,autopct='%1.1f%%')
                        ax.set_title('area='+sel_area+'; n_stim_resp_units='+str(n_stim_resp_units)+'; n_total_units='+str(len(adj_pvals))+'; n_sessions='+str(n_sessions))
                        fig.suptitle('context modulation of stimulus response')
                        fig.tight_layout()


        area_number_context_stim_mod=pd.DataFrame(area_number_context_stim_mod)

        area_fraction_context_stim_mod=area_number_context_stim_mod.copy()

        for rr,row in area_fraction_context_stim_mod.iterrows():
                if row['n_stim_responsive']>0:
                        area_fraction_context_stim_mod.iloc[rr,1:-4]=row.iloc[1:-4]/row['n_stim_responsive']
        if savepath is not None:
                if 'Templeton' in sel_project:
                        temp_savepath=os.path.join(savepath,'area_fraction_context_stim_mod_Templeton.csv')

                else:
                        temp_savepath=os.path.join(savepath,'area_fraction_context_stim_mod_DR.csv')

                area_fraction_context_stim_mod.to_csv(temp_savepath)

        return area_fraction_context_stim_mod