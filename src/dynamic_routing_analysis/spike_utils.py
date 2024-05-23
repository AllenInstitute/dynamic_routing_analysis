import os

import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr
from sklearn.metrics import roc_auc_score

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


def get_structure_probe(units):

    unique_areas=units[:]['structure'].unique()

    structure_probe=pd.DataFrame({
        'structure_probe':np.full(len(units[:]),'',dtype=object),
        'unit_id':units[:]['unit_id']
    })

    for aa in unique_areas:
        unique_probes=units[:].query('structure==@aa')['group_name'].unique()

        if len(unique_probes)>1:
            for up in unique_probes:
                unit_ids=units[:].query('structure==@aa and group_name==@up')['unit_id'].values
                structure_probe.loc[structure_probe['unit_id'].isin(unit_ids),'structure_probe']=aa+'_'+up
        elif len(unique_probes)==1:
            unit_ids=units[:].query('structure==@aa')['unit_id'].values
            structure_probe.loc[structure_probe['unit_id'].isin(unit_ids),'structure_probe']=aa
        else:
            print('no units in '+aa)

    return structure_probe


def compute_lick_modulation(trials, units, session_info, save_path):

    lick_modulation={
        'unit_id':[],
        'session_id':[],
        'project':[],
        # 'structure':[],
    }

    lick_modulation['lick_modulation_index'] = []
    lick_modulation['lick_modulation_zscore'] = []
    lick_modulation['lick_modulation_p_value'] = []
    lick_modulation['lick_modulation_sign'] = []
    lick_modulation['lick_modulation_roc_auc'] = []

    #make data array first
    time_before = 0.5
    time_after = 0.5
    binsize = 0.025
    trial_da = make_neuron_time_trials_tensor(units, trials, time_before, time_after, binsize)
                                                                              
    if "Templeton" in session_info.project:
        lick_trials = trials.query('is_response==True')
        non_lick_trials = trials.query('is_response==False')
        baseline_trials = trials

    elif "DynamicRouting" in session_info.project:
        lick_trials = trials.query('(stim_name=="vis1" and context_name=="aud" and is_response==True) or \
                                (stim_name=="sound1" and context_name=="vis" and is_response==True)')
        non_lick_trials = trials.query('(stim_name=="vis1" and context_name=="aud" and is_response==False) or \
                                        (stim_name=="sound1" and context_name=="vis" and is_response==False)')
        baseline_trials = trials.query('(stim_name=="vis1" and context_name=="aud") or \
                                        (stim_name=="sound1" and context_name=="vis")')
    else:
        print('incompatible project: ',session_info.project,'; skipping')
        return

    #for each unit
    for uu,unit in units.iterrows():
        if 'Templeton' in session_info.project:
            continue

        lick_modulation['unit_id'].append(unit['unit_id'])
        lick_modulation['session_id'].append(str(session_info.id))
        lick_modulation['project'].append(str(session_info.project))
        # lick_modulation['structure'].append(unit['structure'])

        #lick modulation
        lick_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.2,0.5),trials=lick_trials.index).mean(dim='time',skipna=True)
        non_lick_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.2,0.5),trials=non_lick_trials.index).mean(dim='time',skipna=True)
        
        baseline_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(-0.5,-0.2),trials=baseline_trials.index).mean(dim='time',skipna=True)
        
        # lick_diff = lick_frs_by_trial - non_lick_frs_by_trial

        lick_frs_by_trial_zscore = (lick_frs_by_trial.mean(skipna=True)-non_lick_frs_by_trial.mean(skipna=True))/baseline_frs_by_trial.std(skipna=True)
        lick_modulation['lick_modulation_zscore'].append(lick_frs_by_trial_zscore.mean(skipna=True).values)

        lick_modulation_index=(lick_frs_by_trial.mean(skipna=True)-non_lick_frs_by_trial.mean(skipna=True))/(lick_frs_by_trial.mean(skipna=True)+non_lick_frs_by_trial.mean(skipna=True))
        lick_modulation['lick_modulation_index'].append(lick_modulation_index.values)
        
        pval = st.mannwhitneyu(lick_frs_by_trial.values, non_lick_frs_by_trial.values,nan_policy='omit')[1]
        # pval = st.ranksums(lick_frs_by_trial.values, non_lick_frs_by_trial.values,nan_policy='omit')[1]
        lick_modulation['lick_modulation_p_value'].append(pval)

        stim_mod_sign=np.sign(lick_frs_by_trial.mean(skipna=True).values-non_lick_frs_by_trial.mean(skipna=True).values)
        lick_modulation['lick_modulation_sign'].append(stim_mod_sign)

        #ROC AUC
        binary_label = np.concatenate([np.ones(lick_frs_by_trial.size),np.zeros(non_lick_frs_by_trial.size)])
        binary_score = np.concatenate([lick_frs_by_trial.values,non_lick_frs_by_trial.values])
        lick_roc_auc = roc_auc_score(binary_label, binary_score)
        lick_modulation['lick_modulation_roc_auc'].append(lick_roc_auc)

    lick_modulation_df=pd.DataFrame(lick_modulation)
    lick_modulation_df.to_pickle(os.path.join(save_path,session_info.id+'_lick_modulation.pkl'))


def compute_stim_context_modulation(trials, units, session_info, save_path):

    stim_context_modulation = {
        'unit_id':[],
        'session_id':[],
        'project':[],
        'baseline_context_modulation_index':[],
        'baseline_context_modulation_p_value':[],
        'baseline_context_modulation_zscore':[],
        'baseline_context_modulation_sign':[],
        'baseline_context_roc_auc':[],
        'linear_shift_baseline_context_true_value':[],
        'linear_shift_baseline_context_null_median':[],
        'linear_shift_baseline_context_null_mean':[],
        'linear_shift_baseline_context_null_std':[],
        'linear_shift_baseline_context_p_value_higher':[],
        'linear_shift_baseline_context_p_value_lower':[],
        'vis_discrim_roc_auc':[],
        'aud_discrim_roc_auc':[],
        'target_discrim_roc_auc':[],
        'nontarget_discrim_roc_auc':[],
        'vis_vs_aud':[],
        'cr_vs_fa_early_roc_auc':[],
        'hit_vs_cr_early_roc_auc':[],
        'hit_vs_fa_early_roc_auc':[],
        'cr_vs_fa_mid_roc_auc':[],
        'hit_vs_cr_mid_roc_auc':[],
        'hit_vs_fa_mid_roc_auc':[],
        'cr_vs_fa_late_roc_auc':[],
        'hit_vs_cr_late_roc_auc':[],
        'hit_vs_fa_late_roc_auc':[],
    }
    for ss in trials['stim_name'].unique():
        stim_context_modulation[ss+'_context_modulation_index'] = []
        stim_context_modulation[ss+'_context_modulation_zscore'] = []
        stim_context_modulation[ss+'_context_modulation_sign'] = []
        stim_context_modulation[ss+'_context_modulation_p_value'] = []
        stim_context_modulation[ss+'_context_modulation_roc_auc'] = []
        stim_context_modulation[ss+'_evoked_context_modulation_index'] = []
        stim_context_modulation[ss+'_evoked_context_modulation_zscore'] = []
        stim_context_modulation[ss+'_evoked_context_modulation_sign'] = []
        stim_context_modulation[ss+'_evoked_context_modulation_p_value'] = []
        stim_context_modulation[ss+'_stimulus_modulation_index'] = []
        stim_context_modulation[ss+'_stimulus_modulation_zscore'] = []
        stim_context_modulation[ss+'_stimulus_modulation_p_value'] = []
        stim_context_modulation[ss+'_stimulus_modulation_sign'] = []
        stim_context_modulation[ss+'_stimulus_modulation_roc_auc'] = []
        stim_context_modulation[ss+'_stimulus_late_modulation_index'] = []
        stim_context_modulation[ss+'_stimulus_late_modulation_zscore'] = []
        stim_context_modulation[ss+'_stimulus_late_modulation_p_value'] = []
        stim_context_modulation[ss+'_stimulus_late_modulation_sign'] = []
        stim_context_modulation[ss+'_stimulus_late_modulation_roc_auc'] = []
        stim_context_modulation[ss+'_stim_latency'] = []

    contexts=trials['context_name'].unique()

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
        trials['true_context_name']=trials['context_name']

        for block in range(0,6):
            block_start_time=start_time+block*10*60
            block_end_time=start_time+(block+1)*10*60
            block_trials=trials.query('start_time>=@block_start_time').index
            fake_context[block_trials]=block_contexts[block]
            fake_block_nums[block_trials]=block
        
        trials['context_name']=fake_context
        trials['block_index']=fake_block_nums
        trials['is_vis_context']=trials['context_name']=='vis'
        trials['is_aud_context']=trials['context_name']=='aud'

    #make data array first
    time_before = 0.1
    time_after = 0.3
    binsize = 0.025
    trial_da = make_neuron_time_trials_tensor(units, trials, time_before, time_after, binsize)

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

    #for each unit
    for uu,unit in units.iterrows():

        stim_context_modulation['unit_id'].append(unit['unit_id'])
        stim_context_modulation['session_id'].append(str(session_info.id))
        stim_context_modulation['project'].append(str(session_info.project))

        #find baseline frs across all trials
        baseline_frs = trial_da.sel(unit_id=unit['unit_id'],time=slice(-0.1,0)).mean(dim='time')

        vis_baseline_frs = baseline_frs.sel(trials=trials.query('context_name=="vis"').index)
        aud_baseline_frs = baseline_frs.sel(trials=trials.query('context_name=="aud"').index)

        pval = st.mannwhitneyu(vis_baseline_frs.values, aud_baseline_frs.values,nan_policy='omit')[1]
        stim_context_modulation['baseline_context_modulation_p_value'].append(pval)

        vis_baseline_frs = vis_baseline_frs.mean(skipna=True).values
        aud_baseline_frs = aud_baseline_frs.mean(skipna=True).values

        baseline_modulation_zscore=(vis_baseline_frs-aud_baseline_frs)/baseline_frs.std(skipna=True)
        stim_context_modulation['baseline_context_modulation_zscore'].append(baseline_modulation_zscore.values)

        baseline_modulation_index=(vis_baseline_frs-aud_baseline_frs)/(vis_baseline_frs+aud_baseline_frs)
        stim_context_modulation['baseline_context_modulation_index'].append(baseline_modulation_index)

        baseline_mod_sign=np.sign(np.mean(vis_baseline_frs-aud_baseline_frs))
        stim_context_modulation['baseline_context_modulation_sign'].append(baseline_mod_sign)

        #auc for baseline frs
        binary_label=trials['context_name']=='vis'
        baseline_context_auc=roc_auc_score(binary_label,baseline_frs.values)
        stim_context_modulation['baseline_context_roc_auc'].append(baseline_context_auc)

        #linear shifted baseline context
        temp_baseline_context_diff=[]
        for sh,shift in enumerate(shifts):
            labels = middle_4_block_trials['context_name'].values
            input_data = baseline_frs.sel(trials=middle_4_blocks+shift).values
            temp_baseline_context_diff.append(input_data[labels=='vis'].mean()-input_data[labels=='aud'].mean())
        temp_baseline_context_diff=np.asarray(temp_baseline_context_diff)
        
        true_value = temp_baseline_context_diff[shifts==1]
        null_median = np.median(temp_baseline_context_diff[shifts!=1])
        null_mean = np.mean(temp_baseline_context_diff[shifts!=1])
        null_std = np.std(temp_baseline_context_diff[shifts!=1])
        pval_higher = np.mean(temp_baseline_context_diff[shifts!=1]>=true_value)
        pval_lower = np.mean(temp_baseline_context_diff[shifts!=1]<=true_value)

        stim_context_modulation['linear_shift_baseline_context_true_value'].append(true_value)
        stim_context_modulation['linear_shift_baseline_context_null_median'].append(null_median)
        stim_context_modulation['linear_shift_baseline_context_null_mean'].append(null_mean)
        stim_context_modulation['linear_shift_baseline_context_null_std'].append(null_std)
        stim_context_modulation['linear_shift_baseline_context_p_value_higher'].append(pval_higher)
        stim_context_modulation['linear_shift_baseline_context_p_value_lower'].append(pval_lower)


        #cross stimulus discrimination
        #vis1 vs. vis2
        vis1_trials = trials.query('stim_name=="vis1"')
        vis2_trials = trials.query('stim_name=="vis2"')
        vis1_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=vis1_trials.index).mean(dim='time',skipna=True)
        vis2_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=vis2_trials.index).mean(dim='time',skipna=True)
        vis1_and_vis2_frs=np.concatenate([vis1_frs_by_trial.values,vis2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis1_frs_by_trial)),np.zeros(len(vis2_frs_by_trial))])
        vis_discrim_auc=roc_auc_score(binary_label,vis1_and_vis2_frs)
        stim_context_modulation['vis_discrim_roc_auc'].append(vis_discrim_auc)

        #aud1 vs. aud2
        aud1_trials = trials.query('stim_name=="sound1"')
        aud2_trials = trials.query('stim_name=="sound2"')
        aud1_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=aud1_trials.index).mean(dim='time',skipna=True)
        aud2_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=aud2_trials.index).mean(dim='time',skipna=True)
        aud1_and_aud2_frs=np.concatenate([aud1_frs_by_trial.values,aud2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(aud1_frs_by_trial)),np.zeros(len(aud2_frs_by_trial))])
        aud_discrim_auc=roc_auc_score(binary_label,aud1_and_aud2_frs)
        stim_context_modulation['aud_discrim_roc_auc'].append(aud_discrim_auc)

        #targets: vis1 vs sound1
        vis1_vs_aud1_frs=np.concatenate([vis1_frs_by_trial.values,aud1_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis1_frs_by_trial)),np.zeros(len(aud1_frs_by_trial))])
        target_discrim_auc=roc_auc_score(binary_label,vis1_vs_aud1_frs)
        stim_context_modulation['target_discrim_roc_auc'].append(target_discrim_auc)

        #nontargets: vis2 vs sound2
        vis2_vs_aud2_frs=np.concatenate([vis2_frs_by_trial.values,aud2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis2_frs_by_trial)),np.zeros(len(aud2_frs_by_trial))])
        nontarget_discrim_auc=roc_auc_score(binary_label,vis2_vs_aud2_frs)
        stim_context_modulation['nontarget_discrim_roc_auc'].append(nontarget_discrim_auc)

        #vis vs. aud
        vis_and_aud_frs=np.concatenate([vis1_frs_by_trial.values,vis2_frs_by_trial.values,
                                        aud1_frs_by_trial.values,aud2_frs_by_trial.values])
        binary_label=np.concatenate([np.ones(len(vis1_frs_by_trial)+len(vis2_frs_by_trial)),
                                    np.zeros(len(aud1_frs_by_trial)+len(aud2_frs_by_trial))])
        vis_vs_aud_auc=roc_auc_score(binary_label,vis_and_aud_frs)
        stim_context_modulation['vis_vs_aud'].append(vis_vs_aud_auc)

        #HIT/CR/FA - currently only makes sense for DR experiments
        behav_time_windows_start=[0,0.1,0.2]
        behav_time_windows_end=[0.1,0.2,0.3]
        behav_time_window_labels=['early','mid','late']
        if 'DynamicRouting' in session_info.project:
            cr_trials=trials.query('is_response==False and is_correct==True and is_target==True')
            fa_trials=trials.query('is_response==True and is_correct==False and is_target==True')
            hit_trials=trials.query('is_response==True and is_correct==True and is_target==True')

            
            for tw,time_window in enumerate(behav_time_window_labels):
                cr_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(behav_time_windows_start[tw],behav_time_windows_end[tw]),trials=cr_trials.index).mean(dim='time',skipna=True)
                fa_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(behav_time_windows_start[tw],behav_time_windows_end[tw]),trials=fa_trials.index).mean(dim='time',skipna=True)
                hit_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(behav_time_windows_start[tw],behav_time_windows_end[tw]),trials=hit_trials.index).mean(dim='time',skipna=True)

                #cr vs. fa
                cr_and_fa_frs=np.concatenate([cr_frs_by_trial.values,fa_frs_by_trial.values])
                binary_label=np.concatenate([np.ones(len(cr_frs_by_trial)),np.zeros(len(fa_frs_by_trial))])
                cr_vs_fa_auc=roc_auc_score(binary_label,cr_and_fa_frs)
                stim_context_modulation['cr_vs_fa_'+time_window+'_roc_auc'].append(cr_vs_fa_auc)

                #hit vs. cr
                hit_and_cr_frs=np.concatenate([hit_frs_by_trial.values,cr_frs_by_trial.values])
                binary_label=np.concatenate([np.ones(len(hit_frs_by_trial)),np.zeros(len(cr_frs_by_trial))])
                hit_vs_cr_auc=roc_auc_score(binary_label,hit_and_cr_frs)
                stim_context_modulation['hit_vs_cr_'+time_window+'_roc_auc'].append(hit_vs_cr_auc)

                #hit vs. fa
                hit_and_fa_frs=np.concatenate([hit_frs_by_trial.values,fa_frs_by_trial.values])
                binary_label=np.concatenate([np.ones(len(hit_frs_by_trial)),np.zeros(len(fa_frs_by_trial))])
                hit_vs_fa_auc=roc_auc_score(binary_label,hit_and_fa_frs)
                stim_context_modulation['hit_vs_fa_'+time_window+'_roc_auc'].append(hit_vs_fa_auc)
        else:
            for tw,time_window in enumerate(behav_time_window_labels):
                stim_context_modulation['cr_vs_fa_'+time_window+'_roc_auc'].append(np.nan)
                stim_context_modulation['hit_vs_cr_'+time_window+'_roc_auc'].append(np.nan)
                stim_context_modulation['hit_vs_fa_'+time_window+'_roc_auc'].append(np.nan)

        
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
            stim_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=stim_trials.index).mean(dim='time',skipna=True)
            stim_baseline_frs_by_trial = baseline_frs.sel(trials=stim_trials.index)
            stim_frs_by_trial_zscore = (stim_frs_by_trial-stim_baseline_frs_by_trial.mean(skipna=True))/stim_baseline_frs_by_trial.std(skipna=True)
            stim_context_modulation[ss+'_stimulus_modulation_zscore'].append(stim_frs_by_trial_zscore.mean(skipna=True).values)
            stimulus_modulation_index=(stim_frs_by_trial-stim_baseline_frs_by_trial).mean(skipna=True)/(stim_frs_by_trial+stim_baseline_frs_by_trial).mean(skipna=True)
            stim_context_modulation[ss+'_stimulus_modulation_index'].append(stimulus_modulation_index.values)

            pval = st.wilcoxon(stim_frs_by_trial.values, stim_baseline_frs_by_trial.values,nan_policy='omit',zero_method='zsplit')[1]
            stim_context_modulation[ss+'_stimulus_modulation_p_value'].append(pval)
            stim_mod_sign=np.sign(np.mean(stim_frs_by_trial.values-stim_baseline_frs_by_trial.values))
            stim_context_modulation[ss+'_stimulus_modulation_sign'].append(stim_mod_sign)
            #auc for stimulus frs
            stim_and_baseline_frs=np.concatenate([stim_frs_by_trial.values,stim_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(stim_frs_by_trial)),np.zeros(len(stim_baseline_frs_by_trial))])
            stim_context_auc=roc_auc_score(binary_label,stim_and_baseline_frs)
            stim_context_modulation[ss+'_stimulus_modulation_roc_auc'].append(stim_context_auc)

            #stimulus late modulation
            stim_late_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.1,0.2),trials=stim_trials.index).mean(dim='time',skipna=True)
            stim_late_frs_by_trial_zscore = (stim_late_frs_by_trial-stim_baseline_frs_by_trial.mean(skipna=True))/stim_baseline_frs_by_trial.std(skipna=True)
            stim_context_modulation[ss+'_stimulus_late_modulation_zscore'].append(stim_late_frs_by_trial_zscore.mean(skipna=True).values)
            stimulus_late_modulation_index=(stim_late_frs_by_trial-stim_baseline_frs_by_trial).mean(skipna=True)/(stim_late_frs_by_trial+stim_baseline_frs_by_trial).mean(skipna=True)
            stim_context_modulation[ss+'_stimulus_late_modulation_index'].append(stimulus_late_modulation_index.values)

            pval = st.wilcoxon(stim_late_frs_by_trial.values, stim_baseline_frs_by_trial.values,nan_policy='omit',zero_method='zsplit')[1]
            stim_context_modulation[ss+'_stimulus_late_modulation_p_value'].append(pval)
            stim_late_mod_sign=np.sign(np.mean(stim_late_frs_by_trial.values-stim_baseline_frs_by_trial.values))
            stim_context_modulation[ss+'_stimulus_late_modulation_sign'].append(stim_late_mod_sign)
            #auc for stimulus late frs
            stim_late_and_baseline_frs=np.concatenate([stim_late_frs_by_trial.values,stim_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(stim_late_frs_by_trial)),np.zeros(len(stim_baseline_frs_by_trial))])
            stim_late_context_auc=roc_auc_score(binary_label,stim_late_and_baseline_frs)
            stim_context_modulation[ss+'_stimulus_late_modulation_roc_auc'].append(stim_late_context_auc)

            #latency
            stim_latency = np.abs(trial_da).sel(unit_id=unit['unit_id'],time=slice(0,0.3),trials=stim_trials.index).mean(dim='trials',skipna=True).idxmax(dim='time').values
            stim_context_modulation[ss+'_stim_latency'].append(stim_latency)

            #find stim trials in same vs. other context
            same_context_trials = trials.query('context_name==@same_context and stim_name==@ss')
            other_context_trials = trials.query('context_name==@other_context and stim_name==@ss')

            same_context_baseline_frs = baseline_frs.sel(trials=same_context_trials.index)
            other_context_baseline_frs = baseline_frs.sel(trials=other_context_trials.index)

            #find raw frs during stim (first 100ms)
            same_context_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=same_context_trials.index).mean(dim='time',skipna=True)
            other_context_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=other_context_trials.index).mean(dim='time',skipna=True)

            pval = st.mannwhitneyu(same_context_frs_by_trial.values, other_context_frs_by_trial.values,nan_policy='omit')[1]
            stim_context_modulation[ss+'_context_modulation_p_value'].append(pval)

            same_context_frs = same_context_frs_by_trial.mean(skipna=True).values
            other_context_frs = other_context_frs_by_trial.mean(skipna=True).values

            context_modulation_zscore=((same_context_frs-other_context_frs))/stim_baseline_frs_by_trial.std(skipna=True)
            stim_context_modulation[ss+'_context_modulation_zscore'].append(context_modulation_zscore.values)

            # stim context modulation sign
            context_mod_sign=np.sign(np.mean(same_context_frs-other_context_frs))
            stim_context_modulation[ss+'_context_modulation_sign'].append(context_mod_sign)

            # stim context modulation auc
            binary_label=np.concatenate([np.ones(len(same_context_frs_by_trial.values)),np.zeros(len(other_context_frs_by_trial.values))])
            stim_context_auc=roc_auc_score(binary_label,np.concatenate([same_context_frs_by_trial.values,other_context_frs_by_trial.values]))
            stim_context_modulation[ss+'_context_modulation_roc_auc'].append(stim_context_auc)

            #find evoked frs during stim (first 100ms)
            same_context_evoked_frs_by_trial = (trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=same_context_trials.index).mean(dim=['time'],skipna=True)-same_context_baseline_frs)
            other_context_evoked_frs_by_trial = (trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=other_context_trials.index).mean(dim=['time'],skipna=True)-other_context_baseline_frs)

            pval = st.mannwhitneyu(same_context_evoked_frs_by_trial.values, other_context_evoked_frs_by_trial.values,nan_policy='omit')[1]
            stim_context_modulation[ss+'_evoked_context_modulation_p_value'].append(pval)

            same_context_evoked_frs = same_context_evoked_frs_by_trial.mean(skipna=True).values
            other_context_evoked_frs = other_context_evoked_frs_by_trial.mean(skipna=True).values

            context_modulation_evoked_zscore=((same_context_evoked_frs-other_context_evoked_frs))/stim_baseline_frs_by_trial.std(skipna=True)
            stim_context_modulation[ss+'_evoked_context_modulation_zscore'].append(context_modulation_evoked_zscore.values)

            # evoked stim context modulation sign
            context_mod_evoked_sign=np.sign(np.mean(same_context_evoked_frs-other_context_evoked_frs))
            stim_context_modulation[ss+'_evoked_context_modulation_sign'].append(context_mod_evoked_sign)
            
            #negative numbers can make index behave weirdly, so subtract the minimum from both
            if same_context_evoked_frs<0 or other_context_evoked_frs<0:
                same_context_evoked_frs = same_context_evoked_frs - np.min([same_context_evoked_frs,other_context_evoked_frs])
                other_context_evoked_frs = other_context_evoked_frs - np.min([same_context_evoked_frs,other_context_evoked_frs])

            #compute metrics
            raw_fr_metric=(same_context_frs-other_context_frs)/(same_context_frs+other_context_frs)
            stim_context_modulation[ss+'_context_modulation_index'].append(raw_fr_metric)

            evoked_fr_metric=(same_context_evoked_frs-other_context_evoked_frs)/(same_context_evoked_frs+other_context_evoked_frs)
            stim_context_modulation[ss+'_evoked_context_modulation_index'].append(evoked_fr_metric)

    if 'session_id' in units.columns:
        unit_metric_merge=units.reset_index(drop=True).merge(pd.DataFrame(stim_context_modulation),on=['unit_id','session_id'])
    else:
        unit_metric_merge=units.reset_index(drop=True).merge(pd.DataFrame(stim_context_modulation),on=['unit_id'])
    unit_metric_merge.drop(columns=['spike_times','waveform_sd','waveform_mean'], inplace=True, errors='ignore')
    unit_metric_merge.to_pickle(os.path.join(save_path,session_info.id+'_stim_context_modulation.pkl'))


#calculate metrics for channel alignment

def compute_metrics_for_alignment(trials, units, session_info, save_path):



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
        contexts=trials['context_name'].unique()

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
            trials['true_context_name']=trials['context_name']

            for block in range(0,6):
                block_start_time=start_time+block*10*60
                block_end_time=start_time+(block+1)*10*60
                block_trials=trials.query('start_time>=@block_start_time').index
                fake_context[block_trials]=block_contexts[block]
                fake_block_nums[block_trials]=block
            
            trials['context_name']=fake_context
            trials['block_index']=fake_block_nums
            trials['is_vis_context']=trials['context_name']=='vis'
            trials['is_aud_context']=trials['context_name']=='aud'

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
                lick_trials = trials.query('(stim_name=="vis1" and context_name=="aud" and is_response==True) or \
                                        (stim_name=="sound1" and context_name=="vis" and is_response==True)')
                non_lick_trials = trials.query('(stim_name=="vis1" and context_name=="aud" and is_response==False) or \
                                                (stim_name=="sound1" and context_name=="vis" and is_response==False)')
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

