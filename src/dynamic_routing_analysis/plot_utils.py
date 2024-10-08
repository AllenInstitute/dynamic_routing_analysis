import os
import pickle

import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xarray as xr
from matplotlib import patches

from dynamic_routing_analysis import spike_utils


def plot_unit_by_id(sel_unit,save_path=None,show_metric=None):

    unit_df = ds.dataset(npc_lims.get_cache_path('units',session_id=sel_unit[:17],version='0.0.214')
                         ).to_table(filter=(ds.field('unit_id') == sel_unit)).to_pandas()
    session_id=str(unit_df['subject_id'].values[0])+'_'+str(unit_df['date'].values[0])

    trials=pd.read_parquet(
                npc_lims.get_cache_path('trials',session_id,version='any')
            )

    time_before = 0.5
    time_after = 1.0
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(unit_df, trials, time_before, time_after, binsize)

    ##plot PSTH with context differences -- subplot for each stimulus
    fig,ax=plt.subplots(2,2,sharex=True,sharey=True)
    ax=ax.flatten()

    stims=['vis1','vis2','sound1','sound2']

    for st,stim in enumerate(stims):

        stim_trials=trials[:].query('stim_name==@stim')

        vis_context_spikes=trial_da.sel(
            trials=stim_trials.query('is_vis_context').index,
            unit_id=sel_unit,).mean(dim=['trials'])

        aud_context_spikes=trial_da.sel(
            trials=stim_trials.query('is_aud_context').index,
            unit_id=sel_unit,).mean(dim=['trials'])

        ax[st].plot(vis_context_spikes.time, vis_context_spikes.values, label='vis context',color='tab:green')
        ax[st].plot(aud_context_spikes.time, aud_context_spikes.values, label='aud context',color='tab:blue')
        ax[st].axvline(0, color='k', linestyle='--',alpha=0.5)
        ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[st].set_title(stim)
        ax[st].legend()
        ax[st].set_xlim([-0.5,1.0])

        if st>1:
            ax[st].set_xlabel('time (s)')
        if st==0 or st==2:
            ax[st].set_ylabel('spikes/s')

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_context_modulation.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


    #plot lick vs. not lick
    fig,ax=plt.subplots(1,2,figsize=(6.4,3),sharex=True,sharey=True)
    ax=ax.flatten()
    stims=['vis1','sound1']
    for st,stim in enumerate(stims):
        
        if stim=='vis1':
            sel_context='aud'
        elif stim=='sound1':
            sel_context='vis'
        stim_trials=trials[:].query('stim_name==@stim and context_name==@sel_context')

        lick_spikes=trial_da.sel(
            trials=stim_trials.query('is_response').index,
            unit_id=sel_unit,).mean(dim=['trials'])

        non_lick_spikes=trial_da.sel(
            trials=stim_trials.query('not is_response').index,
            unit_id=sel_unit,).mean(dim=['trials'])

        ax[st].plot(lick_spikes.time, lick_spikes.values, label='licks',color='tab:red')
        ax[st].plot(non_lick_spikes.time, non_lick_spikes.values, label='non licks',color='tab:purple')
        ax[st].axvline(0, color='k', linestyle='--',alpha=0.5)
        ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[st].set_title(stim+'; '+sel_context+' context')
        ax[st].legend()
        ax[st].set_xlim([-0.5,1.0])

        if st>1:
            ax[st].set_xlabel('time (s)')
        if st==0 or st==2:
            ax[st].set_ylabel('spikes/s')

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_lick_modulation.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


def plot_stim_response_by_unit_id(sel_unit,save_path=None,show_metric=None):
    unit_df = ds.dataset(npc_lims.get_cache_path('units',session_id=sel_unit[:17],version='any')
                         ).to_table(filter=(ds.field('unit_id') == sel_unit)).to_pandas()
    session_id=str(unit_df['subject_id'].values[0])+'_'+str(unit_df['date'].values[0])

    trials=pd.read_parquet(
                npc_lims.get_cache_path('trials',session_id,version='any')
            )

    ##plot rasters - target only
    fig,ax=plt.subplots(1,2,figsize=(8,5),sharex=True,sharey=True)
    ax=ax.flatten()

    stims=['vis1','sound1']
    stim_names=['vis','aud']

    for st,stim in enumerate(stims):

        stim_trials=trials[:].query('stim_name==@stim')
        ax[st].axvline(0, color='k', linestyle='-',alpha=0.5)
        ax[st].set_title(stim_names[st])
        ax[st].set_xlim([-0.2,0.5])
        ax[st].set_xlabel('time (s)')
        if st==0:
            ax[st].set_ylabel('trials')

        for it,(tt,trial) in enumerate(stim_trials.iterrows()):
            stim_start=trial['stim_start_time']
            spikes=unit_df['spike_times'].iloc[0]-stim_start
            
            trial_spike_inds=(spikes>=-0.2)&(spikes<=0.5)
            if sum(trial_spike_inds)==0:
                continue
            else:
                spikes=spikes[trial_spike_inds]

            ax[st].vlines(spikes,ymin=it,ymax=it+1,linewidth=0.75,color='k')

            

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_target_stim_rasters.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


    time_before = 0.2
    time_after = 0.5
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(unit_df, trials, time_before, time_after, binsize)

    ##plot PSTHs - target only
    fig,ax=plt.subplots(1,2,figsize=(8,2.5),sharex=True,sharey=True)
    ax=ax.flatten()

    stims=['vis1','sound1']
    stim_names=['vis','aud']

    for st,stim in enumerate(stims):

        stim_trials=trials[:].query('stim_name==@stim').index.values

        spikes=trial_da.sel(
            unit_id=sel_unit,
            trials=stim_trials).mean(dim=['trials'])

        ax[st].plot(spikes.time, spikes.values, label=stim)
        ax[st].axvline(0, color='k', linestyle='--',alpha=0.5)
        # ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[st].set_title(stim_names[st])
        ax[st].legend()
        ax[st].set_xlim([-0.2,0.5])
        ax[st].set_xlabel('time (s)')

        if st>1:
            ax[st].set_xlabel('time (s)')
        if st==0 or st==2:
            ax[st].set_ylabel('spikes/s')

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_target_stim_PSTHs.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


def plot_motor_response_by_unit_id(sel_unit,save_path=None,show_metric=None):
    unit_df = ds.dataset(npc_lims.get_cache_path('units',session_id=sel_unit[:17],version='any')
                         ).to_table(filter=(ds.field('unit_id') == sel_unit)).to_pandas()
    session_id=str(unit_df['subject_id'].values[0])+'_'+str(unit_df['date'].values[0])

    trials=pd.read_parquet(
                npc_lims.get_cache_path('trials',session_id,version='any')
            )

    ##plot rasters - vis stim, aud context
    fig,ax=plt.subplots(1,2,figsize=(8,5),sharex=True)
    ax=ax.flatten()

    responses=[True,False]
    response_names=['lick','no lick']
    stim='vis1'
    
    for rr,resp in enumerate(responses):

        if stim=='vis1':
            sel_context='aud'
        elif stim=='sound1':
            sel_context='vis'
        resp_trials=trials[:].query('stim_name==@stim and context_name==@sel_context and is_response==@resp')
        ax[rr].axvline(0, color='k', linestyle='-',alpha=0.5)
        ax[rr].set_title(response_names[rr])
        ax[rr].set_xlim([-0.2,0.5])
        ax[rr].set_xlabel('time (s)')
        if rr==0:
            ax[rr].set_ylabel('trials')

        for it,(tt,trial) in enumerate(resp_trials.iterrows()):
            stim_start=trial['stim_start_time']
            spikes=unit_df['spike_times'].iloc[0]-stim_start
            
            trial_spike_inds=(spikes>=-0.2)&(spikes<=0.5)
            if sum(trial_spike_inds)==0:
                continue
            else:
                spikes=spikes[trial_spike_inds]

            ax[rr].vlines(spikes,ymin=it,ymax=it+1,linewidth=0.75,color='k')

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_lick_rasters.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


    #PSTHs
    time_before = 0.2
    time_after = 0.5
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(unit_df, trials, time_before, time_after, binsize)

    ##plot PSTHs - target only
    fig,ax=plt.subplots(1,2,figsize=(8,2.5),sharex=True,sharey=True)
    ax=ax.flatten()

    responses=[True,False]
    response_names=['lick','no lick']
    # stim='vis1'

    for rr,resp in enumerate(responses):

        if stim=='vis1':
            sel_context='aud'
        elif stim=='sound1':
            sel_context='vis'
        resp_trials=trials[:].query('stim_name==@stim and context_name==@sel_context and is_response==@resp').index.values

        spikes=trial_da.sel(
            unit_id=sel_unit,
            trials=resp_trials).mean(dim=['trials'])

        ax[rr].plot(spikes.time, spikes.values, label=stim)
        ax[rr].axvline(0, color='k', linestyle='--',alpha=0.5)
        # ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[rr].set_title(response_names[rr])
        ax[rr].legend()
        ax[rr].set_xlim([-0.2,0.5])
        ax[rr].set_xlabel('time (s)')

        if rr>1:
            ax[rr].set_xlabel('time (s)')
        if rr==0:
            ax[rr].set_ylabel('spikes/s')


    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_lick_PSTHs.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()
        
def plot_context_offset_by_unit_id(sel_unit,save_path=None,show_metric=None):
    unit_df = ds.dataset(npc_lims.get_cache_path('units',session_id=sel_unit[:17],version='any')
                         ).to_table(filter=(ds.field('unit_id') == sel_unit)).to_pandas()
    session_id=str(unit_df['subject_id'].values[0])+'_'+str(unit_df['date'].values[0])

    trials=pd.read_parquet(
                npc_lims.get_cache_path('trials',session_id,version='any')
            )

    ##plot rasters - target only
    fig,ax=plt.subplots(1,2,figsize=(8,5),sharex=True,sharey=True)
    ax=ax.flatten()

    stims=['vis1','sound1']
    stim_names=['vis','aud']

    for st,stim in enumerate(stims):

        stim_trials=trials[:].query('stim_name==@stim').reset_index()

        colors=np.full(len(stim_trials),'k                 ')
        colors[stim_trials.query('is_vis_context').index]='tab:green'
        colors[stim_trials.query('is_aud_context').index]='tab:blue'

        #aud context trials
        aud_context_trials=stim_trials.query('is_aud_context').index.values

        #make patches during aud context trials
        for it,trial in stim_trials.query('is_aud_context').iterrows():
            stim_start=trial['stim_start_time']
            ax[st].add_patch(patches.Rectangle((-0.2, it), 0.7, 0.5, fill=True, color='grey', 
                                               edgecolor=None,alpha=0.1))

        ax[st].axvline(0, color='k', linestyle='-',alpha=0.5)
        ax[st].set_title(stim_names[st])
        ax[st].set_xlim([-0.2,0.5])
        ax[st].set_xlabel('time (s)')
        if st==0:
            ax[st].set_ylabel('trials')

        for it,(tt,trial) in enumerate(stim_trials.iterrows()):
            stim_start=trial['stim_start_time']
            spikes=unit_df['spike_times'].iloc[0]-stim_start
            
            trial_spike_inds=(spikes>=-0.2)&(spikes<=0.5)
            if sum(trial_spike_inds)==0:
                continue
            else:
                spikes=spikes[trial_spike_inds]

            # ax[st].vlines(spikes,ymin=it,ymax=it+1,linewidth=0.75,color='k')
            ax[st].vlines(spikes,ymin=it,ymax=it+1,linewidth=0.75,color=colors[it])

            

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_context_diff_rasters.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


    time_before = 0.2
    time_after = 0.5
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(unit_df, trials, time_before, time_after, binsize)

    ##plot PSTHs - target only
    fig,ax=plt.subplots(1,2,figsize=(8,2.5),sharex=True,sharey=True)
    ax=ax.flatten()

    stims=['vis1','sound1']
    stim_names=['vis','aud']

    for st,stim in enumerate(stims):

        stim_trials=trials[:].query('stim_name==@stim')

        vis_context_spikes=trial_da.sel(
            trials=stim_trials.query('is_vis_context').index,
            unit_id=sel_unit,).mean(dim=['trials'])

        aud_context_spikes=trial_da.sel(
            trials=stim_trials.query('is_aud_context').index,
            unit_id=sel_unit,).mean(dim=['trials'])

        ax[st].plot(vis_context_spikes.time, vis_context_spikes.values, label='vis context',color='tab:green')
        ax[st].plot(aud_context_spikes.time, aud_context_spikes.values, label='aud context',color='tab:blue')
        ax[st].axvline(0, color='k', linestyle='--',alpha=0.5)
        ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[st].set_title(stim)
        if st==1:
            ax[st].legend()
        ax[st].set_xlim([-0.2,0.5])

        if st>1:
            ax[st].set_xlabel('time (s)')
        if st==0 or st==2:
            ax[st].set_ylabel('spikes/s')

    if show_metric is not None:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0]+'; '+show_metric)
    else:
        fig.suptitle('unit '+unit_df['unit_id'].values[0]+'; '+unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path,unit_df['unit_id'].values[0]+'_context_diff_PSTHs.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


def plot_unit_response_by_task_performance(sel_unit, save_path=None):
    unit_df = ds.dataset(npc_lims.get_cache_path('units')).to_table(
        filter=(ds.field('unit_id') == sel_unit)).to_pandas()
    session_id = str(unit_df['subject_id'].values[0]) + '_' + str(unit_df['date'].values[0])

    trials = pd.read_parquet(
        npc_lims.get_cache_path('trials', session_id, version='any')
    )

    time_before = 0.5
    time_after = 1
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(unit_df, trials, time_before, time_after, binsize)

    ## plot PSTH with context differences -- subplot for each performance
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (8, 8))

    contexts = ['vis', 'aud']
    cols = ['tab:green', 'tab:blue', 'tab:red']
    # target - hits
    for c, context in enumerate(contexts):
        stim_trials = trials[(trials.is_hit) & (trials['is_' + context + '_context'])]
        context_spikes = trial_da.sel(
            trials=stim_trials.index,
            unit_id=sel_unit, ).mean(dim=['trials'])

        rate = np.round(len(trials[(trials.is_hit) & (trials['is_' + context + '_context'])])
                        / len(trials[(trials.is_go) & (trials['is_' + context + '_context'])]), 2)
        ax[0, c].plot(context_spikes.time, context_spikes.values, label= 'hits, rate:' + str(rate), color=cols[0])
        ax[0, c].set_title(context + ' context')

    conditions_set = ['correct_reject', 'false_alarm']
    for c1, context in enumerate(contexts):
        for c2, condition in enumerate(conditions_set):
            # target
            stim_trials = trials[(trials['is_' + condition]) & (trials['is_' + context + '_context']) & (trials.is_target)]
            target_spikes = trial_da.sel(
                trials=stim_trials.index,
                unit_id=sel_unit, ).mean(dim=['trials'])

            rate = np.round(len(trials[(trials['is_' + condition]) & (trials['is_' + context + '_context']) & (trials.is_target)])
                        / len(trials[(trials.is_nogo) & (trials['is_' + context + '_context']) & (trials.is_target)]), 2)
            ax[0, c1].plot(target_spikes.time, target_spikes.values, label= condition + ', rate: ' + str(rate), color=cols[c2 + 1])

            # non-target
            stim_trials = trials[
                (trials['is_' + condition]) & (trials['is_' + context + '_context']) & (trials.is_nontarget)]
            target_spikes = trial_da.sel(
                trials=stim_trials.index,
                unit_id=sel_unit, ).mean(dim=['trials'])

            rate = np.round(
                len(trials[(trials['is_' + condition]) & (trials['is_' + context + '_context']) & (trials.is_nontarget)])
                / len(trials[(trials.is_nogo) & (trials['is_' + context + '_context']) & (trials.is_nontarget)]), 2)
            ax[1, c1].plot(target_spikes.time, target_spikes.values, label= condition + ', rate: ' + str(rate),
                           color=cols[c2+1])

    ax = ax.flatten()
    for c in range(4):
        ax[c].axvline(0, color='k', linestyle='--', alpha=0.5)
        ax[c].axvline(0.5, color='k', linestyle='--', alpha=0.5)
        ax[c].legend()
        ax[c].set_xlim([-0.5, 1.0])

        if c > 1:
            ax[c].set_xlabel('time (s)')
        if c == 0:
            ax[c].set_ylabel('Target - spikes/s')
        if c == 2:
            ax[c].set_ylabel('Non-target - spikes/s')

    fig.suptitle('unit ' + unit_df['unit_id'].values[0] + '; ' + unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, unit_df['unit_id'].values[0] + '_performance.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()


def plot_unit_response_by_task_performance_stim_aligned(sel_unit, save_path=None):
    unit_df = ds.dataset(npc_lims.get_cache_path('units')).to_table(
        filter=(ds.field('unit_id') == sel_unit)).to_pandas()
    session_id = str(unit_df['subject_id'].values[0]) + '_' + str(unit_df['date'].values[0])

    trials = pd.read_parquet(
        npc_lims.get_cache_path('trials', session_id, version='any')
    )

    time_before = 0.5
    time_after = 1
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(unit_df, trials, time_before, time_after, binsize)

    ## plot PSTH with context differences -- subplot for each performance
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(4, 6))

    stims = ['vis', 'aud']
    cols = ['tab:green', 'tab:blue', 'tab:red']
    # target - hits
    for s, stim in enumerate(stims):
        stim_trials = trials[(trials.is_hit) & (trials['is_' + stim + '_target'])]
        stim_spikes = trial_da.sel(
            trials=stim_trials.index,
            unit_id=sel_unit, ).mean(dim=['trials'])

        rate = np.round(len(trials[(trials.is_hit) & (trials['is_' + stim + '_target'])])
                        / len(trials[(trials.is_go) & (trials['is_' + stim + '_target'])]), 2)
        ax[s].plot(stim_spikes.time, stim_spikes.values, label='hits, rate:' + str(rate), color=cols[0])
        ax[s].set_title(stim + ' stimulus')

    conditions_set = ['correct_reject', 'false_alarm']
    for s, stim in enumerate(stims):
        for c, condition in enumerate(conditions_set):
            # target
            stim_trials = trials[(trials['is_' + condition]) & (trials['is_' + stim + '_target'])]
            target_spikes = trial_da.sel(
                trials=stim_trials.index,
                unit_id=sel_unit, ).mean(dim=['trials'])

            rate = np.round(len(trials[(trials['is_' + condition]) & (trials['is_' + stim + '_target'])])
                            / len(trials[(trials.is_nogo) & (trials['is_' + stim + '_target'])]), 2)
            ax[s].plot(target_spikes.time, target_spikes.values, label=condition + ', rate: ' + str(rate),
                       color=cols[c + 1])

    ax = ax.flatten()
    for s in range(2):
        ax[s].axvline(0, color='k', linestyle='--', alpha=0.5)
        ax[s].axvline(0.5, color='k', linestyle='--', alpha=0.5)
        ax[s].legend()
        ax[s].set_xlim([-0.5, 1.0])
        ax[s].set_ylabel('Target - spikes/s')
    ax[1].set_xlabel('time (s)')
    fig.suptitle('unit ' + unit_df['unit_id'].values[0] + '; ' + unit_df['structure'].values[0])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, unit_df['unit_id'].values[0] + '_performance_stim_target.png'),
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)
        plt.close()

def plot_stimulus_modulation_pie_chart(adj_pvals,sel_project,savepath=None):
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

    #any stim
    # any_stim_resp=adj_pvals.query('vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05')
    any_stim_resp=adj_pvals.query('any_stim<0.05')
    stim_and_context=adj_pvals.query('any_stim<0.05 and context<0.05')

    #catch
    catch_stim_resp=adj_pvals.query('catch<0.05')

    #none
    no_stim_resp=adj_pvals.query('vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and catch>=0.05')

    #stimulus responses
    labels=['vis1 only','vis2 only','both vis',
            'sound1 only','sound2 only','both sound',
            'mixed','none','catch']
    sizes=[len(vis1_stim_resp),len(vis2_stim_resp),len(both_vis_stim_resp),
            len(sound1_stim_resp),len(sound2_stim_resp),len(both_sound_stim_resp),
            len(mixed_stim_resp),len(no_stim_resp),len(catch_stim_resp)]

    fig,ax=plt.subplots()
    ax.pie(sizes,labels=labels,autopct='%1.1f%%')
    ax.set_title('n = '+str(len(adj_pvals))+' units')
    fig.suptitle('stimulus responsiveness')
    fig.tight_layout()

    if savepath is not None:
        if 'Templeton' in sel_project:
            temp_savepath=os.path.join(savepath,"stimulus_responsiveness_Templeton.png")
        else:
            temp_savepath=os.path.join(savepath,"stimulus_responsiveness_DR.png")   

        fig.savefig(temp_savepath,
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)  
        


def plot_context_stim_lick_modulation_pie_chart(adj_pvals,sel_project,savepath=None):

    #lick modulation only
    lick_resp=adj_pvals.query('lick<0.05 and context>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05')

    #lick and context
    lick_and_context_resp=adj_pvals.query('context<0.05 and lick<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05')

    #lick and stimulus
    lick_and_stim_resp=adj_pvals.query('lick<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and context>=0.05')

    #all three
    all_resp=adj_pvals.query('context<0.05 and lick<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05)')

    #stimulus modulation only
    only_stim_resp=adj_pvals.query('(vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and context>=0.05 and lick>=0.05')

    #context modulation only
    context_resp=adj_pvals.query('context<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05')

    #stim and context modulation
    stim_and_context_resp=adj_pvals.query('context<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and lick>=0.05')

    neither_stim_nor_context_resp=adj_pvals.query('context>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05')  


    labels=['stimulus only','stimulus and context','context only',
            'context and lick','lick only', 'lick & stimulus & context',
            'lick and stimulus',  'none']
    sizes=[len(only_stim_resp),len(stim_and_context_resp),len(context_resp),
            len(lick_and_context_resp),len(lick_resp),len(all_resp),
            len(lick_and_stim_resp), len(neither_stim_nor_context_resp)]

    fig,ax=plt.subplots()
    ax.pie(sizes,labels=labels,autopct='%1.1f%%',
        colors=['tab:blue', 'tab:orange', 'tab:green',
                'tab:red' , 'tab:purple', 'tab:brown', 
                'tab:pink', 'grey'])
    ax.set_title('n = '+str(len(adj_pvals))+' units')
    fig.suptitle('context, lick, and stim modulation')
    fig.tight_layout()

    if savepath is not None:
        if 'Templeton' in sel_project:
            temp_savepath=os.path.join(savepath,"context_stim_lick_Templeton.png")
        else:
            temp_savepath=os.path.join(savepath,"context_stim_lick_DR.png")   

        fig.savefig(temp_savepath,
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None)  
        

def plot_context_mod_stim_resp_pie_chart(adj_pvals,sel_project,savepath=None):

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
    vis1_context_evoked_stim_mod=adj_pvals.query('vis1_context_evoked<0.05 and vis2_context_evoked>=0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05')
    vis2_context_evoked_stim_mod=adj_pvals.query('vis2_context_evoked<0.05 and vis1_context_evoked>=0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05')
    sound1_context_evoked_stim_mod=adj_pvals.query('sound1_context_evoked<0.05 and sound2_context_evoked>=0.05 and vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and any_stim<0.05')
    sound2_context_evoked_stim_mod=adj_pvals.query('sound2_context_evoked<0.05 and sound1_context_evoked>=0.05 and vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and any_stim<0.05')

    both_vis_context_evoked_stim_mod=adj_pvals.query('vis1_context_evoked<0.05 and vis2_context_evoked<0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05')
    both_aud_context_evoked_stim_mod=adj_pvals.query('sound1_context_evoked<0.05 and sound2_context_evoked<0.05 and vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and any_stim<0.05')
    multi_modal_context_evoked_stim_mod=adj_pvals.query('((vis1_context_evoked<0.05 or vis2_context_evoked<0.05) and (sound1_context_evoked<0.05 or sound2_context_evoked<0.05)) and any_stim<0.05')

    no_context_evoked_stim_mod=adj_pvals.query('vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05')

    labels=['vis1 only','vis2 only','both vis',
            'sound1 only','sound2 only','both sound',
            'mixed','none']
    sizes=[len(vis1_context_stim_mod),len(vis2_context_stim_mod),len(both_vis_context_stim_mod),
            len(sound1_context_stim_mod),len(sound2_context_stim_mod),len(both_aud_context_stim_mod),
            len(multi_modal_context_stim_mod),len(no_context_stim_mod)]

    fig,ax=plt.subplots()
    ax.pie(sizes,labels=labels,autopct='%1.1f%%')
    ax.set_title('n = '+str(len(adj_pvals))+' units')
    # ax.set_title('n = '+str(len(stim_and_context))+' units')

    fig.suptitle('context modulation of stimulus response')
    fig.tight_layout()

    if savepath is not None:
        if 'Templeton' in sel_project:
            temp_savepath=os.path.join(savepath,"context_mod_stim_resp_Templeton.png")
        else:
            temp_savepath=os.path.join(savepath,"context_mod_stim_resp_DR.png")   

        fig.savefig(temp_savepath,
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None) 
        
    #evoked
    labels=['vis1 only','vis2 only','both vis',
            'sound1 only','sound2 only','both sound',
            'mixed','none']
    sizes=[len(vis1_context_evoked_stim_mod),len(vis2_context_evoked_stim_mod),len(both_vis_context_evoked_stim_mod),
            len(sound1_context_evoked_stim_mod),len(sound2_context_evoked_stim_mod),len(both_aud_context_evoked_stim_mod),
            len(multi_modal_context_evoked_stim_mod),len(no_context_evoked_stim_mod)]

    fig,ax=plt.subplots()
    ax.pie(sizes,labels=labels,autopct='%1.1f%%')
    ax.set_title('n = '+str(len(adj_pvals))+' units')

    fig.suptitle('context modulation of evoked stimulus response')
    fig.tight_layout()

    if savepath is not None:
        if 'Templeton' in sel_project:
            temp_savepath=os.path.join(savepath,"evoked_context_mod_stim_resp_Templeton.png")
        else:
            temp_savepath=os.path.join(savepath,"evoked_context_mod_stim_resp_DR.png")   

        fig.savefig(temp_savepath,
                    dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0.1,
                    metadata=None) 
        

def plot_single_session_decoding_results(path):

    use_half_shifts=False

    decoder_results=pickle.load(open(path,'rb'))
    session_id=list(decoder_results.keys())[0]

    shifts=decoder_results[session_id]['shifts']
    areas=decoder_results[session_id]['areas']
    n_repeats=25

    half_neg_shift=np.round(shifts.min()/2)
    half_pos_shift=np.round(shifts.max()/2)

    half_neg_shift_ind=np.where(shifts==half_neg_shift)[0][0]
    half_pos_shift_ind=np.where(shifts==half_pos_shift)[0][0]
    half_shift_inds=np.arange(half_neg_shift_ind,half_pos_shift_ind+1)

    if use_half_shifts==False:
        half_shift_inds=np.arange(len(shifts))

    bal_acc={}
    for aa in areas:
        if aa in decoder_results[session_id]['results']:
            bal_acc[aa]=[]
            for rr in range(n_repeats):
                temp_bal_acc=[]
                for sh in half_shift_inds:
                    if sh in list(decoder_results[session_id]['results'][aa]['shift'][rr].keys()):
                        temp_bal_acc.append(decoder_results[session_id]['results'][aa]['shift'][rr][sh]['balanced_accuracy_test'])
                if len(temp_bal_acc)>0:
                    bal_acc[aa].append(np.array(temp_bal_acc))
            bal_acc[aa]=np.vstack(bal_acc[aa])

    for aa in areas:
        if aa in decoder_results[session_id]['results']:
            mean_acc=np.nanmean(bal_acc[aa],axis=0)
            
            true_acc=mean_acc[shifts[half_shift_inds]==1]
            pval=np.round(np.nanmean(mean_acc>=true_acc),decimals=4)
            
            fig,ax=plt.subplots(1,1)
            ax.axhline(true_acc,color='k',linestyle='--',alpha=0.5)
            ax.axvline(1,color='k',linestyle='--',alpha=0.5)
            ax.plot(shifts[half_shift_inds],bal_acc[aa].T,alpha=0.5,color='gray')
            ax.plot(shifts[half_shift_inds],mean_acc,color='k',linewidth=2)
            ax.set_xlabel('trial shift')
            ax.set_ylabel('balanced accuracy')
            ax.set_title(str(aa)+' p='+str(pval))
