import os

import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xarray as xr
from matplotlib import patches

from dynamic_routing_analysis import spike_utils


def plot_unit_by_id(sel_unit,save_path=None,show_metric=None):

    unit_df = ds.dataset(npc_lims.get_cache_path('units',version='any')).to_table(filter=(ds.field('unit_id') == sel_unit)).to_pandas()
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

