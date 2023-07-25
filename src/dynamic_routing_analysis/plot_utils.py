'''
plotting utilities for DR pilot data

'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import scipy.signal as sg
import xarray as xr


# functions for binning the spiking data into a convenient shape for plotting
def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count 
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for 
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins


def make_neuron_time_trials_tensor(unit_ids, spike_times, stim_start_time, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a tensor with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        unit_ids: unit_id, i.e. index from units table (same form as session.units table)
        spike_times: spike times corresponding to each unit (spike_times column from units table)
        stim_start_time: the time the stimulus started for each trial
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_tensor: tensor storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
    '''
    neuron_number = len(unit_ids)
    trial_number = len(stim_start_time)
    unit_tensor = np.zeros((neuron_number, int(trial_duration/bin_size), trial_number))
    
    for iu,unit_id in enumerate(unit_ids):
        unit_spike_times = spike_times[unit_id]
        for tt, trial_stim_start in enumerate(stim_start_time):
            unit_tensor[iu, :, tt] = makePSTH(unit_spike_times, 
                                                [trial_stim_start-time_before], 
                                                trial_duration, 
                                                binSize=bin_size)[0]
    return unit_tensor



# make a data array
def make_data_array(unit_ids, spike_times, stim_start_time, time_before_flash = 0.5, trial_duration = 2, bin_size = 0.001):
    '''
    
    '''

    # Make tensor (3-D matrix [units,time,trials])
    trial_tensor = make_neuron_time_trials_tensor(unit_ids, spike_times, stim_start_time, 
                                                  time_before_flash, trial_duration, 
                                                  bin_size)
    # make xarray data array
    trial_da = xr.DataArray(trial_tensor, dims=("unit_id", "time", "trials"), 
                               coords={
                                   "unit_id": unit_ids,
                                   "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                   "trials": stim_start_time.index.values
                                   })
    return trial_da

def find_area_borders(unit_area):
    #area borders
    borders=np.where(unit_area.iloc[:-1].values!=unit_area.iloc[1:].values)[0]
    all_edges=np.hstack([0,borders,len(unit_area)])
    border_midpoints=all_edges[:-1]+(all_edges[1:]-all_edges[:-1])/2
    border_labels=unit_area.iloc[border_midpoints.astype('int')].values
    
    return borders,all_edges,border_midpoints,border_labels

def average_across_trials(trial_da,trial_idx,stim_name):
    ntrials_per_stim={}
    
    #Average & normalize responses of each unit to each stimulus
    gwindow = sg.windows.gaussian(15, std=5)

    #find baseline mean and std per unit
    baseline_mean_per_trial=trial_da.sel(time=slice(-0.5,-0.4)).mean(dim=["time"])
    baseline_mean=baseline_mean_per_trial.mean(dim="trials").values
    baseline_std=baseline_mean_per_trial.std(dim="trials").values

    #find unique stimuli
    stimuli = np.unique(stim_name.values)

    #remove catch trials for this plot
    stimuli = stimuli[stimuli!='catch']

    #pre-allocate array for average
    unit_frs_by_stim = np.zeros((len(trial_da.unit_id),len(trial_da.time),len(stimuli)))
    
    #normalize each unit's avg FRs to its baseline  FR
    for ss,stim in enumerate(stimuli):
        stim_trials = stim_name[stim_name==stim].index.values
        ntrials_per_stim[stim] = len(stim_trials)
        unit_frs_by_stim[:,:,ss] = trial_da.sel(trials=stim_trials).mean(dim="trials").values

        # z-score each unit rel to its baseline
        unit_frs_by_stim[:,:,ss] = ((unit_frs_by_stim[:,:,ss].T- baseline_mean.T)/baseline_std.T).T

        for iu in range(0,len(trial_da.unit_id)):
            unit_frs_by_stim[iu,:,ss]=sg.convolve(unit_frs_by_stim[iu,:,ss],
                                                    gwindow,mode='same')/np.sum(gwindow)
            
    return unit_frs_by_stim,ntrials_per_stim,stimuli


def plot_heatmaps_with_borders(unit_frs_by_stim,trial_duration,time_before_flash,borders,
                               border_midpoints,border_labels,stimuli,ntrials_per_stim,title):

    fig,ax=plt.subplots(1,4,figsize=(10,8))

    for xx in range(0,len(stimuli)): 
        im = ax[xx].imshow(unit_frs_by_stim[:,:,xx],aspect='auto',vmin=-3,vmax=3,
                       cmap=plt.get_cmap('bwr'),interpolation='none',
                       extent=(-time_before_flash,trial_duration-time_before_flash,
                               0,unit_frs_by_stim.shape[0]))

        ax[xx].axvline(0,color='k',linestyle='--',linewidth=1)
        ax[xx].set_title(stimuli[xx]+' n='+str(ntrials_per_stim[stimuli[xx]]))
        ax[xx].set_xlim(-0.5,1.5)
        ax[xx].hlines(unit_frs_by_stim.shape[0]-borders,xmin=-0.5,xmax=1.5,
                       color='k',linewidth=1)
        ax[xx].set_yticks(unit_frs_by_stim.shape[0]-border_midpoints)
        ax[xx].set_yticklabels(border_labels)
        if xx>0:
            ax[xx].set_yticklabels([])

    # this adjusts the other plots to make space for the colorbar
    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.3) 
    cax = plt.axes([0.85, 0.1, 0.025, 0.8])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('z-scored firing rates')
    fig.suptitle(title)
    
    figpath='/root/capsule/results'
    figname=title+'_heatmap.png'
    plt.savefig(os.path.join(figpath,figname), dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',
                transparent=True, bbox_inches='tight', pad_inches=0.1,
                metadata=None)