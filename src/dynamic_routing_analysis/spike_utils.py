import matplotlib.pyplot as plt
import numpy as np
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
        sel_spike_times = spike_times[(spike_times > event_time - time_before) & (spike_times < event_time + time_after)]-event_time
        spike_histogram, bin_edges = np.histogram(sel_spike_times, bins = bins)

        event_aligned_spikes.append(spike_histogram)

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    return np.vstack(event_aligned_spikes).T, bin_centers


def make_neuron_time_trials_tensor(units, trials, time_before, time_after, bin_size):
    
    #units: units to include in tensor
    #trials: trials to include in tensor
    #time_before: time before event to include in PSTH
    #time_after: time after event to include in PSTH
    #bin_size: size of each bin in seconds
    #returns: 3d tensor of shape (units, time, trials)

    unit_count = len(units[:])
    trial_count = len(trials[:])
    time_count = int((time_before+time_after)/bin_size)

    tensor = np.zeros((unit_count, time_count-1, trial_count))

    for uu, unit in units[:].iterrows():
        spike_times = np.array(unit['spike_times'])
        event_times = trials[:]['stim_start_time']
        event_aligned_spikes, bin_centers = makePSTH(spike_times, event_times, time_before, time_after, bin_size)
        tensor[uu,:,:] = event_aligned_spikes/bin_size

    trial_da = xr.DataArray(tensor, dims=("unit_id", "time", "trials"), 
                            coords={
                                "unit_id": units[:].index.values,
                                "time": bin_centers,
                                "trials": trials[:].index.values
                                })

    return trial_da
