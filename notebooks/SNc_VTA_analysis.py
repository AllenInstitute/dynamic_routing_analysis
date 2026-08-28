#%%
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
# import npc_lims
# import npc_sessions


#%%
def getGoodUnits(df):
    return df.query("isi_violations_ratio <= 0.5 & activity_drift <= 0.2 & presence_ratio >= 0.7 & amplitude_cutoff <= 0.1 & decoder_label != 'noise'")

def getAlignedSpikes(spikeTimes,startTimes,windowDur,binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    spikes = np.zeros((len(startTimes),bins.size-1))
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikeTimes,start)
        endInd = np.searchsorted(spikeTimes,start+windowDur)
        spikes[i] = np.histogram(spikeTimes[startInd:endInd]-start, bins)[0]
    return spikes

#%%

cache_version = 'v0.0.289'

allUnitsDf = pd.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/'+cache_version+'/consolidated/units.parquet')

#%%
areas = ('SNc','VTA')

unitsDf = allUnitsDf.query('structure.isin(@areas)')

sessionIds = unitsDf['session_id'].unique()

nUnits = [getGoodUnits(allUnitsDf.query('session_id==@sid & structure.isin(@areas)')).shape[0] for sid in sessionIds]

nSessions = np.sum(np.array(nUnits)>0)

# #%%
# unitsDf = pd.read_parquet(r"\\allen\programs\mindscope\workgroups\dynamicrouting\ben\SNc_VTA_waveforms.parquet")
# unitsDf = getGoodUnits(unitsDf)

#%%
nUnits = unitsDf.shape[0]
nSessions = len(unitsDf['session_id'].unique())
nMice = len(np.unique([session[:6] for session in unitsDf['session_id']]))

#%%
trialsDf = pd.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/'+cache_version+'/consolidated/trials.parquet')

#%%
def getSpikeDur(s,thresh=0.2):
    neg = np.argmin(s)
    pos = np.argmax(s)
    if neg < pos:
        s = -s
        pos,neg = neg,pos
    start = np.where(s[:pos] < s[pos] * thresh)[0][-1]
    triphase = np.where(s[neg:] > s[pos] * thresh)[0] 
    if any(triphase):
        end = neg + triphase[-1] + 1
    else:
        end = neg + np.where(s[neg:] > s[neg] * thresh)[0][0]
    return start, end

#%% 
fig = plt.figure()
wf = np.stack(unitsDf['waveform_mean'])
n = wf.shape[0]
t = np.arange(wf.shape[1])/30000
nrows = int(n**0.5)
ncols = int(n**0.5 + 1)
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
i = 0
j = 0
for w in wf:
    start,end = getSpikeDur(w)
    if j==ncols:
        i += 1
        j = 0
    ax = fig.add_subplot(gs[i,j])
    j += 1
    ax.plot(t,w,'k')
    ax.plot([start/30000,end/30000],[0,0],'r')
    ax.set_xlim([0.002,0.006])
    ax.set_axis_off()

#%%
spikeDur = []
for w in wf:
    start,end = getSpikeDur(w)
    spikeDur.append((end-start)/30000)
spikeDur = 1000 * np.array(spikeDur)

#%%
spontRate = []
for u,s in zip(unitsDf['unit_id'],unitsDf['session_id']):
    spikeTimes = unitsDf.query('unit_id==@u')['spike_times'].iloc[0]
    trials = trialsDf.query("session_id==@s")
    spontRate.append(np.mean(getAlignedSpikes(spikeTimes,trials['quiescent_start_time'],1.5,1.5),axis=0)[0])
spontRate = np.array(spontRate)

#%%
durThresh = 0.9
rateThresh = 12

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([durThresh]*2,[0,200],'k--',alpha=0.25)
ax.plot([0,4],[rateThresh]*2,'k--',alpha=0.25)
ax.plot(spikeDur,spontRate,'o',mec='k',mfc='none',alpha=0.5)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,2.5])
ax.set_ylim([0,125])
ax.set_xlabel('Spike duration (ms)')
ax.set_ylabel('Quiescent period firing rate (spikes/s)')
plt.tight_layout()

#%%
isDop = (spikeDur > durThresh) & (spontRate < rateThresh)

#%%
binSize = 0.05
preTime = 1.5
windowDur = preTime * 2
t = np.arange(0,windowDur,binSize) + binSize/2 - preTime
psth = {dop: {cont: [] for cont in ('contingent','noncontingent')} for dop in ('dop','notDop')}
for dop,uind in zip(psth,(isDop,~isDop)):
    for u,s in zip(unitsDf[uind]['unit_id'],unitsDf[uind]['session_id']):
        spikeTimes = unitsDf[uind].query('unit_id==@u')['spike_times'].iloc[0]
        trials = trialsDf.query("session_id==@s")
        for cont in psth[dop]:
            ind = (trials['trial_index_in_block']==0) & trials['is_'+cont+'_reward']
            startTimes = trials[ind]['reward_time']
            psth[dop][cont].append(np.mean(getAlignedSpikes(spikeTimes,startTimes-preTime,windowDur,binSize),axis=0))

#%%
for dop,ylim in zip(psth,([0,12],[0,24])):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,0],[0,100],'k--',alpha=0.25)
    for cont,alpha in zip(psth[dop],(1,0.5)):
        r = np.nanmean(psth[dop][cont],axis=0)/binSize
        ax.plot(t,r,color='k',alpha=alpha,label=cont)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim(ylim)
        ax.set_xlabel('Time from reward (s; first trial of block)')
        ax.set_ylabel('Spikes/s')
        if dop=='dop':
            ax.legend()
    plt.tight_layout()

#%%
uind = isDop
minTrials = 1
binSize = 0.05
preTime = 1.5
windowDur = preTime * 2
t = np.arange(0,windowDur+binSize,binSize)[:-1] + binSize/2 - preTime
blockType = ('vis rewarded','aud rewarded')
stimType = ('vis target','aud target')
trialType = ('first','last')
respType = ('response','no response')
alignTo = ('stimulus','outcome')
psth = {block: {stim: {trial: {resp: {align: [] for align in alignTo} for resp in respType} for trial in trialType} for stim in stimType} for block in blockType}
for u,s in zip(unitsDf[uind]['unit_id'],unitsDf[uind]['session_id']):
    spikeTimes = unitsDf[uind].query('unit_id==@u')['spike_times'].iloc[0]
    trials = trialsDf.query("session_id==@s")
    isResp = trials['is_response']
    isVisTarg = trials['is_vis_target']
    isAudTarg = trials['is_aud_target']
    isVisRew = trials['is_vis_rewarded']
    isAudRew = trials['is_aud_rewarded']
    for block in blockType:
        blockTrials = isVisRew if 'vis' in block else isAudRew
        for stim in stimType:
            stimTrials = isVisTarg if 'vis' in stim else isAudTarg
            for trial in trialType:
                isTrial = np.zeros(stimTrials.size,dtype=bool)
                nTrialsAvg = 1 if trial=='first' else 5
                i = slice(0,nTrialsAvg) if trial=='first' else slice(-nTrialsAvg,None)
                isTrial[[np.where(stimTrials & (trials['block_index']==b))[0][i] for b in range(1,6)]] = True
                for resp in respType:
                    ind = blockTrials & isTrial & (isResp if resp=='response' else ~isResp)
                    for align in alignTo:
                        if ind.sum() < minTrials:
                            psth[block][stim][trial][resp][align].append(np.full(t.size,np.nan))
                        else:
                            if align=='stimulus': 
                                startTimes = trials[ind]['stim_start_time']
                            elif align=='outcome':
                                startTimes =  trials[ind]['response_time'] if resp=='response' else trials[ind]['response_window_stop_time']
                            psth[block][stim][trial][resp][align].append(np.mean(getAlignedSpikes(spikeTimes,startTimes-preTime,windowDur,binSize),axis=0))



# %%
for align in alignTo:
    for block in blockType:
        fig = plt.figure()
        fig.suptitle(block+', align to '+align)
        gs = matplotlib.gridspec.GridSpec(2,2)
        for i,stim in enumerate(stimType):
            for j,resp in enumerate(respType):
                ax = fig.add_subplot(gs[i,j])
                ax.plot([0,0],[0,100],'k--',alpha=0.25)
                for trial,alpha in zip(trialType,(1,0.5)):
                    r = np.nanmean(psth[block][stim][trial][resp][align],axis=0)/binSize
                    ax.plot(t,r,color='k',alpha=alpha,label=trial+' trial')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_ylim([0,12])
                if i==0 and j==1:
                    ax.legend(loc='upper right',fontsize=8)
                ax.set_title(stim+', '+resp)
        plt.tight_layout()

# %%
for align in alignTo:
    fig = plt.figure()
    fig.suptitle('align to '+align)
    gs = matplotlib.gridspec.GridSpec(2,2)
    for i,stim in enumerate(stimType):
        for j,trial in enumerate(trialType):
            ax = fig.add_subplot(gs[i,j])
            ax.plot([0,0],[0,100],'k--',alpha=0.25)
            for block in blockType:
                alpha,lbl = (1,'rewarded') if ('vis' in stim and 'vis' in block) or ('aud' in stim and 'aud' in block) else (0.5,'non-rewarded')
                r = np.nanmean(psth[block][stim][trial]['response'][align],axis=0)/binSize
                ax.plot(t,r,color='k',alpha=alpha,label=lbl)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_ylim([0,12])
            if i==0 and j==1:
                ax.legend(loc='upper right',fontsize=8)
            ax.set_title(stim+', '+trial+' trial')
    plt.tight_layout()

# %%