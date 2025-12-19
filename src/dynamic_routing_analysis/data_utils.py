import glob
import os

import lazynwb
import npc_lims
import numpy as np
import pandas as pd

vid_angle_npc_names={
    'behavior':'side',
    'face':'front',
    'eye':'eye',
}

def load_trials_or_units(session, table_name, version='any'):
    # convenience function to load trials or units from cache if available,
    # otherwise from npc_sessions
    if session.__class__.__name__ == "NWBFile":
        return getattr(session, table_name)[:]

    if table_name == 'trials':
        try:
            table=pd.read_parquet(
                    npc_lims.get_cache_path('trials',session.id,version=version)
                )
            print(session.id,'cached trials loaded')
        except:
            print(session.id,'cached trials not found, loading with npc_sessions')
            try:
                table = session.trials[:]
            except:
                print(session.id,'loading trials failed')
                return None
    elif table_name == 'units':
        try:
            table=pd.read_parquet(
                    npc_lims.get_cache_path('units',session.id,version=version)
                )
            print(session.id,'cached units loaded')
        except:
            print(session.id,'cached units not found, loading with npc_sessions')
            try:
                table = session.units[:]
            except:
                print(session.id,'loading units failed')
                return None
    return table


def generate_spontaneous_trials_table(session_id,distribution='DR',n_trials=1000,min_iti=5.5,datacube_version='any',random_seed=None):
    # function to generate spontaneous trials for dynamic routing sessions
    
    epochs=pd.read_parquet(
                    npc_lims.get_cache_path('epochs',session_id,version=datacube_version)
                )
    
    path = 's3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/{session_id}.nwb'
    internal_path = 'processing/behavior/rewards'
    rewards = lazynwb.get_df(path.format(session_id=session_id), internal_path, as_polars=False)
    
    spont_epochs=epochs.query('script_name.str.contains("Spontaneous")')

    if len(spont_epochs)==0:
        raise ValueError(f'No spontaneous epochs found for session {session_id}')
    
    spont_trials={
        'trial_index':[],
        'session_id':[],
        'start_time':[],
        'stop_time':[], #equals start_time + min_iti
        'epoch_idx':[],
        'epoch_name':[],
        'is_rewarded':[]
    }

    for rr,row in spont_epochs.iterrows():
        spont_start=row['start_time']
        spont_end=row['stop_time']
        # spont_duration=spont_end-spont_start
        
        sampleITIs=generate_itis(distribution=distribution,n_trials=n_trials,min_iti=min_iti,random_seed=random_seed)
        cum_sampleITIs=np.cumsum(sampleITIs)
        valid_starts=spont_start+cum_sampleITIs[cum_sampleITIs+spont_start<spont_end]
        is_rewarded=np.zeros(len(valid_starts),dtype=bool)

        #find rewards and align trials to them, prevent any overlapping trials
        reward_times=rewards.query('timestamps>=@spont_start and timestamps<=@spont_end')['timestamps'].to_numpy()
        if len(reward_times)>0:
            #remove any valid_starts that would overlap with reward times
            for rt in reward_times:
                incl_idx=np.abs(valid_starts-rt)>(min_iti)
                valid_starts=valid_starts[incl_idx]
                is_rewarded= is_rewarded[incl_idx]
            #add trials aligned to rewards
            valid_starts = np.concatenate([valid_starts, reward_times])
            is_rewarded = np.concatenate([is_rewarded, np.ones(len(reward_times), dtype=bool)])

        spont_trials['session_id'].append(np.repeat(session_id,len(valid_starts)))
        spont_trials['start_time'].append(valid_starts)
        spont_trials['epoch_idx'].append(np.repeat(rr,len(valid_starts)))
        spont_trials['epoch_name'].append(np.repeat(row['script_name'],len(valid_starts)))
        spont_trials['is_rewarded'].append(is_rewarded)

    spont_trials['session_id'] = np.concatenate(spont_trials['session_id'])
    spont_trials['start_time'] = np.concatenate(spont_trials['start_time'])
    spont_trials['stop_time'] = spont_trials['start_time'] + min_iti
    spont_trials['epoch_idx'] = np.concatenate(spont_trials['epoch_idx'])
    spont_trials['epoch_name'] = np.concatenate(spont_trials['epoch_name'])
    spont_trials['is_rewarded'] =  np.concatenate(spont_trials['is_rewarded'])
    spont_trials['trial_index'] = np.arange(len(spont_trials['start_time']))

    return pd.DataFrame(spont_trials).sort_values('start_time').reset_index(drop=True)



def generate_itis(distribution='DR',n_trials=None,min_iti=5.5,random_seed=None):
    #generate inter-trial intervals similar to task trial timing for DR or templeton, or fixed ITI

    import random

    if random_seed is not None:
        random.seed(random_seed)

    if distribution=='DR':
        preStimFramesFixed = 90 # min frames between start of trial and stimulus onset
        preStimFramesVariableMean = 60 # mean of additional preStim frames drawn from exponential distribution
        preStimFramesMax = 360 # max total preStim frames
        # quiescentFrames = 90 # frames before stim onset during which licks delay stim onset
        responseWindow = [6,60]
        postResponseWindowFrames = 180

    elif distribution=='Templeton':
        preStimFramesVariableMean = 30 
        preStimFramesMax = 240
        postResponseWindowFrames = 120

    if n_trials is not None:
        return [generate_itis(distribution=distribution,min_iti=min_iti) for _ in range(n_trials)]
    else:
        if distribution == 'same':
            trial_total_length = min_iti
        else:
            val = preStimFramesFixed + random.expovariate(1/preStimFramesVariableMean) if preStimFramesVariableMean > 1 else preStimFramesFixed + preStimFramesVariableMean
            trial_total_length = (int(min(val,preStimFramesMax)) + responseWindow[1] + postResponseWindowFrames)/60 #in seconds

        return trial_total_length
    

def load_facemap_data(session,session_info=None,trials=None,vid_angle=None,keep_n_SVDs=500,use_s3=True):
    # function to load facemap data from s3 or local cache
    if not vid_angle:
        raise ValueError("vid_angle must be specified")

    if session.__class__.__name__ == "NWBFile":
        if trials is None:
            trials = session.trials[:]
        if not any("facemap" in k for k in session.processing["behavior"].data_interfaces.keys()):
            raise AttributeError(
                f"Facemap data not found in {session.session_id} NWB file"
            )
        facemap = session.processing["behavior"].data_interfaces[
            f"facemap_{vid_angle_npc_names[vid_angle]}_camera"
        ]
        cam_frames = facemap.timestamps[:]

    elif not use_s3:
        if vid_angle=='behavior':
            multi_ROI_path=r"D:\DR Pilot Data\full_video_multi_ROI"
            _dir,vidfilename=os.path.split(glob.glob(os.path.join(session_info.allen_path,"Behavior_*.mp4"))[0])
        elif vid_angle=='face':
            multi_ROI_path=r"D:\DR Pilot Data\full_video_multi_ROI_face"
            _dir,vidfilename=os.path.split(glob.glob(os.path.join(session_info.allen_path,"Face_*.mp4"))[0])

        behav_path = os.path.join(multi_ROI_path,vidfilename[:-4]+'_trimmed_proc.npy')
        behav_info=np.load(behav_path,allow_pickle=True)

        for frame_time in session._video_frame_times:
            if vid_angle_npc_names[vid_angle] in frame_time.name:
                cam_frames=frame_time.timestamps
                break

        facemap_info={}

        # actually keep all ROIs
        # facemap_info['motion']=behav_info.item()['motion']
        facemap_info['motSVD']=behav_info.item()['motSVD']
    # use s3 data
    else:
        import npc_sessions

        camera_to_facemap_name = {
            "face": "Face",
            "behavior": "Behavior",
        }
        motion_svd = npc_sessions.utils.get_facemap_output_from_s3(
                    session.id, camera_to_facemap_name[vid_angle], "motSVD"
                )

        for frame_time in session._video_frame_times:
            if vid_angle_npc_names[vid_angle] in frame_time.name:
                cam_frames=frame_time.timestamps
                break

        facemap_info = {
            #'motion': behav_info['motion'],
            'motSVD': motion_svd
        }

    # calculate mean face motion, SVD in 1 sec prior to each trial
    # 1 sec before stimulus onset
    time_before=0.2
    time_after=0
    fps=60

    behav_SVD_by_trial={}
    behav_motion_by_trial={}
    mean_trial_behav_SVD={}
    mean_trial_behav_motion={}

    if session.__class__.__name__ == "NWBFile":
        rr = 0
        motsvd = np.asarray(facemap.data[:, :])

        behav_SVD_by_trial[rr] = np.zeros(
            (int((time_before + time_after) * fps), keep_n_SVDs, len(trials))
        )
        behav_motion_by_trial[rr] = np.zeros(
            (int((time_before + time_after) * fps), len(trials))
        )

        behav_SVD_by_trial[rr][:] = np.nan
        behav_motion_by_trial[rr][:] = np.nan

        for tt, stimStartTime in enumerate(trials[:]["stim_start_time"]):
            if len(np.where(facemap.timestamps[:]  >= stimStartTime)[0]) > 0:
                stim_start_frame = np.where(facemap.timestamps[:] >= stimStartTime)[0][0]
                trial_start_frame = int(stim_start_frame - time_before * fps)
                trial_end_frame = int(stim_start_frame + time_after * fps)
                if (
                    trial_start_frame < motsvd[:, 0].shape[0]
                    and trial_end_frame < motsvd[:, 0].shape[0]
                ):
                    behav_SVD_by_trial[rr][:, :, tt] = motsvd[
                        trial_start_frame:trial_end_frame, :keep_n_SVDs
                    ]
                    # behav_motion_by_trial[rr][:,tt] = facemap_info['motion'][trial_start_frame:trial_end_frame]
                else:
                    break

        mean_trial_behav_SVD[rr] = np.nanmean(behav_SVD_by_trial[rr], axis=0)

    if not use_s3:
        for rr in range(0,len(facemap_info['motSVD'])):
            behav_SVD_by_trial[rr] = np.zeros((int((time_before+time_after)*fps),keep_n_SVDs,len(trials)))
            behav_motion_by_trial[rr] = np.zeros((int((time_before+time_after)*fps),len(trials)))

            behav_SVD_by_trial[rr][:]=np.nan
            behav_motion_by_trial[rr][:]=np.nan

            for tt,stimStartTime in enumerate(trials[:]['stim_start_time']):
                if len(np.where(cam_frames>=stimStartTime)[0])>0:
                    stim_start_frame=np.where(cam_frames>=stimStartTime)[0][0]
                    trial_start_frame=int(stim_start_frame-time_before*fps)
                    trial_end_frame=int(stim_start_frame+time_after*fps)
                    if trial_start_frame<facemap_info['motSVD'][rr][:,0].shape[0] and trial_end_frame<facemap_info['motSVD'][rr][:,0].shape[0]:
                        behav_SVD_by_trial[rr][:,:,tt] = facemap_info['motSVD'][rr][trial_start_frame:trial_end_frame,:keep_n_SVDs]
                        behav_motion_by_trial[rr][:,tt] = facemap_info['motion'][rr][trial_start_frame:trial_end_frame]
                    else:
                        break

            mean_trial_behav_SVD[rr] = np.nanmean(behav_SVD_by_trial[rr],axis=0)
            mean_trial_behav_motion[rr] = np.nanmean(behav_motion_by_trial[rr],axis=0)

    else:
        if session.__class__.__name__ == "NWBFile":
            motsvd = np.asarray(facemap.data[:, :])
        else:
            motsvd = np.asarray(facemap_info["motSVD"][:, :])

        rr = 0
        behav_SVD_by_trial[rr] = np.zeros(
            (int((time_before + time_after) * fps), keep_n_SVDs, len(trials))
        )
        behav_motion_by_trial[rr] = np.zeros(
            (int((time_before + time_after) * fps), len(trials))
        )

        behav_SVD_by_trial[rr][:] = np.nan
        behav_motion_by_trial[rr][:] = np.nan

        for tt, stimStartTime in enumerate(trials[:]["stim_start_time"]):
            if len(np.where(cam_frames >= stimStartTime)[0]) > 0:
                stim_start_frame = np.where(cam_frames >= stimStartTime)[0][0]
                trial_start_frame = int(stim_start_frame - time_before * fps)
                trial_end_frame = int(stim_start_frame + time_after * fps)
                if (
                    trial_start_frame < motsvd[:, 0].shape[0]
                    and trial_end_frame < motsvd[:, 0].shape[0]
                ):
                    behav_SVD_by_trial[rr][:, :, tt] = motsvd[
                        trial_start_frame:trial_end_frame, :keep_n_SVDs
                    ]
                    # behav_motion_by_trial[rr][:,tt] = facemap_info['motion'][trial_start_frame:trial_end_frame]
                else:
                    break

        mean_trial_behav_SVD[rr] = np.nanmean(behav_SVD_by_trial[rr], axis=0)
        # mean_trial_behav_motion[rr] = np.nanmean(behav_motion_by_trial[rr],axis=0)

    return mean_trial_behav_SVD #mean_trial_behav_motion


def load_LP_data(session, trials=None, vid_angle=None, LP_parts_to_keep=None):
    if not vid_angle:
        raise ValueError("vid_angle must be specified")

    def eu_dist(x, y):
        return np.sqrt((x) ** 2 + (492 - y) ** 2)

    def zscore(x):
        return (x - np.nanmean(x)) / np.nanstd(x)

    def part_info_LP(part, df):
        confidence = df[part_name + '_likelihood'].values.astype('float')
        temp_error = df[part_name + '_temporal_norm'].values.astype('float')
        x = df[part + '_x'].values.astype('float')
        y = df[part + '_y'].values.astype('float')

        x[(confidence < 0.98) | (temp_error > np.nanmean(temp_error) + 3*np.nanstd(temp_error))] = np.nan
        y[(confidence < 0.98) | (temp_error > np.nanmean(temp_error) + 3*np.nanstd(temp_error))] = np.nan

        xy = eu_dist(x, y)

        xy = pd.Series(xy).interpolate(limit_direction='both', method = 'nearest').to_numpy()

        return np.nan_to_num(xy)

    # function to load lightning pose parts from npc_sessions
    if LP_parts_to_keep is None:
        LP_parts_to_keep = ['ear_base_l', 'jaw', 'nose_tip', 'whisker_pad_l_side']

    vid_angle_idx = {
        'behavior': 0,
        'face': 1,
    }
    camera_idx = vid_angle_idx[vid_angle]
    if session.__class__.__name__ == "NWBFile":
        if trials is None:
            trials = session.trials[:]
        if not any(
            k.startswith('lp_')
            for k in session.processing["behavior"].data_interfaces.keys()
        ):
            raise AttributeError(
                f"lightning_pose data not found in {session.session_id} NWB file"
            )
        df = session.processing["behavior"][
            f"lp_{vid_angle_npc_names[vid_angle]}_camera"
        ][:]
        cam_frames = df.timestamps.values

        LP_traces = []
        for part_no, part_name in enumerate(LP_parts_to_keep):
            if f"{part_name}_x" not in df.columns:
                continue
            xy = part_info_LP(part_name, df)
            LP_traces.append(xy)

        if not LP_traces:
            raise ValueError(f"None of requested LP parts found for {vid_angle} camera: {LP_parts_to_keep}")
    else:
        df = session._lp[camera_idx][:]
        cam_frames = df['timestamps'].values.astype('float')

        LP_traces = []
        for part_no, part_name in enumerate(LP_parts_to_keep):
            xy = part_info_LP(part_name, df)
            LP_traces.append(xy)

    LP_info = {
        'LP_traces': np.array(LP_traces).T
    }

    # calculate mean position of all face parts in 200ms sec prior to each trial
    time_before = 0.2
    time_after = 0
    fps = 60

    behav_SVD_by_trial = {}
    mean_trial_behav_SVD = {}

    rr = 0
    LP_traces = np.asarray(LP_info['LP_traces'][:, :])

    behav_SVD_by_trial[rr] = np.zeros((int((time_before + time_after) * fps), len(LP_parts_to_keep), len(trials)))
    behav_SVD_by_trial[rr][:] = np.nan

    for tt, stimStartTime in enumerate(trials[:]['stim_start_time']):
        if len(np.where(cam_frames >= stimStartTime)[0]) > 0:
            stim_start_frame = np.where(cam_frames >= stimStartTime)[0][0]
            trial_start_frame = int(stim_start_frame - time_before * fps)
            trial_end_frame = int(stim_start_frame + time_after * fps)
            if trial_start_frame < LP_traces[:, 0].shape[0] and trial_end_frame < LP_traces[:, 0].shape[0]:
                behav_SVD_by_trial[rr][:, :, tt] = LP_traces[trial_start_frame:trial_end_frame, :]
            else:
                break

    mean_trial_behav_SVD[rr] = np.nan_to_num(np.nanmedian(behav_SVD_by_trial[rr], axis=0))

    return mean_trial_behav_SVD
