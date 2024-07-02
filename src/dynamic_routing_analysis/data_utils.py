import glob
import os

import npc_lims
import npc_sessions
import numpy as np
import pandas as pd


def load_trials_or_units(session, table_name):
    # convenience function to load trials or units from cache if available, 
    # otherwise from npc_sessions
    if table_name == 'trials':
        try:
            table=pd.read_parquet(
                    npc_lims.get_cache_path('trials',session.id,version='any')
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
                    npc_lims.get_cache_path('units',session.id,version='any')
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



def load_facemap_data(session,session_info,trials,vid_angle,keep_n_SVDs=500,use_s3=True):
    #function to load facemap data from s3 or local cache
    vid_angle_npc_names={
            'behavior':'side',
            'face':'front',
            'eye':'eye',
            }

    if use_s3==False:
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

        #actually keep all ROIs
        #facemap_info['motion']=behav_info.item()['motion']
        facemap_info['motSVD']=behav_info.item()['motSVD']
    #use s3 data
    else:
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

    #calculate mean face motion, SVD in 1 sec prior to each trial
    # 1 sec before stimulus onset
    time_before=0.2
    time_after=0
    fps=60

    behav_SVD_by_trial={}
    behav_motion_by_trial={}
    mean_trial_behav_SVD={}
    mean_trial_behav_motion={}

    if use_s3==False:
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
        rr=0
        motsvd=np.asarray(facemap_info['motSVD'][:,:])

        behav_SVD_by_trial[rr] = np.zeros((int((time_before+time_after)*fps),keep_n_SVDs,len(trials)))
        behav_motion_by_trial[rr] = np.zeros((int((time_before+time_after)*fps),len(trials)))

        behav_SVD_by_trial[rr][:]=np.nan
        behav_motion_by_trial[rr][:]=np.nan

        for tt,stimStartTime in enumerate(trials[:]['stim_start_time']):
            if len(np.where(cam_frames>=stimStartTime)[0])>0:
                stim_start_frame=np.where(cam_frames>=stimStartTime)[0][0]
                trial_start_frame=int(stim_start_frame-time_before*fps)
                trial_end_frame=int(stim_start_frame+time_after*fps)
                if trial_start_frame<motsvd[:,0].shape[0] and trial_end_frame<motsvd[:,0].shape[0]:
                    behav_SVD_by_trial[rr][:,:,tt] = motsvd[trial_start_frame:trial_end_frame,:keep_n_SVDs]    
                    # behav_motion_by_trial[rr][:,tt] = facemap_info['motion'][trial_start_frame:trial_end_frame]
                else:
                    break

        mean_trial_behav_SVD[rr] = np.nanmean(behav_SVD_by_trial[rr],axis=0)
        # mean_trial_behav_motion[rr] = np.nanmean(behav_motion_by_trial[rr],axis=0)

    return mean_trial_behav_SVD #mean_trial_behav_motion