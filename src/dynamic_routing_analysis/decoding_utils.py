import gc
import logging
import pickle
import time

import npc_lims
import numpy as np
import pandas as pd
import upath
import zarr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from dynamic_routing_analysis import data_utils, spike_utils

logger = logging.getLogger(__name__)

class NotEnoughBlocksError(Exception):
    pass


# Dump the dictionary to the Zarr file
def dump_dict_to_zarr(group, data):
    for key, value in data.items():
        print(key)
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            dump_dict_to_zarr(group=subgroup, data=value)
        #elif isinstance(value, npc_lims.SessionInfo):
            #continue
        #elif isinstance(value, upath.implementations.cloud.S3Path):
            #value=str(value)
        else:
            try:
                group[key] = value
            except:
                logger.warning(f'Could not save {key} of type {type(value)} to zarr {group}')

# 'linearSVC' or 'LDA' or 'RandomForest'
def decoder_helper(input_data,labels,decoder_type='linearSVC',crossval='5_fold',
                   crossval_index=None,labels_as_index=False,train_test_split_input=None,
                   regularization=None,penalty=None,solver=None,n_jobs=None,set_random_state=None):
    
    #helper function to decode labels from input data using different decoder models

    if decoder_type=='linearSVC':
        from sklearn.svm import LinearSVC
        if regularization is None:
            regularization = 1.0
        if penalty is None:
            penalty = 'l2'
        if solver is not None:
            logger.warning('Solver not used for LinearSVC')
        clf=LinearSVC(max_iter=5000,dual='auto',class_weight='balanced',
                      C=regularization,penalty=penalty,n_jobs=n_jobs)
    elif decoder_type=='LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        if solver is None:
            solver = 'svd'
        if regularization is not None:
            logger.warning('Regularization not used for LDA')
        if penalty is not None:
            logger.warning('Penalty not used for LDA')
        clf=LinearDiscriminantAnalysis(solver=solver,n_jobs=n_jobs)
    elif decoder_type=='RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        if regularization is not None:
            logger.warning('Regularization not used for RandomForest')
        if penalty is not None:
            logger.warning('Penalty not used for RandomForest')
        if solver is not None:
            logger.warning('Solver not used for RandomForest')
        clf=RandomForestClassifier(class_weight='balanced',n_jobs=n_jobs)
    elif decoder_type=='LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        if regularization is None:
            regularization = 1.0
        if penalty is None:
            penalty = 'l2'
        if solver is None:
            solver = 'lbfgs'
        clf=LogisticRegression(max_iter=5000,class_weight='balanced',
                               C=regularization,penalty=penalty,solver=solver,n_jobs=n_jobs)

    output={}

    # scaler = StandardScaler()
    scaler = RobustScaler()

    scaler.fit(input_data)
    X = scaler.transform(input_data)
    unique_labels=np.unique(labels)
    if labels_as_index==True:
        labels=np.array([np.where(unique_labels==x)[0][0] for x in labels])

    y = labels

    if len(np.unique(labels))>2:
        y_dec_func=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    else:
        y_dec_func=np.full(len(y), fill_value=np.nan)

    if type(y[0])==bool:
        ypred=np.full(len(y), fill_value=False)
    elif type(y[0])==str:
        ypred=np.full(len(y), fill_value='       ')
    else:
        ypred=np.full(len(y), fill_value=np.nan)
    
    ypred_proba=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    decision_function=np.full((len(y)), fill_value=np.nan)

    tidx_used=[]

    coefs=[]
    classes=[]
    intercept=[]
    params=[]
    ypred_all=[]
    ypred_train=[]
    ytrue_train=[]
    train_trials=[]
    test_trials=[]
    dec_func_all=[]
    y_dec_func=[]
    models=[]
    cr_dict_train = []
    balanced_accuracy_train = []

    cr_dict_test = []
    balanced_accuracy_test = []

    #make train, test splits based on block number
    if crossval=='blockwise':
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        block_numbers=np.unique(block_number)
        for bb in block_numbers:
            not_block_inds=np.where(block_number!=bb)[0]
            train.append(not_block_inds)

            #equalize number of trials for each condition
            subset_ind=[]
            conds = np.unique(y[not_block_inds])
            cond_count=[]
            for cc in conds:
                cond_count.append(np.sum(y[not_block_inds]==cc))
            use_trnum=np.min(cond_count)
            for cc in conds:
                cond_inds=np.where(y[not_block_inds]==cc)[0]
                subset_ind.append(np.random.choice(cond_inds,use_trnum,replace=False))
            subset_ind=np.sort(np.hstack(subset_ind))
            train.append(not_block_inds[subset_ind])

            block_inds=np.where(block_number==bb)[0]
            test.append(block_inds)
        train_test_split=zip(train,test)

    elif 'forecast' in crossval:
        if crossval_index is None:
            raise ValueError('Must provide crossval_index')
        train=[]
        test=[]
        block_number=crossval_index
        block_numbers=np.unique(block_number)

        #make list of training and test blocks
        #training = adjacent blocks
        #testing = block before and block after (if exist)
        train_blocks=[]
        test_blocks=[]

        if crossval=='forecast_train_2':
            for bb in block_numbers[:-1]:
                train_blocks.append([bb,bb+1])
                if bb==0:
                    test_blocks.append([bb+2])
                elif bb==block_numbers[-2]:
                    test_blocks.append([bb-1])
                else:
                    test_blocks.append([bb-1,bb+2])

        elif crossval=='forecast_train_3':
            for bb in block_numbers[:-2]:
                train_blocks.append([bb,bb+1,bb+2])
                if bb==0:
                    test_blocks.append([bb+3])
                elif bb==block_numbers[-3]:
                    test_blocks.append([bb-1])
                else:
                    test_blocks.append([bb-1,bb+3])

        for bb in range(0,len(train_blocks)):
            train.append(np.where(np.isin(block_number,train_blocks[bb]))[0])
            test.append(np.where(np.isin(block_number,test_blocks[bb]))[0])

        train_test_split=zip(train,test)

    elif crossval=='5_fold':
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        train_test_split = skf.split(input_data, labels)

    elif crossval=='5_fold_constant':
        if train_test_split_input is None:
            raise ValueError('Must provide train_test_split_input')
        train_test_split = train_test_split_input

    elif crossval=='5_fold_set_random_state':
        if set_random_state==None:
            set_random_state=0
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=set_random_state)
        train_test_split = skf.split(input_data, labels)

    for train,test in train_test_split:

        clf.fit(X[train],y[train])
        prediction=clf.predict(X[test])
        balanced_accuracy_train.append(balanced_accuracy_score(y[train], clf.predict(X[train]),
                                                               sample_weight=None, adjusted=False))

        balanced_accuracy_test.append(balanced_accuracy_score(y[test], clf.predict(X[test]),
                                                               sample_weight=None, adjusted=False))

        ypred_all.append(prediction)
        ypred_train.append(clf.predict(X[train]))
        ytrue_train.append(y[train])
        y_dec_func.append(decision_function)
        tidx_used.append([test])
        classes.append(clf.classes_)
        # intercept.append(clf.intercept_)
        params.append(clf.get_params())
        train_trials.append(train)
        test_trials.append(test)

        if decoder_type == 'LDA' or decoder_type == 'RandomForest' or decoder_type=='LogisticRegression':
            ypred_proba[test,:] = clf.predict_proba(X[test])
        else:
            ypred_proba[test,:] = np.full((len(test),len(np.unique(labels))), fill_value=np.nan)

        if decoder_type=='LDA' or decoder_type=='linearSVC' or decoder_type=='LogisticRegression':
            decision_function[test]=clf.decision_function(X[test])
        else:
            decision_function[test]=np.full((len(test)), fill_value=np.nan)

        models.append(clf)

    #fit on all trials
    clf.fit(X, y)
    ypred = clf.predict(X)

    if decoder_type == 'LDA' or decoder_type == 'RandomForest' or decoder_type=='LogisticRegression':
        predict_proba_all_trials = clf.predict_proba(X)
    else:
        predict_proba_all_trials = np.full((X.shape[0],len(np.unique(labels))), fill_value=np.nan)

    if decoder_type == 'LDA' or decoder_type == 'linearSVC' or decoder_type == 'LogisticRegression':
        coefs = clf.coef_
        dec_func_all_trials = clf.decision_function(X)
    else:
        coefs = np.full((X.shape[1]), fill_value=False)
        dec_func_all_trials = np.full((X.shape[0])), fill_value=np.nan)

    intercept = clf.intercept_

    #scikit-learn's classification report
    output['cr']=cr_dict_test

    #predicted label from training/testing on all trials
    output['pred_label']=ypred
    #true original label
    output['true_label']=y
    #predicted label for trials in each train-test fold
    output['pred_label_all']=ypred_all
    #indices of trials used in test for each fold
    output['trials_used']=tidx_used

    #decision function using cross-validated folds
    output['decision_function']=decision_function
    #decision function from training/testing on all trials
    output['decision_function_all']=dec_func_all_trials
    #predict probability using cross-validated folds
    output['predict_proba']=ypred_proba
    #predict probability from training/testing on all trials
    output['predict_proba_all_trials']=predict_proba_all_trials if 'predict_proba_all_trials' in locals() else None
    #coefficients for each feature
    output['coefs']=coefs
    output['classes']=classes
    output['intercept']=intercept
    #input parameters
    output['params']=params
    #mean balanced accuracy across folds
    output['balanced_accuracy_test']=np.nanmean(balanced_accuracy_test)

    output['pred_label_train']=ypred_train
    output['true_label_train']=ytrue_train

    output['cr_train']=cr_dict_train
    #balanced accuracy for training data (all folds)
    output['balanced_accuracy_train']=np.nanmean(balanced_accuracy_train)

    output['train_trials']=train_trials
    output['test_trials']=test_trials

    output['models']=models
    output['scaler']=scaler
    output['label_names']=unique_labels
    output['labels']=labels

    return output


##TODO:
# incorporate additional parameters
# add option to decode from timebins
# add option to use inputs with top decoding weights (use_coefs)
def decode_context_with_linear_shift(session=None,params=None,trials=None,units=None,session_info=None,use_zarr=False):

    decoder_results={}

    input_data_type=params['input_data_type']
    vid_angle_facemotion =params['vid_angle_facemotion']
    vid_angle_LP = params['vid_angle_LP']
    central_section=params['central_section']
    exclude_cue_trials=params['exclude_cue_trials']
    n_unit_threshold=params['n_unit_threshold']
    keep_n_SVDs=params['keep_n_SVDs']
    LP_parts_to_keep = params['LP_parts_to_keep']

    # predict=params['predict']
    # trnum=params['trnum']
    n_units_input=params['n_units']
    # u_min=params['u_min']
    n_repeats=params['n_repeats']
    spikes_binsize=params['spikes_binsize']
    spikes_time_before=params['spikes_time_before']
    spikes_time_after=params['spikes_time_after']

    # decoder_binsize=params['decoder_binsize']
    # decoder_time_before=params['decoder_time_before']
    # decoder_time_after=params['decoder_time_after']
    # balance_labels=params['balance_labels']
    savepath=params['savepath']
    filename=params['filename']
    # use_structure_probe=params['use_structure_probe']
    crossval=params['crossval']
    # all_areas=params['all_areas']
    labels_as_index=params['labels_as_index']
    decoder_type=params['decoder_type']
    # use_coefs=params['use_coefs']
    # generate_labels=params['generate_labels']

    if 'regularization' in params:
        regularization=params['regularization']
    else:
        regularization=None
    if 'penalty' in params:
        penalty=params['penalty']
    else:
        penalty=None
    if 'solver' in params:
        solver=params['solver']
    else:
        solver=None

    if 'only_use_all_units' in params:
        only_use_all_units=params['only_use_all_units']
    else:
        only_use_all_units=False

    if 'return_results' in params:
        return_results=params['return_results']
    else:
        return_results=False

    if 'split_area_by_probe' in params:
        split_area_by_probe=params['split_area_by_probe']
    else:
        split_area_by_probe=1

    if 'select_single_area' in params:
        select_single_area=params['select_single_area']
    else:
        select_single_area=None

    if session is not None:
        session_id = session.session_id
        if session_info is None:
            session_info = npc_lims.get_session_info(session_id)
    elif session_info is not None:
        session_id=str(session_info.id)
    logger.info(f'{session_id} | Parameters parsed. Starting decoding analysis')

    ##Option to input session or trials/units/session_info directly
    ##note: inputting session may not work with Code Ocean

    if trials is None and session is not None:
        trials = data_utils.load_trials_or_units(session, 'trials')
    elif trials is None:
        trials=pd.read_parquet(
            npc_lims.get_cache_path('trials',session_id)
        )

    trials_original_index=trials.index.values

    if exclude_cue_trials:
        trials=trials.query('is_reward_scheduled==False')
        trials_original_index=trials.index.values
        trials=trials.reset_index()

    if 'predict' in params:
        predict=params['predict']
    else:
        predict='context'

    if predict=='vis_appropriate_response':
        trials=trials.query('(is_vis_target and is_aud_context) or (is_aud_target and is_vis_context)')
        trials_original_index=trials.index.values
        trials=trials.reset_index()
        # is_vis_appropriate_response=np.zeros(len(trials))
        is_vis_appropriate_response=(trials['is_vis_target'].values & trials['is_response'].values) | (trials['is_aud_target'].values & ~trials['is_response'].values)

    if input_data_type=='spikes':
        #make data array
        if units is None and session is not None:
            units = data_utils.load_trials_or_units(session, 'units')
        elif units is None:
            units=pd.read_parquet(
                npc_lims.get_cache_path('units',session_id)
            )

        if split_area_by_probe==1:
            #add probe to structure name
            structure_probe=spike_utils.get_structure_probe(units)
            for uu, unit in units.iterrows():
                units.loc[units['unit_id']==unit['unit_id'],'structure']=structure_probe.loc[structure_probe['unit_id']==unit['unit_id'],'structure_probe']

        #make trial data array for baseline activity
        trial_da = spike_utils.make_neuron_time_trials_tensor(units, trials, spikes_time_before, spikes_time_after, spikes_binsize)

    ### TODO: update to work with code ocean
    elif input_data_type=='facemap':
        # mean_trial_behav_SVD,mean_trial_behav_motion = load_facemap_data(session,session_info,trials,vid_angle)
        mean_trial_behav_SVD = data_utils.load_facemap_data(session,session_info,trials,vid_angle_facemotion,keep_n_SVDs)

    # Shailaja
    elif input_data_type == 'LP':
        mean_trial_behav_SVD = data_utils.load_LP_data(session, trials, vid_angle_LP, LP_parts_to_keep)


    #make fake blocks for templeton sessions
    if 'Templeton' in session_info.project:
        start_time=trials['start_time'].iloc[0]
        fake_context=np.full(len(trials), fill_value='nan')
        fake_block_nums=np.full(len(trials), fill_value=np.nan)
        block_context_names=['vis','aud']

        if np.random.choice(block_context_names,1)=='vis':
            block_contexts=['vis','aud','vis','aud','vis','aud']
        else:
            block_contexts=['aud','vis','aud','vis','aud','vis']

        for block in range(0,6):
            block_start_time=start_time+block*10*60
            block_end_time=start_time+(block+1)*10*60
            block_trials=trials[:].query('start_time>=@block_start_time').index
            fake_context[block_trials]=block_contexts[block]
            fake_block_nums[block_trials]=block
        trials['block_index']=fake_block_nums
        trials['context_name']=fake_context

    n_unique_blocks=len(trials['block_index'].unique())
    if n_unique_blocks<6:
        raise NotEnoughBlocksError('Not enough blocks ('+str(n_unique_blocks)+') in session '+session_id)

    if central_section=='4_blocks':
        #find middle 4 block labels
        middle_4_block_trials=trials.query('block_index>0 and block_index<5')
        middle_4_blocks=middle_4_block_trials.index.values

        #find the number of trials to shift by, from -1 to +1 block
        negative_shift=middle_4_blocks.min()
        positive_shift=trials.index.max()-middle_4_blocks.max()
        shifts=np.arange(-negative_shift,positive_shift+1)
    elif central_section=='4_blocks_plus':
        #find middle 4 block labels
        first_block=trials.query('block_index==0').index.values
        middle_of_first=first_block[np.ceil(len(first_block)/2).astype('int')]

        last_block=trials.query('block_index==5').index.values
        middle_of_last=last_block[np.ceil(len(last_block)/2).astype('int')]

        middle_4_block_trials=trials.loc[middle_of_first:middle_of_last]
        middle_4_blocks=middle_4_block_trials.index.values

        #find the number of trials to shift by, from -1 to +1 block
        negative_shift=middle_4_blocks.min()
        positive_shift=trials.index.max()-middle_4_blocks.max()
        shifts=np.arange(-negative_shift,positive_shift+1)

    decoder_results[session_id]={}
    decoder_results[session_id]['trials_original_index']=trials_original_index
    decoder_results[session_id]['shifts'] = shifts
    decoder_results[session_id]['middle_4_blocks'] = middle_4_blocks
    decoder_results[session_id]['spikes_binsize'] = spikes_binsize
    decoder_results[session_id]['spikes_time_before'] = spikes_time_before
    decoder_results[session_id]['spikes_time_after'] = spikes_time_after
    # decoder_results[session_id]['decoder_binsize'] = decoder_binsize
    # decoder_results[session_id]['decoder_time_before'] = decoder_time_before
    # decoder_results[session_id]['decoder_time_after'] = decoder_time_after
    decoder_results[session_id]['input_data_type'] = input_data_type
    decoder_results[session_id]['n_units'] = n_units_input
    decoder_results[session_id]['n_repeats'] = n_repeats
    decoder_results[session_id]['crossval'] = crossval
    decoder_results[session_id]['labels_as_index'] = labels_as_index
    decoder_results[session_id]['decoder_type'] = decoder_type
    decoder_results[session_id]['regularization'] = regularization
    decoder_results[session_id]['penalty'] = penalty
    decoder_results[session_id]['solver'] = solver
    decoder_results[session_id]['only_use_all_units'] = only_use_all_units

    import dataclasses
    session_info=dataclasses.asdict(session_info)
    for ii in session_info.keys():
        if type(session_info[ii]) not in [int, str, bool, dict, list]:
            session_info[ii]=str(session_info[ii])

    decoder_results[session_id]['session_info'] = session_info
    #keep track of which cache path was used
    try:
        decoder_results[session_id]['trial_cache_path'] = str(npc_lims.get_cache_path('trials',session_id,version='any'))
    except:
        decoder_results[session_id]['trial_cache_path'] = ''
    try:
        decoder_results[session_id]['unit_cache_path'] = str(npc_lims.get_cache_path('units',session_id,version='any'))
    except:
        decoder_results[session_id]['unit_cache_path'] = ''

    if input_data_type=='facemap':
        decoder_results[session_id]['vid_angle'] = vid_angle_facemotion

    if input_data_type=='LP':
        decoder_results[session_id]['vid_angle'] = vid_angle_LP
    decoder_results[session_id]['trials'] = trials
    decoder_results[session_id]['results'] = {}


    if input_data_type=='spikes':
        if only_use_all_units:
            areas=['all']
        else:
            areas=units['structure'].unique()
            if select_single_area is None:
                areas=np.concatenate([areas,['all']])

            #add non-probe-specific area to areas
            all_probe_areas=[]
            if len(units.query('structure.str.contains("probe")'))>0:
                probe_areas=units.query('structure.str.contains("probe")')['structure'].unique()
                for pa in probe_areas:
                    all_probe_areas.append([pa.split('_')[0]+'_all'])

            general_areas=np.unique(np.array(all_probe_areas))
            areas=np.concatenate([areas,general_areas])

            #consolidate SC areas
            for aa in areas:
                if aa in ['SCop','SCsg','SCzo']:
                    if 'SCs' not in areas:
                        areas=np.concatenate([areas,['SCs']])
                elif aa in ['SCig','SCiw','SCdg','SCdw']:
                    if 'SCm' not in areas:
                        areas=np.concatenate([areas,['SCm']])

    elif input_data_type=='facemap' or input_data_type=='LP':
        # areas = list(mean_trial_behav_SVD.keys())
        areas=[0]

    decoder_results[session_id]['areas'] = areas

    for aa in areas:
        #make shifted trial data array
        if input_data_type=='spikes':
            if aa == 'all':
                area_units=units
            elif '_all' in aa:
                temp_area=aa.split('_')[0]
                possible_probe_areas=[temp_area+'_probeA',temp_area+'_probeB',temp_area+'_probeC',
                                    temp_area+'_probeD',temp_area+'_probeE',temp_area+'_probeF']
                area_units=units.query('structure in @possible_probe_areas')
            elif aa=='SCs':
                area_units=units.query('structure=="SCop" or structure=="SCsg" or structure=="SCzo"')
            elif aa=='SCm':
                area_units=units.query('structure=="SCig" or structure=="SCiw" or structure=="SCdg" or structure=="SCdw"')
            else:
                area_units=units.query('structure==@aa')

            n_units=len(area_units)
            if n_units<n_unit_threshold:
                continue

            area_unit_ids=area_units['unit_id'].values
        else:
            area_units = [0]

        decoder_results[session_id]['results'][aa]={}
        decoder_results[session_id]['results'][aa]['shift']={}
        decoder_results[session_id]['results'][aa]['no_shift']={}

        if input_data_type=='spikes':

            decoder_results[session_id]['results'][aa]['unit_ids']={}
            decoder_results[session_id]['results'][aa]['n_units']={}
            decoder_results[session_id]['results'][aa]['unit_ids']=area_units['unit_id'].values
            decoder_results[session_id]['results'][aa]['n_units']=len(area_units)

            #find mean ccf location of units
            decoder_results[session_id]['results'][aa]['ccf_ap_mean']=area_units['ccf_ap'].mean()
            decoder_results[session_id]['results'][aa]['ccf_dv_mean']=area_units['ccf_dv'].mean()
            decoder_results[session_id]['results'][aa]['ccf_ml_mean']=area_units['ccf_ml'].mean()


        #loop through repeats
        for nunits in n_units_input:
            if nunits!='all':
                if nunits>len(area_units):
                    continue
            decoder_results[session_id]['results'][aa]['shift'][nunits]={}
            decoder_results[session_id]['results'][aa]['no_shift'][nunits]={}
            for rr in range(n_repeats):
                decoder_results[session_id]['results'][aa]['shift'][nunits][rr]={}
                decoder_results[session_id]['results'][aa]['no_shift'][nunits][rr]={}

                if input_data_type=='spikes':
                    if nunits=='all':
                        sel_units=area_unit_ids
                    else:
                        sel_units=np.random.choice(area_unit_ids,nunits,replace=False)
                elif input_data_type=='facemap':
                    if nunits=='all':
                        sel_units=np.arange(0,keep_n_SVDs)
                    else:
                        sel_units=np.random.choice(np.arange(0,keep_n_SVDs),nunits,replace=False)

                elif input_data_type=='LP':

                    sel_units=np.arange(0, len(LP_parts_to_keep))

                decoder_results[session_id]['results'][aa]['shift'][nunits][rr]['sel_units']=sel_units
                decoder_results[session_id]['results'][aa]['no_shift'][nunits][rr]['sel_units']=sel_units

                #run once with all trials and no shifts
                if predict=='context':
                    labels=trials['context_name'].values
                elif predict=='vis_appropriate_response':
                    labels=is_vis_appropriate_response

                if input_data_type=='spikes':
                    input_data=trial_da.sel(unit_id=sel_units).mean(dim='time').values.T
                elif input_data_type=='facemap' or input_data_type == 'LP':
                    input_data=mean_trial_behav_SVD[aa][sel_units].T

                if crossval=='5_fold_constant':
                    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=165482)
                    train_test_split = skf.split(np.zeros(len(labels)), labels)
                else:
                    train_test_split=None
                decoder_results[session_id]['results'][aa]['no_shift'][nunits][rr] = decoder_helper(
                    input_data=input_data,
                    labels=labels,
                    decoder_type=decoder_type,
                    crossval=crossval,
                    crossval_index=None,
                    labels_as_index=labels_as_index,
                    train_test_split_input=train_test_split,
                    regularization=regularization,
                    penalty=penalty,
                    solver=solver)

                #loop through shifts
                for sh,shift in enumerate(shifts):

                    if predict=='context':
                        labels=middle_4_block_trials['context_name'].values
                    elif predict=='vis_appropriate_response':
                        labels=is_vis_appropriate_response[middle_4_blocks]

                    if crossval=='5_fold_constant':
                        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=165482)
                        train_test_split = skf.split(np.zeros(len(labels)), labels)
                    else:
                        train_test_split=None

                    if input_data_type=='spikes':
                        shifted_trial_da = trial_da.sel(trials=middle_4_blocks+shift,unit_id=sel_units).mean(dim='time').values
                        input_data=shifted_trial_da.T

                    elif input_data_type=='facemap' or input_data_type == 'LP':
                        trials_used=middle_4_blocks+shift
                        shift_exists=[]
                        for tt in trials_used:
                            if tt<mean_trial_behav_SVD[aa].shape[1]:
                                shift_exists.append(True)
                            else:
                                shift_exists.append(False)
                        shift_exists=np.array(shift_exists)
                        trials_used=trials_used[shift_exists]

                        SVD=mean_trial_behav_SVD[aa][:,trials_used]
                        input_data=SVD.T

                        if np.sum(np.isnan(input_data))>0:
                            incl_inds=~np.isnan(input_data).any(axis=1)
                            input_data=input_data[incl_inds,:]
                            labels=labels[incl_inds]

                    decoder_results[session_id]['results'][aa]['shift'][nunits][rr][sh] = decoder_helper(
                        input_data=input_data,
                        labels=labels,
                        decoder_type=decoder_type,
                        crossval=crossval,
                        crossval_index=None,
                        labels_as_index=labels_as_index,
                        train_test_split_input=train_test_split,
                        regularization=regularization,
                        penalty=penalty,
                        solver=solver)


        logger.info(f'{session_id} | area: {aa} | Finished decoding')
    logger.info(f'{session_id} | Finished all decoding')

    #save results
    if session_id not in filename:
        filename=session_id+'_'+filename
    path = upath.UPath(savepath, filename)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f'{session_id} | Saving raw decoding results to {path}')
    path.write_bytes(
        pickle.dumps(decoder_results, protocol=pickle.HIGHEST_PROTOCOL)
    )
    if use_zarr:
        logger.warning('use_zarr not implemented for raw decoding results - saved as .pkl')
        ### too many incompatible data types to save as zarr
        # Create a Zarr group
        # zarr_file = zarr.open(upath.UPath(savepath) /  (filename + '.zarr'), mode='w')
        # dump_dict_to_zarr(group=zarr_file, data=decoder_results)

    # logger.info(f'time elapsed: {time.time()-start_time}')
    for var in ('trial_da','units','trials'):
        if var in locals():
            del var

    gc.collect()
    if return_results:
        return decoder_results


def decode_stimulus_across_context(session=None,params=None,trials=None,units=None,session_info=None):

    decoder_results={}

    input_data_type=params['input_data_type']
    # central_section=params['central_section']
    exclude_cue_trials=params['exclude_cue_trials']
    n_unit_threshold=params['n_unit_threshold']

    n_units_input=params['n_units']
    n_repeats=params['n_repeats']
    spikes_binsize=params['spikes_binsize']
    spikes_time_before=params['spikes_time_before']
    spikes_time_after=params['spikes_time_after']

    savepath=params['savepath']
    filename=params['filename']
    # crossval=params['crossval']
    labels_as_index=params['labels_as_index']
    decoder_type=params['decoder_type']

    if 'regularization' in params:
        regularization=params['regularization']
    else:
        regularization=None
    if 'penalty' in params:
        penalty=params['penalty']
    else:
        penalty=None
    if 'solver' in params:
        solver=params['solver']
    else:
        solver=None

    if 'only_use_all_units' in params:
        only_use_all_units=params['only_use_all_units']
    else:
        only_use_all_units=False

    if 'return_results' in params:
        return_results=params['return_results']
    else:
        return_results=False

    if 'split_area_by_probe' in params:
        split_area_by_probe=params['split_area_by_probe']
    else:
        split_area_by_probe=1

    if session is not None:
        session_id = session.session_id
        if session_info is None:
            session_info = npc_lims.get_session_info(session_id)
    elif session_info is not None:
        session_id=str(session_info.id)
    logger.info(f'{session_id} | Parameters parsed. Starting decoding analysis')

    if trials is None and session is not None:
        trials = data_utils.load_trials_or_units(session, 'trials')
    elif trials is None:
        trials=pd.read_parquet(
            npc_lims.get_cache_path('trials',session_id)
        )

    trials_original_index=trials.index.values

    if exclude_cue_trials:
        trials=trials.query('is_reward_scheduled==False')
        trials_original_index=trials.index.values
        trials=trials.reset_index()

    #make data array
    if units is None and session is not None:
        units = data_utils.load_trials_or_units(session, 'units')
    elif units is None:
        units=pd.read_parquet(
            npc_lims.get_cache_path('units',session_id)
        )

    #add probe to structure name
    if split_area_by_probe==1:
        structure_probe=spike_utils.get_structure_probe(units)
        for uu, unit in units.iterrows():
            units.loc[units['unit_id']==unit['unit_id'],'structure']=structure_probe.loc[structure_probe['unit_id']==unit['unit_id'],'structure_probe']

    #make trial data array for baseline activity
    trial_da = spike_utils.make_neuron_time_trials_tensor(units, trials, spikes_time_before, spikes_time_after, spikes_binsize)

    #make fake blocks for templeton sessions
    if 'Templeton' in session_info.project:
        start_time=trials['start_time'].iloc[0]
        fake_context=np.full(len(trials), fill_value='nan')
        fake_block_nums=np.full(len(trials), fill_value=np.nan)
        block_context_names=['vis','aud']

        if np.random.choice(block_context_names,1)=='vis':
            block_contexts=['vis','aud','vis','aud','vis','aud']
        else:
            block_contexts=['aud','vis','aud','vis','aud','vis']

        for block in range(0,6):
            block_start_time=start_time+block*10*60
            block_end_time=start_time+(block+1)*10*60
            block_trials=trials[:].query('start_time>=@block_start_time').index
            fake_context[block_trials]=block_contexts[block]
            fake_block_nums[block_trials]=block
        trials['block_index']=fake_block_nums
        trials['context_name']=fake_context

    decoder_results[session_id]={}
    decoder_results[session_id]['trials_original_index']=trials_original_index
    decoder_results[session_id]['spikes_binsize'] = spikes_binsize
    decoder_results[session_id]['spikes_time_before'] = spikes_time_before
    decoder_results[session_id]['spikes_time_after'] = spikes_time_after
    decoder_results[session_id]['input_data_type'] = input_data_type
    decoder_results[session_id]['n_units'] = n_units_input
    decoder_results[session_id]['n_repeats'] = n_repeats
    decoder_results[session_id]['labels_as_index'] = labels_as_index
    decoder_results[session_id]['decoder_type'] = decoder_type
    decoder_results[session_id]['regularization'] = regularization
    decoder_results[session_id]['penalty'] = penalty
    decoder_results[session_id]['solver'] = solver
    decoder_results[session_id]['only_use_all_units'] = only_use_all_units

    import dataclasses
    session_info=dataclasses.asdict(session_info)
    for ii in session_info.keys():
        if type(session_info[ii]) not in [int, str, bool, dict, list]:
            session_info[ii]=str(session_info[ii])

    decoder_results[session_id]['session_info'] = session_info

    if only_use_all_units:
        areas=['all']
    else:
        areas=units['structure'].unique()
        areas=np.concatenate([areas,['all']])

        #add non-probe-specific area to areas
        all_probe_areas=[]
        if len(units.query('structure.str.contains("probe")'))>0:
            probe_areas=units.query('structure.str.contains("probe")')['structure'].unique()
            for pa in probe_areas:
                all_probe_areas.append([pa.split('_')[0]+'_all'])

        general_areas=np.unique(np.array(all_probe_areas))
        areas=np.concatenate([areas,general_areas])

        #consolidate SC areas
        for aa in areas:
            if aa in ['SCop','SCsg','SCzo']:
                if 'SCs' not in areas:
                    areas=np.concatenate([areas,['SCs']])
            elif aa in ['SCig','SCiw','SCdg','SCdw']:
                if 'SCm' not in areas:
                    areas=np.concatenate([areas,['SCm']])

    decoder_results[session_id]['areas'] = areas
    decoder_results[session_id]['results'] = {}

    for aa in areas:
        if aa == 'all':
            area_units=units
        elif '_all' in aa:
            temp_area=aa.split('_')[0]
            possible_probe_areas=[temp_area+'_probeA',temp_area+'_probeB',temp_area+'_probeC',
                                temp_area+'_probeD',temp_area+'_probeE',temp_area+'_probeF']
            area_units=units.query('structure in @possible_probe_areas')
        elif aa=='SCs':
            area_units=units.query('structure=="SCop" or structure=="SCsg" or structure=="SCzo"')
        elif aa=='SCm':
            area_units=units.query('structure=="SCig" or structure=="SCiw" or structure=="SCdg" or structure=="SCdw"')
        else:
            area_units=units.query('structure==@aa')

        n_units=len(area_units)
        if n_units<n_unit_threshold:
            continue

        area_unit_ids=area_units['unit_id'].values

        decoder_results[session_id]['results'][aa]={}

        decoder_results[session_id]['results'][aa]['predict_vis_stim_vis_context']={}
        decoder_results[session_id]['results'][aa]['predict_vis_stim_aud_context']={}
        decoder_results[session_id]['results'][aa]['predict_aud_stim_aud_context']={}
        decoder_results[session_id]['results'][aa]['predict_aud_stim_vis_context']={}
        decoder_results[session_id]['results'][aa]['unit_ids']={}
        decoder_results[session_id]['results'][aa]['n_units']={}
        decoder_results[session_id]['results'][aa]['unit_ids']=area_units['unit_id'].values
        decoder_results[session_id]['results'][aa]['n_units']=len(area_units)

        #find mean ccf location of units
        decoder_results[session_id]['results'][aa]['ccf_ap_mean']=area_units['ccf_ap'].mean()
        decoder_results[session_id]['results'][aa]['ccf_dv_mean']=area_units['ccf_dv'].mean()
        decoder_results[session_id]['results'][aa]['ccf_ml_mean']=area_units['ccf_ml'].mean()

        #loop through repeats
        for nunits in n_units_input:
            if nunits!='all' and nunits>len(area_units):
                continue
            decoder_results[session_id]['results'][aa]['predict_vis_stim_vis_context'][nunits]={}
            decoder_results[session_id]['results'][aa]['predict_vis_stim_aud_context'][nunits]={}
            decoder_results[session_id]['results'][aa]['predict_aud_stim_aud_context'][nunits]={}
            decoder_results[session_id]['results'][aa]['predict_aud_stim_vis_context'][nunits]={}

            for rr in range(n_repeats):
                decoder_results[session_id]['results'][aa]['predict_vis_stim_vis_context'][nunits][rr]={}
                decoder_results[session_id]['results'][aa]['predict_vis_stim_aud_context'][nunits][rr]={}
                decoder_results[session_id]['results'][aa]['predict_aud_stim_aud_context'][nunits][rr]={}
                decoder_results[session_id]['results'][aa]['predict_aud_stim_vis_context'][nunits][rr]={}

                if nunits=='all':
                    sel_units=area_unit_ids
                else:
                    sel_units=np.random.choice(area_unit_ids,nunits,replace=False)

                decoder_results[session_id]['results'][aa]['predict_vis_stim_vis_context'][nunits][rr]['sel_units']=sel_units
                decoder_results[session_id]['results'][aa]['predict_vis_stim_aud_context'][nunits][rr]['sel_units']=sel_units
                decoder_results[session_id]['results'][aa]['predict_aud_stim_aud_context'][nunits][rr]['sel_units']=sel_units
                decoder_results[session_id]['results'][aa]['predict_aud_stim_vis_context'][nunits][rr]['sel_units']=sel_units

                #decode visual stimuli in visual context, 5-fold crossval
                vis_stim_vis_context_trials=trials.query('is_vis_stim and is_vis_context')
                labels=vis_stim_vis_context_trials['stim_name'].values
                input_data=trial_da.sel(trials=vis_stim_vis_context_trials.index,unit_id=sel_units).mean(dim='time').values.T

                decoder_results[session_id]['results'][aa]['predict_vis_stim_vis_context']['trial_idx']=vis_stim_vis_context_trials.index.values
                decoder_results[session_id]['results'][aa]['predict_vis_stim_vis_context'][nunits][rr] = decoder_helper(
                    input_data=input_data,
                    labels=labels,
                    decoder_type=decoder_type,
                    crossval='5_fold',
                    crossval_index=None,
                    labels_as_index=labels_as_index,
                    regularization=regularization,
                    penalty=penalty,
                    solver=solver)

                #decode visual stimuli in auditory context, train on vis context, test on aud context
                #also split vis stim aud context trials into 5 folds;
                #train on vis context train set fold, test on aud context test set folds
                vis_stim_trials=trials.query('is_vis_stim')
                vis_stim_trial_original_index=vis_stim_trials.index.values
                vis_stim_trials=vis_stim_trials.reset_index()

                labels=vis_stim_trials['stim_name'].values
                input_data=trial_da.sel(trials=vis_stim_trials.index,unit_id=sel_units).mean(dim='time').values.T

                vis_stim_vis_context_trials=vis_stim_trials.query('is_vis_stim and is_vis_context')
                skf = StratifiedKFold(n_splits=5,shuffle=True)
                train_test_split_vis_stim_vis_context = skf.split(np.zeros(len(vis_stim_vis_context_trials)), vis_stim_vis_context_trials['stim_name'])

                vis_stim_aud_context_trials=vis_stim_trials.query('is_vis_stim and is_aud_context')
                skf = StratifiedKFold(n_splits=5,shuffle=True)
                train_test_split_vis_stim_aud_context = skf.split(np.zeros(len(vis_stim_aud_context_trials)), vis_stim_aud_context_trials['stim_name'])

                train=[]
                for train_index, test_index in train_test_split_vis_stim_vis_context:
                    train.append(vis_stim_vis_context_trials.index.values[train_index])
                test=[]
                for train_index, test_index in train_test_split_vis_stim_aud_context:
                    test.append(vis_stim_aud_context_trials.index.values[test_index])

                # train=[vis_stim_trials.reset_index().query('is_vis_context').index.values]
                # test=[vis_stim_trials.reset_index().query('is_aud_context').index.values]

                train_test_split=zip(train,test)
                decoder_results[session_id]['results'][aa]['predict_vis_stim_aud_context']['trial_idx']=vis_stim_trial_original_index
                decoder_results[session_id]['results'][aa]['predict_vis_stim_aud_context'][nunits][rr] = decoder_helper(
                    input_data=input_data,
                    labels=labels,
                    decoder_type=decoder_type,
                    crossval='5_fold_constant',
                    crossval_index=None,
                    labels_as_index=labels_as_index,
                    train_test_split_input=train_test_split,
                    regularization=regularization,
                    penalty=penalty,
                    solver=solver)

                #decode auditory stimuli in auditory context, 5-fold crossval
                aud_stim_aud_context_trials=trials.query('is_aud_stim and is_aud_context')
                labels=aud_stim_aud_context_trials['stim_name'].values
                input_data=trial_da.sel(trials=aud_stim_aud_context_trials.index,unit_id=sel_units).mean(dim='time').values.T

                decoder_results[session_id]['results'][aa]['predict_aud_stim_aud_context']['trial_idx']=aud_stim_aud_context_trials.index.values
                decoder_results[session_id]['results'][aa]['predict_aud_stim_aud_context'][nunits][rr] = decoder_helper(
                    input_data=input_data,
                    labels=labels,
                    decoder_type=decoder_type,
                    crossval='5_fold',
                    crossval_index=None,
                    labels_as_index=labels_as_index,
                    regularization=regularization,
                    penalty=penalty,
                    solver=solver)

                #decode auditory stimuli in visual context, train on aud context, test on vis context
                aud_stim_trials=trials.query('is_aud_stim')
                aud_stim_trial_original_index=aud_stim_trials.index.values
                aud_stim_trials=aud_stim_trials.reset_index()
                labels=aud_stim_trials['stim_name'].values
                input_data=trial_da.sel(trials=aud_stim_trials.index,unit_id=sel_units).mean(dim='time').values.T

                aud_stim_aud_context_trials=aud_stim_trials.query('is_aud_stim and is_aud_context')
                skf = StratifiedKFold(n_splits=5,shuffle=True)
                train_test_split_aud_stim_aud_context = skf.split(np.zeros(len(aud_stim_aud_context_trials)), aud_stim_aud_context_trials['stim_name'])

                aud_stim_vis_context_trials=aud_stim_trials.query('is_aud_stim and is_vis_context')
                skf = StratifiedKFold(n_splits=5,shuffle=True)
                train_test_split_aud_stim_vis_context = skf.split(np.zeros(len(aud_stim_vis_context_trials)), aud_stim_vis_context_trials['stim_name'])

                train=[]
                for train_index, test_index in train_test_split_aud_stim_aud_context:
                    train.append(aud_stim_aud_context_trials.index.values[train_index])
                test=[]
                for train_index, test_index in train_test_split_aud_stim_vis_context:
                    test.append(aud_stim_vis_context_trials.index.values[test_index])

                # train=[aud_stim_trials.reset_index().query('is_aud_context').index.values]
                # test=[aud_stim_trials.reset_index().query('is_vis_context').index.values]

                train_test_split=zip(train,test)
                decoder_results[session_id]['results'][aa]['predict_aud_stim_vis_context']['trial_idx']=aud_stim_trial_original_index
                decoder_results[session_id]['results'][aa]['predict_aud_stim_vis_context'][nunits][rr] = decoder_helper(
                    input_data=input_data,
                    labels=labels,
                    decoder_type=decoder_type,
                    crossval='5_fold_constant',
                    crossval_index=None,
                    labels_as_index=labels_as_index,
                    train_test_split_input=train_test_split,
                    regularization=regularization,
                    penalty=penalty,
                    solver=solver)

                if nunits=='all':
                    break

        logger.info(f'{session_id} | area: {aa} | Finished decoding')
    logger.info(f'{session_id} | Finished all decoding')

    #save results
    if session_id not in filename:
        filename=session_id+'_'+filename
    path = upath.UPath(savepath, filename)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f'{session_id} | Saving raw decoding results to {path}')
    path.write_bytes(
        pickle.dumps(decoder_results, protocol=pickle.HIGHEST_PROTOCOL)
    )

    del trial_da
    del units
    del trials
    gc.collect()
    if return_results:
        return decoder_results

    if return_results:
        return decoder_results


def concat_decoder_results(files,savepath=None,return_table=True,single_session=False,use_zarr=False):

    logger.info(f'Making decoder analysis summary tables with input arguments: {locals()}') # keep this on first line of function

    use_half_shifts=False
    n_repeats=25

    all_bal_acc={}
    all_trials_bal_acc={}

    linear_shift_dict={
        'session_id':[],
        'project':[],
        'area':[],

        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
        'probe':[],
        'cross_modal_dprime':[],
        'n_good_blocks':[],
    }

    if type(files) is not list:
        files=[files]
    #assume first file has same nunits as all others
    decoder_results=pickle.loads(upath.UPath(files[0]).read_bytes())
    session_id=list(decoder_results.keys())[0]
    nunits_global=decoder_results[session_id]['n_units']

    for nu in nunits_global:
        linear_shift_dict['true_accuracy_'+str(nu)]=[]
        linear_shift_dict['null_accuracy_mean_'+str(nu)]=[]
        linear_shift_dict['null_accuracy_median_'+str(nu)]=[]
        linear_shift_dict['null_accuracy_std_'+str(nu)]=[]
        linear_shift_dict['p_value_'+str(nu)]=[]
        linear_shift_dict['true_accuracy_all_trials_no_shift_'+str(nu)]=[]

    #loop through sessions
    for file in files:
        decoder_results=pickle.loads(upath.UPath(file).read_bytes())
        session_id=str(list(decoder_results.keys())[0])
        session_info=npc_lims.get_session_info(session_id)
        project=str(session_info.project)
        logger.info('loading session: '+session_id)
        try:
            performance=pd.read_parquet(
                        npc_lims.get_cache_path('performance',session_info.id,version='any')
                    )
        except:
            logger.info('no cached performance table, skipping')
            continue

        if session_info.is_annotated==False:
            logger.info('session not annotated, skipping')
            continue

        all_bal_acc[session_id]={}
        all_trials_bal_acc[session_id]={}

        nunits=decoder_results[session_id]['n_units']
        if nunits!=nunits_global:
            logger.info('WARNING, session '+session_id+' has different n_units; skipping')
            continue

        shifts=decoder_results[session_id]['shifts']
        #extract results according to the trial shift
        half_neg_shift=np.ceil(shifts.min()/2)
        half_pos_shift=np.ceil(shifts.max()/2)
        # half_shifts=np.arange(-half_neg_shift,half_pos_shift+1)
        half_neg_shift_ind=np.where(shifts==half_neg_shift)[0][0]
        half_pos_shift_ind=np.where(shifts==half_pos_shift)[0][0]
        half_shift_inds=np.arange(half_neg_shift_ind,half_pos_shift_ind+1)

        all_bal_acc[session_id]['shifts']=shifts
        all_bal_acc[session_id]['half_shift_inds']=half_shift_inds
        if use_half_shifts:
            half_shifts=shifts[half_shift_inds]
        else:
            half_shifts=shifts

        half_shift_inds=np.arange(len(half_shifts))

        areas=list(decoder_results[session_id]['results'].keys())

        #TODO: add decoder accuracy using all trials (no shift)

        #save balanced accuracy by shift
        for aa in areas:
            if aa in decoder_results[session_id]['results']:
                all_bal_acc[session_id][aa]={}
                all_trials_bal_acc[session_id][aa]={}
                ### ADD LOOP FOR NUNITS ###
                for nu in nunits:
                    if nu not in decoder_results[session_id]['results'][aa]['shift'].keys():
                        continue
                    all_bal_acc[session_id][aa][nu]=[]
                    all_trials_bal_acc[session_id][aa][nu]=[]
                    for rr in range(n_repeats):
                        if rr in decoder_results[session_id]['results'][aa]['shift'][nu].keys():
                            temp_bal_acc=[]
                        # else:
                        #     logger.info('n repeats invalid: '+str(rr))
                        #     continue
                            for sh in half_shift_inds:
                                if sh in list(decoder_results[session_id]['results'][aa]['shift'][nu][rr].keys()):
                                    temp_bal_acc.append(decoder_results[session_id]['results'][aa]['shift'][nu][rr][sh]['balanced_accuracy_test'])

                            if len(temp_bal_acc)>0:
                                all_bal_acc[session_id][aa][nu].append(np.array(temp_bal_acc))

                            all_trials_bal_acc[session_id][aa][nu].append(decoder_results[session_id]['results'][aa]['no_shift'][nu][rr]['balanced_accuracy_test'])

                    all_bal_acc[session_id][aa][nu]=np.vstack(all_bal_acc[session_id][aa][nu])
                    all_bal_acc[session_id][aa][nu]=np.nanmean(all_bal_acc[session_id][aa][nu],axis=0)

                    all_trials_bal_acc[session_id][aa][nu]=np.nanmean(all_trials_bal_acc[session_id][aa][nu])

                if type(aa)==str:
                    if '_probe' in aa:
                        area_name=aa.split('_probe')[0]
                        probe_name=aa.split('_probe')[1]
                    elif '_all' in aa:
                        area_name=aa.split('_all')[0]
                        probe_name='all'
                    else:
                        area_name=aa
                        probe_name=''
                else:
                    area_name=aa
                    probe_name=''

                ### LOOP THROUGH NUNITS TO APPEND TO DICT ###

                for nu in nunits:
                    if nu in all_bal_acc[session_id][aa].keys():

                        true_acc_ind=np.where(half_shifts==0)[0][0]
                        null_acc_ind=np.where(half_shifts!=0)[0]
                        true_accuracy=all_bal_acc[session_id][aa][nu][true_acc_ind]
                        null_accuracy_mean=np.mean(all_bal_acc[session_id][aa][nu][null_acc_ind])
                        null_accuracy_median=np.median(all_bal_acc[session_id][aa][nu][null_acc_ind])
                        null_accuracy_std=np.std(all_bal_acc[session_id][aa][nu][null_acc_ind])
                        p_value=np.mean(all_bal_acc[session_id][aa][nu][null_acc_ind]>=true_accuracy)

                        linear_shift_dict['true_accuracy_'+str(nu)].append(true_accuracy)
                        linear_shift_dict['null_accuracy_mean_'+str(nu)].append(null_accuracy_mean)
                        linear_shift_dict['null_accuracy_median_'+str(nu)].append(null_accuracy_median)
                        linear_shift_dict['null_accuracy_std_'+str(nu)].append(null_accuracy_std)
                        linear_shift_dict['p_value_'+str(nu)].append(p_value)

                    else:
                        linear_shift_dict['true_accuracy_'+str(nu)].append(np.nan)
                        linear_shift_dict['null_accuracy_mean_'+str(nu)].append(np.nan)
                        linear_shift_dict['null_accuracy_median_'+str(nu)].append(np.nan)
                        linear_shift_dict['null_accuracy_std_'+str(nu)].append(np.nan)
                        linear_shift_dict['p_value_'+str(nu)].append(np.nan)

                    if nu in all_trials_bal_acc[session_id][aa].keys():
                        true_accuracy=all_trials_bal_acc[session_id][aa][nu]
                        linear_shift_dict['true_accuracy_all_trials_no_shift_'+str(nu)].append(true_accuracy)
                    else:
                        linear_shift_dict['true_accuracy_all_trials_no_shift_'+str(nu)].append(np.nan)

                #make big dict/dataframe for this:
                #save true decoding, mean/median null decoding, and p value for each area/probe
                linear_shift_dict['session_id'].append(session_id)
                linear_shift_dict['project'].append(project)
                linear_shift_dict['area'].append(area_name)
                linear_shift_dict['cross_modal_dprime'].append(performance['cross_modal_dprime'].mean())
                linear_shift_dict['n_good_blocks'].append(np.sum(performance['cross_modal_dprime']>=1.0))

                # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
                if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                    linear_shift_dict['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                    linear_shift_dict['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                    linear_shift_dict['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])
                    linear_shift_dict['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])
                    linear_shift_dict['probe'].append(probe_name)
                else:
                    linear_shift_dict['ccf_ap_mean'].append(np.nan)
                    linear_shift_dict['ccf_dv_mean'].append(np.nan)
                    linear_shift_dict['ccf_ml_mean'].append(np.nan)
                    linear_shift_dict['n_units'].append(np.nan)
                    linear_shift_dict['probe'].append(np.nan)

        logger.info(f"{session_id} | area: {aa} | Finished")
    logger.info(f"{session_id} | Finished concatenating decoding results")

    linear_shift_df=pd.DataFrame(linear_shift_dict)

    if use_zarr==True:

        results={
            session_id:{
                'linear_shift_summary_table':linear_shift_dict,
                },
        }
        path = files[0] # TODO @egmcbride - shouldn't this be `savepath`?
        logger.info(f'Saving concatenated decoder results to zarr file: {path}')
        zarr_file = zarr.open(path, mode='w')

        dump_dict_to_zarr(zarr_file, results)

    elif use_zarr==False:
        if savepath is not None:
            if not upath.UPath(savepath).is_dir():
                upath.UPath(savepath).mkdir(parents=True)

            logger.info(f'Saving decoder results table to: {savepath}')
            if single_session:
                linear_shift_df.to_csv(upath.UPath(savepath / (session_id+'_linear_shift_decoding_results.csv')))
            else:
                linear_shift_df.to_csv(upath.UPath(savepath / 'all_linear_shift_decoding_results.csv'))

    del decoder_results
    gc.collect()

    if return_table:
        return linear_shift_df



def compute_significant_decoding_by_area(all_decoder_results,combine_areas_across_probes=True):

    #determine different numbers of units from all_decoder_results
    n_units=[]
    for col in all_decoder_results.filter(like='true_accuracy').columns.values:
        if 'no_shift' in col:
            continue
        if len(col.split('_'))>2:
            n_units.append('_'+col.split('_')[2])
        else:
            n_units.append('')

    #compare DR and Templeton:
    p_threshold=0.05

    DR_linear_shift_df=all_decoder_results.query('project=="DynamicRouting" and n_good_blocks>=4')

    if combine_areas_across_probes:
        if 'all' in DR_linear_shift_df['probe'].unique():
            DR_linear_shift_df=DR_linear_shift_df.query('probe.isna() or probe == "all"')
        else:
            print('ERROR: no combined probes in results')
            return
    else:
        DR_linear_shift_df=DR_linear_shift_df.query('probe != "all"')


    #fraction significant
    frac_sig_DR={
        'area':[],
        'n_expts_DR':[],
    }
    for nu in n_units:
        frac_sig_DR['frac_sig_DR'+nu]=[]

    for area in DR_linear_shift_df['area'].unique():
        frac_sig_DR['area'].append(area)
        frac_sig_DR['n_expts_DR'].append(len(DR_linear_shift_df.query('area==@area')))
        for nu in n_units:
            frac_sig_DR['frac_sig_DR'+nu].append(np.mean(DR_linear_shift_df.query('area==@area')['p_value'+nu]<p_threshold))

    frac_sig_DR_df=pd.DataFrame(frac_sig_DR)
    #diff from null
    diff_from_null_DR={
        'area':[],
        'n_expts_DR':[],
    }
    for nu in n_units:
        diff_from_null_DR['diff_from_null_mean_DR'+nu]=[]
        diff_from_null_DR['diff_from_null_median_DR'+nu]=[]
        diff_from_null_DR['diff_from_null_sem_DR'+nu]=[]
        diff_from_null_DR['true_accuracy_DR'+nu]=[]
        diff_from_null_DR['true_accuracy_sem_DR'+nu]=[]
        diff_from_null_DR['null_median_DR'+nu]=[]
        diff_from_null_DR['null_median_sem_DR'+nu]=[]


    for area in DR_linear_shift_df['area'].unique():
        diff_from_null_DR['area'].append(area)
        diff_from_null_DR['n_expts_DR'].append(len(DR_linear_shift_df.query('area==@area')))
        for nu in n_units:
            diff_from_null_DR['diff_from_null_mean_DR'+nu].append((DR_linear_shift_df.query('area==@area')['true_accuracy'+nu]-
                                                        DR_linear_shift_df.query('area==@area')['null_accuracy_median'+nu]).mean())
            diff_from_null_DR['diff_from_null_median_DR'+nu].append((DR_linear_shift_df.query('area==@area')['true_accuracy'+nu]-
                                                            DR_linear_shift_df.query('area==@area')['null_accuracy_median'+nu]).median())
            diff_from_null_DR['diff_from_null_sem_DR'+nu].append((DR_linear_shift_df.query('area==@area')['true_accuracy'+nu]-
                                                                  DR_linear_shift_df.query('area==@area')['null_accuracy_median'+nu]).sem())
            diff_from_null_DR['true_accuracy_DR'+nu].append(DR_linear_shift_df.query('area==@area')['true_accuracy'+nu].median())
            diff_from_null_DR['true_accuracy_sem_DR'+nu].append(DR_linear_shift_df.query('area==@area')['true_accuracy'+nu].sem())
            diff_from_null_DR['null_median_DR'+nu].append(DR_linear_shift_df.query('area==@area')['null_accuracy_median'+nu].median())
            diff_from_null_DR['null_median_sem_DR'+nu].append(DR_linear_shift_df.query('area==@area')['null_accuracy_median'+nu].sem())

    diff_from_null_DR_df=pd.DataFrame(diff_from_null_DR)

    Templeton_linear_shift_df=all_decoder_results.query('project.str.contains("Templeton")')

    #fraction significant
    frac_sig_Templ={
        'area':[],
        'n_expts_Templ':[],
    }

    for nu in n_units:
        frac_sig_Templ['frac_sig_Templ'+nu]=[]
    for area in Templeton_linear_shift_df['area'].unique():
        frac_sig_Templ['area'].append(area)
        frac_sig_Templ['n_expts_Templ'].append(len(Templeton_linear_shift_df.query('area==@area')))
        for nu in n_units:
            frac_sig_Templ['frac_sig_Templ'+nu].append(np.mean(Templeton_linear_shift_df.query('area==@area')['p_value'+nu]<p_threshold))

    frac_sig_Templ_df=pd.DataFrame(frac_sig_Templ)
    #diff from null
    diff_from_null_Templ={
        'area':[],
        'n_expts_Templ':[],
    }
    for nu in n_units:
        diff_from_null_Templ['diff_from_null_mean_Templ'+nu]=[]
        diff_from_null_Templ['diff_from_null_median_Templ'+nu]=[]
        diff_from_null_Templ['diff_from_null_sem_Templ'+nu]=[]
        diff_from_null_Templ['true_accuracy_Templ'+nu]=[]
        diff_from_null_Templ['true_accuracy_sem_Templ'+nu]=[]
        diff_from_null_Templ['null_median_Templ'+nu]=[]
        diff_from_null_Templ['null_median_sem_Templ'+nu]=[]

    for area in Templeton_linear_shift_df['area'].unique():
        diff_from_null_Templ['area'].append(area)
        diff_from_null_Templ['n_expts_Templ'].append(len(Templeton_linear_shift_df.query('area==@area')))
        for nu in n_units:
            diff_from_null_Templ['diff_from_null_mean_Templ'+nu].append((Templeton_linear_shift_df.query('area==@area')['true_accuracy'+nu]-
                                                        Templeton_linear_shift_df.query('area==@area')['null_accuracy_mean'+nu]).mean())
            diff_from_null_Templ['diff_from_null_median_Templ'+nu].append((Templeton_linear_shift_df.query('area==@area')['true_accuracy'+nu]-
                                                            Templeton_linear_shift_df.query('area==@area')['null_accuracy_median'+nu]).median())
            diff_from_null_Templ['diff_from_null_sem_Templ'+nu].append((Templeton_linear_shift_df.query('area==@area')['true_accuracy'+nu]-
                                                                    Templeton_linear_shift_df.query('area==@area')['null_accuracy_mean'+nu]).sem())
            diff_from_null_Templ['true_accuracy_Templ'+nu].append(Templeton_linear_shift_df.query('area==@area')['true_accuracy'+nu].median())
            diff_from_null_Templ['true_accuracy_sem_Templ'+nu].append(Templeton_linear_shift_df.query('area==@area')['true_accuracy'+nu].sem())
            diff_from_null_Templ['null_median_Templ'+nu].append(Templeton_linear_shift_df.query('area==@area')['null_accuracy_median'+nu].median())
            diff_from_null_Templ['null_median_sem_Templ'+nu].append(Templeton_linear_shift_df.query('area==@area')['null_accuracy_median'+nu].sem())

    diff_from_null_Templ_df=pd.DataFrame(diff_from_null_Templ)

    all_frac_sig_df=pd.merge(frac_sig_DR_df,frac_sig_Templ_df,on='area',how='outer')
    all_diff_from_null_df=pd.merge(diff_from_null_DR_df,diff_from_null_Templ_df,on='area',how='outer')

    return all_frac_sig_df,all_diff_from_null_df


def concat_trialwise_decoder_results(files,savepath=None,return_table=False,n_units=None,single_session=False,use_zarr=False):

    logger.info(f'Making trialwise decoder analysis summary tables with input arguments {locals()}') # keep this on first line of function

    #load sessions as we go
    use_half_shifts=False
    n_repeats=25

    decoder_confidence_versus_response_type={
        'session':[],
        'area':[],
        'project':[],
        'probe':[],
        'vis_context_dprime':[],
        'aud_context_dprime':[],
        'overall_dprime':[],
        'n_good_blocks':[],

        'vis_hit_confidence':[],
        'vis_fa_confidence':[],
        'vis_cr_confidence':[],
        'aud_hit_confidence':[],
        'aud_fa_confidence':[],
        'aud_cr_confidence':[],
        'correct_confidence':[],
        'incorrect_confidence':[],
        'cr_all_confidence':[],
        'fa_all_confidence':[],
        'hit_all_confidence':[],

        'vis_hit_null_confidence':[],
        'vis_fa_null_confidence':[],
        'vis_cr_null_confidence':[],
        'aud_hit_null_confidence':[],
        'aud_fa_null_confidence':[],
        'aud_cr_null_confidence':[],
        'correct_null_confidence':[],
        'incorrect_null_confidence':[],
        'cr_all_null_confidence':[],
        'fa_all_null_confidence':[],
        'hit_all_null_confidence':[],

        'vis_hit_predict_proba':[],
        'vis_fa_predict_proba':[],
        'vis_cr_predict_proba':[],
        'aud_hit_predict_proba':[],
        'aud_fa_predict_proba':[],
        'aud_cr_predict_proba':[],
        'correct_predict_proba':[],
        'incorrect_predict_proba':[],
        'cr_all_predict_proba':[],
        'fa_all_predict_proba':[],
        'hit_all_predict_proba':[],

        'vis_hit_null_predict_proba':[],
        'vis_fa_null_predict_proba':[],
        'vis_cr_null_predict_proba':[],
        'aud_hit_null_predict_proba':[],
        'aud_fa_null_predict_proba':[],
        'aud_cr_null_predict_proba':[],
        'correct_null_predict_proba':[],
        'incorrect_null_predict_proba':[],
        'cr_all_null_predict_proba':[],
        'fa_all_null_predict_proba':[],
        'hit_all_null_predict_proba':[],

        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
    }

    decoder_confidence_dprime_by_block={
        'session':[],
        'area':[],
        'project':[],
        'probe':[],
        'block':[],
        'cross_modal_dprime':[],
        'n_good_blocks':[],
        'confidence':[],
        'null_confidence':[],
        'null_min_confidence':[],
        'predict_proba':[],
        'predict_proba_null':[],
        'predict_proba_null_min':[],
        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
    }

    decoder_confidence_by_switch={
        'session':[],
        'area':[],
        'project':[],
        'probe':[],
        'switch_trial':[],
        'block':[],
        'dprime_before':[],
        'dprime_after':[],
        'confidence':[],
        'null_confidence':[],
        'null_min_confidence':[],
        'predict_proba':[],
        'predict_proba_null':[],
        'predict_proba_null_min':[],
        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
    }

    decoder_confidence_versus_trials_since_rewarded_target={
        'session':[],
        'area':[],
        'project':[],
        'probe':[],
        'trial_index':[],
        'trials_since_rewarded_target':[],
        'time_since_rewarded_target':[],
        'trials_since_last_information':[],
        'time_since_last_information':[],
        'trials_since_last_information_no_targets':[],
        'time_since_last_information_no_targets':[],
        'confidence':[],
        'confidence_null':[],
        'confidence_null_min':[],
        'predict_proba':[],
        'predict_proba_null':[],
        'predict_proba_null_min':[],
        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
        'cross_modal_dprime':[],
        'n_good_blocks':[],
    }

    decoder_confidence_before_after_target={
        'session':[],
        'area':[],
        'project':[],
        'probe':[],
        'cross_modal_dprime':[],
        'n_good_blocks':[],
        'rewarded_target':[],
        'rewarded_target_plus_one':[],
        'non_rewarded_target':[],
        'non_rewarded_target_plus_one':[],
        'non_response_non_rewarded_target':[],
        'non_response_non_rewarded_target_plus_one':[],
        'non_response_non_target_trials':[],
        'non_response_non_target_trials_plus_one':[],
        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
    }

    #TODO: add table with decoder condfidence for all trials, plus other useful session-level information
    decoder_confidence_all_trials={
        'session':[],
        'area':[],
        'project':[],
        'probe':[],
        'cross_modal_dprime':[],
        'n_good_blocks':[],
        'trial_index':[],
        'confidence':[],
        'predict_proba':[],
        'ccf_ap_mean':[],
        'ccf_dv_mean':[],
        'ccf_ml_mean':[],
        'n_units':[],
    }

    start_time=time.time()

    ##loop through sessions##
    if single_session:
        if type(files) is not list:
            files=[files]

    for file in files:

        session_start_time=time.time()
        decoder_results=pickle.loads(upath.UPath(file).read_bytes())

        session_id=list(decoder_results.keys())[0]
        session_info=npc_lims.get_session_info(session_id)
        session_id_str=str(session_id)
        project=str(session_info.project)
        #load session


        try:
            trials=pd.read_parquet(
                    npc_lims.get_cache_path('trials',session_id,version='any')
                    )
            performance=pd.read_parquet(
                    npc_lims.get_cache_path('performance',session_id,version='any')
                    )
        except:
            logger.info(f'{session_id} | trials or performance not available from cache; skipping session')
            continue

        trials_since_rewarded_target=[]
        time_since_rewarded_target=[]
        last_rewarded_time=np.nan
        last_rewarded_trial=np.nan
        trials_since_last_information=[]
        time_since_last_information=[]
        last_informative_trial=np.nan
        last_informative_time=np.nan

        trials_since_last_information_no_targets=[]
        time_since_last_information_no_targets=[]

        non_response_flag=False

        for tt,trial in trials.iterrows():
            #track trials/time since last bit of information, exclude trials after non-responses to targets

            if non_response_flag==True:
                trials_since_last_information_no_targets.append(np.nan)
                time_since_last_information_no_targets.append(np.nan)
            else:
                trials_since_last_information_no_targets.append(tt-last_informative_trial)
                time_since_last_information_no_targets.append(trial['start_time']-last_informative_time)

            trials_since_last_information.append(tt-last_informative_trial)
            time_since_last_information.append(trial['start_time']-last_informative_time)

            #trials/time since last rewarded target
            trials_since_rewarded_target.append(tt-last_rewarded_trial)
            time_since_rewarded_target.append(trial['start_time']-last_rewarded_time)

            if trial['is_target'] and not trial['is_response']:
                non_response_flag=True

            elif trial['is_target'] and trial['is_response']:
                last_informative_time=trial['start_time']
                last_informative_trial=tt
                non_response_flag=False

            if trial['is_rewarded'] and trial['is_target']:
                last_rewarded_time=trial['reward_time']
                last_rewarded_trial=tt


        trials['trials_since_rewarded_target']=trials_since_rewarded_target
        trials['time_since_rewarded_target']=time_since_rewarded_target

        trials['trials_since_last_information']=trials_since_last_information
        trials['time_since_last_information']=time_since_last_information

        trials['trials_since_last_information_no_targets']=trials_since_last_information_no_targets
        trials['time_since_last_information_no_targets']=time_since_last_information_no_targets

        #select the middle 4 blocks
        trials['original_index']=trials.index.values
        trials_middle=trials.iloc[decoder_results[session_id]['middle_4_blocks']]
        trials_middle=trials_middle.reset_index()
        trials_middle.loc[:,'id']=trials_middle.index.values

        areas=list(decoder_results[session_id]['results'].keys())

        ##loop through areas##
        for aa in areas:
            if 'n_units' not in decoder_results[session_id]['results'][aa]['shift'].keys():
                for key in (
                    'ccf_ap_mean',
                    'ccf_dv_mean',
                    'ccf_ml_mean',
                    'n_units',
                ):
                    decoder_confidence_all_trials.pop(key, None)
                continue

            if type(aa)==str:
                if '_probe' in aa:
                    area_name=aa.split('_probe')[0]
                    probe_name=aa.split('_probe')[1]
                elif '_all' in aa:
                    area_name=aa.split('_all')[0]
                    probe_name='all'
                else:
                    area_name=aa
                    probe_name=''
            else:
                area_name=aa
                probe_name=''

            #make corrected decoder confidence
            shifts=decoder_results[session_id]['shifts']
            areas=decoder_results[session_id]['areas']
            half_neg_shift=np.ceil(shifts.min()/2)
            half_pos_shift=np.ceil(shifts.max()/2)
            half_neg_shift_ind=np.where(shifts==half_neg_shift)[0][0]
            half_pos_shift_ind=np.where(shifts==half_pos_shift)[0][0]
            half_shift_inds=np.arange(half_neg_shift_ind,half_pos_shift_ind+1)
            if use_half_shifts==False:
                half_shift_inds=np.arange(len(shifts))

            decision_function_shifts=[]
            predict_proba_shifts=[]

            confidence_all_trials=[]
            predict_proba_all_trials=[]

            for sh in half_shift_inds:
                temp_shifts=[]
                temp_proba_shifts=[]
                for rr in range(n_repeats):
                    if n_units is not None:

                        if n_units=='all' and rr>0:
                            continue
                        if sh in list(decoder_results[session_id]['results'][aa]['shift'][n_units][rr].keys()):
                            temp_shifts.append(decoder_results[session_id]['results'][aa]['shift'][n_units][rr][sh]['decision_function'])
                            temp_proba_shifts.append(decoder_results[session_id]['results'][aa]['shift'][n_units][rr][sh]['predict_proba'][:,1])

                        if sh==0:
                            confidence_all_trials.append(decoder_results[session_id]['results'][aa]['no_shift'][n_units][rr]['decision_function'])
                            predict_proba_all_trials.append(decoder_results[session_id]['results'][aa]['no_shift'][n_units][rr]['predict_proba'][:,1])
                    else:
                        if sh in list(decoder_results[session_id]['results'][aa]['shift'][rr].keys()):
                            temp_shifts.append(decoder_results[session_id]['results'][aa]['shift'][rr][sh]['decision_function'])
                            temp_proba_shifts.append(decoder_results[session_id]['results'][aa]['shift'][rr][sh]['predict_proba'][:,1])
                if len(temp_shifts)>0:
                    decision_function_shifts.append(np.nanmean(np.vstack(temp_shifts),axis=0))
                    predict_proba_shifts.append(np.nanmean(np.vstack(temp_proba_shifts),axis=0))
                else:
                    decision_function_shifts.append(np.nan)
                    predict_proba_shifts.append(np.nan)

            confidence_all_trials=np.nanmean(np.vstack(confidence_all_trials),axis=0)
            predict_proba_all_trials=np.nanmean(np.vstack(predict_proba_all_trials),axis=0)

            # true_label=decoder_results[session_id]['results'][aa]['shift'][np.where(shifts==0)[0][0]]['true_label']

            try:
                decision_function_shifts=np.vstack(decision_function_shifts)
                predict_proba_shifts=np.vstack(predict_proba_shifts)
            except:
                logger.info(f'{session_id} | failed to stack decision functions / predict_proba; skipping')
                continue

            # #normalize all decision function values to the stdev of all the nulls
            # decision_function_shifts=decision_function_shifts/np.nanstd(decision_function_shifts[:])

            # corrected_decision_function=decision_function_shifts[shifts[half_shift_inds]==0,:].flatten()-np.median(decision_function_shifts,axis=0)

            # #option to normalize after, if n_units=='all', to account for different #'s of units
            # if n_units=='all':
            #     corrected_decision_function=corrected_decision_function/np.std(np.abs(corrected_decision_function))

            #for now, do NOT correct decision function values:
            corrected_decision_function=decision_function_shifts[shifts[half_shift_inds]==0,:].flatten()
            null_decision_function=np.median(decision_function_shifts[shifts[half_shift_inds]!=0,:],axis=0)
            null_min_decision_function=np.mean(np.vstack([decision_function_shifts[0,:],decision_function_shifts[-1,:]]),axis=0)

            predict_proba=predict_proba_shifts[shifts[half_shift_inds]==0,:].flatten()
            null_predict_proba=np.median(predict_proba_shifts[shifts[half_shift_inds]!=0,:],axis=0)
            null_min_predict_proba=np.mean(np.vstack([predict_proba_shifts[0,:],predict_proba_shifts[-1,:]]),axis=0)

            #get trial indices for each type
            vis_HIT_idx=trials_middle.query('(is_correct==True and is_target==True and is_vis_context==True \
                                        and is_response==True and is_reward_scheduled==False)').index.values
            aud_HIT_idx=trials_middle.query('(is_correct==True and is_target==True and is_vis_context==False \
                                        and is_response==True and is_reward_scheduled==False)').index.values
            vis_CR_idx=trials_middle.query('(is_correct==True and is_target==True and is_vis_context==True \
                                        and is_response==False and is_reward_scheduled==False)').index.values
            aud_CR_idx=trials_middle.query('(is_correct==True and is_target==True and is_vis_context==False \
                                        and is_response==False and is_reward_scheduled==False)').index.values
            vis_FA_idx=trials_middle.query('(is_correct==False and is_target==True and is_vis_context==True \
                                        and is_response==True and is_reward_scheduled==False)').index.values
            aud_FA_idx=trials_middle.query('(is_correct==False and is_target==True and is_vis_context==False \
                                        and is_response==True and is_reward_scheduled==False)').index.values

            correct_vis_idx=trials_middle.query('is_correct==True and is_target==True and is_reward_scheduled==False').index.values
            incorrect_vis_idx=trials_middle.query('is_correct==False and is_target==True and is_reward_scheduled==False').index.values
            correct_aud_idx=trials_middle.query('is_correct==True and is_target==True and is_reward_scheduled==False').index.values
            incorrect_aud_idx=trials_middle.query('is_correct==False and is_target==True and is_reward_scheduled==False').index.values

            #find average confidence per hit, fa, cr
            vis_HIT_mean=np.nanmean(corrected_decision_function[vis_HIT_idx])
            aud_HIT_mean=np.nanmean(corrected_decision_function[aud_HIT_idx])
            vis_CR_mean=np.nanmean(corrected_decision_function[vis_CR_idx])
            aud_CR_mean=np.nanmean(corrected_decision_function[aud_CR_idx])
            vis_FA_mean=np.nanmean(corrected_decision_function[vis_FA_idx])
            aud_FA_mean=np.nanmean(corrected_decision_function[aud_FA_idx])

            correct_mean=np.nanmean(np.hstack([corrected_decision_function[correct_vis_idx],-corrected_decision_function[correct_aud_idx]]))
            incorrect_mean=np.nanmean(np.hstack([corrected_decision_function[incorrect_vis_idx],-corrected_decision_function[incorrect_aud_idx]]))

            CR_all_mean=np.nanmean(np.hstack([corrected_decision_function[vis_CR_idx],-corrected_decision_function[aud_CR_idx]]))
            FA_all_mean=np.nanmean(np.hstack([corrected_decision_function[vis_FA_idx],-corrected_decision_function[aud_FA_idx]]))
            HIT_all_mean=np.nanmean(np.hstack([corrected_decision_function[vis_HIT_idx],-corrected_decision_function[aud_HIT_idx]]))

            #find average null confidence per hit, fa, cr
            vis_HIT_null_mean=np.nanmean(null_decision_function[vis_HIT_idx])
            aud_HIT_null_mean=np.nanmean(null_decision_function[aud_HIT_idx])
            vis_CR_null_mean=np.nanmean(null_decision_function[vis_CR_idx])
            aud_CR_null_mean=np.nanmean(null_decision_function[aud_CR_idx])
            vis_FA_null_mean=np.nanmean(null_decision_function[vis_FA_idx])
            aud_FA_null_mean=np.nanmean(null_decision_function[aud_FA_idx])

            correct_null_mean=np.nanmean(np.hstack([null_decision_function[correct_vis_idx],-null_decision_function[correct_aud_idx]]))
            incorrect_null_mean=np.nanmean(np.hstack([null_decision_function[incorrect_vis_idx],-null_decision_function[incorrect_aud_idx]]))

            CR_all_null_mean=np.nanmean(np.hstack([null_decision_function[vis_CR_idx],-null_decision_function[aud_CR_idx]]))
            FA_all_null_mean=np.nanmean(np.hstack([null_decision_function[vis_FA_idx],-null_decision_function[aud_FA_idx]]))
            HIT_all_null_mean=np.nanmean(np.hstack([null_decision_function[vis_HIT_idx],-null_decision_function[aud_HIT_idx]]))

            #find average predict_proba per hit, fa, cr
            vis_HIT_proba_mean=np.nanmean(predict_proba[vis_HIT_idx])
            aud_HIT_proba_mean=np.nanmean(predict_proba[aud_HIT_idx])
            vis_CR_proba_mean=np.nanmean(predict_proba[vis_CR_idx])
            aud_CR_proba_mean=np.nanmean(predict_proba[aud_CR_idx])
            vis_FA_proba_mean=np.nanmean(predict_proba[vis_FA_idx])
            aud_FA_proba_mean=np.nanmean(predict_proba[aud_FA_idx])

            correct_proba_mean=np.nanmean(np.hstack([predict_proba[correct_vis_idx],1-predict_proba[correct_aud_idx]]))
            incorrect_proba_mean=np.nanmean(np.hstack([predict_proba[incorrect_vis_idx],1-predict_proba[incorrect_aud_idx]]))

            CR_all_proba_mean=np.nanmean(np.hstack([predict_proba[vis_CR_idx],1-predict_proba[aud_CR_idx]]))
            FA_all_proba_mean=np.nanmean(np.hstack([predict_proba[vis_FA_idx],1-predict_proba[aud_FA_idx]]))
            HIT_all_proba_mean=np.nanmean(np.hstack([predict_proba[vis_HIT_idx],1-predict_proba[aud_HIT_idx]]))

            #find average null predict_proba per hit, fa, cr
            vis_HIT_null_proba_mean=np.nanmean(null_predict_proba[vis_HIT_idx])
            aud_HIT_null_proba_mean=np.nanmean(null_predict_proba[aud_HIT_idx])
            vis_CR_null_proba_mean=np.nanmean(null_predict_proba[vis_CR_idx])
            aud_CR_null_proba_mean=np.nanmean(null_predict_proba[aud_CR_idx])
            vis_FA_null_proba_mean=np.nanmean(null_predict_proba[vis_FA_idx])
            aud_FA_null_proba_mean=np.nanmean(null_predict_proba[aud_FA_idx])

            correct_null_proba_mean=np.nanmean(np.hstack([null_predict_proba[correct_vis_idx],1-null_predict_proba[correct_aud_idx]]))
            incorrect_null_proba_mean=np.nanmean(np.hstack([null_predict_proba[incorrect_vis_idx],1-null_predict_proba[incorrect_aud_idx]]))

            CR_all_null_proba_mean=np.nanmean(np.hstack([null_predict_proba[vis_CR_idx],1-null_predict_proba[aud_CR_idx]]))
            FA_all_null_proba_mean=np.nanmean(np.hstack([null_predict_proba[vis_FA_idx],1-null_predict_proba[aud_FA_idx]]))
            HIT_all_null_proba_mean=np.nanmean(np.hstack([null_predict_proba[vis_HIT_idx],1-null_predict_proba[aud_HIT_idx]]))

            #append to table
            decoder_confidence_versus_response_type['session'].append(session_id_str)
            decoder_confidence_versus_response_type['area'].append(area_name)
            decoder_confidence_versus_response_type['project'].append(project)
            decoder_confidence_versus_response_type['probe'].append(probe_name)
            if performance.query('rewarded_modality=="vis"').empty:
                decoder_confidence_versus_response_type['vis_context_dprime'].append(np.nan)
            else:
                decoder_confidence_versus_response_type['vis_context_dprime'].append(performance.query('rewarded_modality=="vis"')['cross_modal_dprime'].values[0])
            decoder_confidence_versus_response_type['vis_hit_confidence'].append(vis_HIT_mean)
            decoder_confidence_versus_response_type['vis_fa_confidence'].append(vis_FA_mean)
            decoder_confidence_versus_response_type['vis_cr_confidence'].append(vis_CR_mean)
            decoder_confidence_versus_response_type['vis_hit_null_confidence'].append(vis_HIT_null_mean)
            decoder_confidence_versus_response_type['vis_fa_null_confidence'].append(vis_FA_null_mean)
            decoder_confidence_versus_response_type['vis_cr_null_confidence'].append(vis_CR_null_mean)
            if performance.query('rewarded_modality=="aud"').empty:
                decoder_confidence_versus_response_type['aud_context_dprime'].append(np.nan)
            else:
                decoder_confidence_versus_response_type['aud_context_dprime'].append(performance.query('rewarded_modality=="aud"')['cross_modal_dprime'].values[0])
            decoder_confidence_versus_response_type['aud_hit_confidence'].append(aud_HIT_mean)
            decoder_confidence_versus_response_type['aud_fa_confidence'].append(aud_FA_mean)
            decoder_confidence_versus_response_type['aud_cr_confidence'].append(aud_CR_mean)
            decoder_confidence_versus_response_type['aud_hit_null_confidence'].append(aud_HIT_null_mean)
            decoder_confidence_versus_response_type['aud_fa_null_confidence'].append(aud_FA_null_mean)
            decoder_confidence_versus_response_type['aud_cr_null_confidence'].append(aud_CR_null_mean)

            decoder_confidence_versus_response_type['overall_dprime'].append(performance['cross_modal_dprime'].mean())
            decoder_confidence_versus_response_type['n_good_blocks'].append(performance.query('cross_modal_dprime>=1.0')['cross_modal_dprime'].count())
            decoder_confidence_versus_response_type['correct_confidence'].append(correct_mean)
            decoder_confidence_versus_response_type['incorrect_confidence'].append(incorrect_mean)
            decoder_confidence_versus_response_type['cr_all_confidence'].append(CR_all_mean)
            decoder_confidence_versus_response_type['fa_all_confidence'].append(FA_all_mean)
            decoder_confidence_versus_response_type['hit_all_confidence'].append(HIT_all_mean)
            decoder_confidence_versus_response_type['correct_null_confidence'].append(correct_null_mean)
            decoder_confidence_versus_response_type['incorrect_null_confidence'].append(incorrect_null_mean)
            decoder_confidence_versus_response_type['cr_all_null_confidence'].append(CR_all_null_mean)
            decoder_confidence_versus_response_type['fa_all_null_confidence'].append(FA_all_null_mean)
            decoder_confidence_versus_response_type['hit_all_null_confidence'].append(HIT_all_null_mean)

            #append predict_proba values
            decoder_confidence_versus_response_type['vis_hit_predict_proba'].append(vis_HIT_proba_mean)
            decoder_confidence_versus_response_type['vis_fa_predict_proba'].append(vis_FA_proba_mean)
            decoder_confidence_versus_response_type['vis_cr_predict_proba'].append(vis_CR_proba_mean)
            decoder_confidence_versus_response_type['aud_hit_predict_proba'].append(aud_HIT_proba_mean)
            decoder_confidence_versus_response_type['aud_fa_predict_proba'].append(aud_FA_proba_mean)
            decoder_confidence_versus_response_type['aud_cr_predict_proba'].append(aud_CR_proba_mean)
            decoder_confidence_versus_response_type['correct_predict_proba'].append(correct_proba_mean)
            decoder_confidence_versus_response_type['incorrect_predict_proba'].append(incorrect_proba_mean)
            decoder_confidence_versus_response_type['cr_all_predict_proba'].append(CR_all_proba_mean)
            decoder_confidence_versus_response_type['fa_all_predict_proba'].append(FA_all_proba_mean)
            decoder_confidence_versus_response_type['hit_all_predict_proba'].append(HIT_all_proba_mean)
            #append null predict_proba values
            decoder_confidence_versus_response_type['vis_hit_null_predict_proba'].append(vis_HIT_null_proba_mean)
            decoder_confidence_versus_response_type['vis_fa_null_predict_proba'].append(vis_FA_null_proba_mean)
            decoder_confidence_versus_response_type['vis_cr_null_predict_proba'].append(vis_CR_null_proba_mean)
            decoder_confidence_versus_response_type['aud_hit_null_predict_proba'].append(aud_HIT_null_proba_mean)
            decoder_confidence_versus_response_type['aud_fa_null_predict_proba'].append(aud_FA_null_proba_mean)
            decoder_confidence_versus_response_type['aud_cr_null_predict_proba'].append(aud_CR_null_proba_mean)
            decoder_confidence_versus_response_type['correct_null_predict_proba'].append(correct_null_proba_mean)
            decoder_confidence_versus_response_type['incorrect_null_predict_proba'].append(incorrect_null_proba_mean)
            decoder_confidence_versus_response_type['cr_all_null_predict_proba'].append(CR_all_null_proba_mean)
            decoder_confidence_versus_response_type['fa_all_null_predict_proba'].append(FA_all_null_proba_mean)
            decoder_confidence_versus_response_type['hit_all_null_predict_proba'].append(HIT_all_null_proba_mean)

            # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
            if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                decoder_confidence_versus_response_type['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                decoder_confidence_versus_response_type['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                decoder_confidence_versus_response_type['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])
                decoder_confidence_versus_response_type['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])
            else:
                decoder_confidence_versus_response_type['ccf_ap_mean'].append(np.nan)
                decoder_confidence_versus_response_type['ccf_dv_mean'].append(np.nan)
                decoder_confidence_versus_response_type['ccf_ml_mean'].append(np.nan)
                decoder_confidence_versus_response_type['n_units'].append(np.nan)

            #find decoder confidence according to time/trials since last rewarded target
            #3 arrays - time since last rewarded target, trials since last rewarded target, decoder confidence
            trials_since_rewarded_target=trials_middle['trials_since_rewarded_target'].values
            time_since_rewarded_target=trials_middle['time_since_rewarded_target'].values
            confidence=corrected_decision_function[trials_middle.index.values]
            confidence_null=null_decision_function[trials_middle.index.values]
            confidence_null_min=null_min_decision_function[trials_middle.index.values]
            temp_predict_proba=predict_proba[trials_middle.index.values]
            temp_predict_proba_null=null_predict_proba[trials_middle.index.values]
            temp_predict_proba_null_min=null_min_predict_proba[trials_middle.index.values]

            for tt,trial in trials_middle.reset_index().iterrows():
                #reverse sign if other modality
                if trial['is_aud_context']:
                    confidence[tt]=-confidence[tt]

            #append to table per session and area
            decoder_confidence_versus_trials_since_rewarded_target['session'].append(session_id_str)
            decoder_confidence_versus_trials_since_rewarded_target['area'].append(area_name)
            decoder_confidence_versus_trials_since_rewarded_target['project'].append(project)
            decoder_confidence_versus_trials_since_rewarded_target['probe'].append(probe_name)
            decoder_confidence_versus_trials_since_rewarded_target['trial_index'].append(trials_middle['original_index'].values)
            decoder_confidence_versus_trials_since_rewarded_target['trials_since_rewarded_target'].append(trials_since_rewarded_target)
            decoder_confidence_versus_trials_since_rewarded_target['time_since_rewarded_target'].append(time_since_rewarded_target)
            decoder_confidence_versus_trials_since_rewarded_target['confidence'].append(confidence)
            decoder_confidence_versus_trials_since_rewarded_target['confidence_null'].append(confidence_null)
            decoder_confidence_versus_trials_since_rewarded_target['confidence_null_min'].append(confidence_null_min)
            decoder_confidence_versus_trials_since_rewarded_target['predict_proba'].append(temp_predict_proba)
            decoder_confidence_versus_trials_since_rewarded_target['predict_proba_null'].append(temp_predict_proba_null)
            decoder_confidence_versus_trials_since_rewarded_target['predict_proba_null_min'].append(temp_predict_proba_null_min)

            #trials/time since last bit of information
            trials_since_last_information=trials_middle['trials_since_last_information'].values
            time_since_last_information=trials_middle['time_since_last_information'].values

            decoder_confidence_versus_trials_since_rewarded_target['trials_since_last_information'].append(trials_since_last_information)
            decoder_confidence_versus_trials_since_rewarded_target['time_since_last_information'].append(time_since_last_information)

            #trials/time since last bit of information, excluding trials after non-responses to targets
            trials_since_last_information_no_targets=trials_middle['trials_since_last_information_no_targets'].values
            time_since_last_information_no_targets=trials_middle['time_since_last_information_no_targets'].values

            decoder_confidence_versus_trials_since_rewarded_target['trials_since_last_information_no_targets'].append(trials_since_last_information_no_targets)
            decoder_confidence_versus_trials_since_rewarded_target['time_since_last_information_no_targets'].append(time_since_last_information_no_targets)

            # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
            if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                decoder_confidence_versus_trials_since_rewarded_target['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                decoder_confidence_versus_trials_since_rewarded_target['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                decoder_confidence_versus_trials_since_rewarded_target['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])
                decoder_confidence_versus_trials_since_rewarded_target['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])
            else:
                decoder_confidence_versus_trials_since_rewarded_target['ccf_ap_mean'].append(np.nan)
                decoder_confidence_versus_trials_since_rewarded_target['ccf_dv_mean'].append(np.nan)
                decoder_confidence_versus_trials_since_rewarded_target['ccf_ml_mean'].append(np.nan)
                decoder_confidence_versus_trials_since_rewarded_target['n_units'].append(np.nan)

            decoder_confidence_versus_trials_since_rewarded_target['cross_modal_dprime'].append(performance['cross_modal_dprime'].mean())
            decoder_confidence_versus_trials_since_rewarded_target['n_good_blocks'].append(np.sum(performance['cross_modal_dprime']>=1.0))

            #decoder confidence for every trial
            decoder_confidence_all_trials['session'].append(session_id_str)
            decoder_confidence_all_trials['area'].append(area_name)
            decoder_confidence_all_trials['project'].append(project)
            decoder_confidence_all_trials['probe'].append(probe_name)
            decoder_confidence_all_trials['cross_modal_dprime'].append(performance['cross_modal_dprime'].mean())
            decoder_confidence_all_trials['n_good_blocks'].append(np.sum(performance['cross_modal_dprime']>=1.0))
            decoder_confidence_all_trials['trial_index'].append(trials['original_index'].values)
            decoder_confidence_all_trials['confidence'].append(confidence_all_trials)
            decoder_confidence_all_trials['predict_proba'].append(predict_proba_all_trials)
            if 'n_units' in decoder_results[session_id]['results'][aa].keys():
                decoder_confidence_all_trials['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])

            # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
            if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                decoder_confidence_all_trials['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                decoder_confidence_all_trials['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                decoder_confidence_all_trials['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])

            ##loop through blocks##
            blocks=trials_middle['block_index'].unique()
            for bb in blocks:
                block_trials=trials_middle.query('block_index==@bb and is_reward_scheduled==False')
                if len( block_trials )==0:
                    continue
                #find average confidence and dprime for the block
                if block_trials['is_vis_context'].values[0]:
                    multiplier=1
                    additive=0
                elif block_trials['is_aud_context'].values[0]:
                    multiplier=-1
                    additive=1

                block_dprime=performance.query('block_index==@bb')['cross_modal_dprime'].values[0]
                block_mean=np.nanmean(corrected_decision_function[block_trials.index.values])*multiplier
                block_mean_null=np.nanmean(null_decision_function[block_trials.index.values])*multiplier
                block_mean_null_min=np.nanmean(null_min_decision_function[block_trials.index.values])*multiplier
                block_predict_proba=additive+np.nanmean(predict_proba[block_trials.index.values])*multiplier
                block_predict_proba_null=additive+np.nanmean(null_predict_proba[block_trials.index.values])*multiplier
                block_predict_proba_null_min=additive+np.nanmean(null_min_predict_proba[block_trials.index.values])*multiplier

                #append to table
                decoder_confidence_dprime_by_block['session'].append(session_id_str)
                decoder_confidence_dprime_by_block['area'].append(area_name)
                decoder_confidence_dprime_by_block['project'].append(project)
                decoder_confidence_dprime_by_block['probe'].append(probe_name)
                decoder_confidence_dprime_by_block['block'].append(bb)
                decoder_confidence_dprime_by_block['cross_modal_dprime'].append(block_dprime)
                decoder_confidence_dprime_by_block['n_good_blocks'].append(np.sum(performance['cross_modal_dprime']>=1.0))
                decoder_confidence_dprime_by_block['confidence'].append(block_mean)
                decoder_confidence_dprime_by_block['null_confidence'].append(block_mean_null)
                decoder_confidence_dprime_by_block['null_min_confidence'].append(block_mean_null_min)
                decoder_confidence_dprime_by_block['predict_proba'].append(block_predict_proba)
                decoder_confidence_dprime_by_block['predict_proba_null'].append(block_predict_proba_null)
                decoder_confidence_dprime_by_block['predict_proba_null_min'].append(block_predict_proba_null_min)

                # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
                if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                    decoder_confidence_dprime_by_block['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                    decoder_confidence_dprime_by_block['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                    decoder_confidence_dprime_by_block['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])
                    decoder_confidence_dprime_by_block['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])
                else:
                    decoder_confidence_dprime_by_block['ccf_ap_mean'].append(np.nan)
                    decoder_confidence_dprime_by_block['ccf_dv_mean'].append(np.nan)
                    decoder_confidence_dprime_by_block['ccf_ml_mean'].append(np.nan)
                    decoder_confidence_dprime_by_block['n_units'].append(np.nan)

            #get confidence around the block switch
            switch_trials=trials_middle.query('is_context_switch')
            ##loop through switches##
            for st,switch_trial in switch_trials.iloc[1:].iterrows():
                if switch_trial['is_vis_context']:
                    multiplier=1
                    additive=0
                elif switch_trial['is_aud_context']:
                    multiplier=-1
                    additive=1
                switch_trial_block_index=switch_trial['block_index']
                #append to table
                decoder_confidence_by_switch['session'].append(session_id_str)
                decoder_confidence_by_switch['area'].append(area_name)
                decoder_confidence_by_switch['project'].append(project)
                decoder_confidence_by_switch['probe'].append(probe_name)
                decoder_confidence_by_switch['switch_trial'].append(switch_trial['id'])
                decoder_confidence_by_switch['block'].append(switch_trial_block_index)
                decoder_confidence_by_switch['dprime_before'].append(performance.query('block_index==(@switch_trial_block_index-1)')['cross_modal_dprime'].values[0])
                decoder_confidence_by_switch['dprime_after'].append(performance.query('block_index==(@switch_trial_block_index)')['cross_modal_dprime'].values[0])
                decoder_confidence_by_switch['confidence'].append(corrected_decision_function[switch_trial['id']-20:switch_trial['id']+30]*multiplier)
                decoder_confidence_by_switch['null_confidence'].append(null_decision_function[switch_trial['id']-20:switch_trial['id']+30]*multiplier)
                decoder_confidence_by_switch['null_min_confidence'].append(null_min_decision_function[switch_trial['id']-20:switch_trial['id']+30]*multiplier)
                decoder_confidence_by_switch['predict_proba'].append(additive+predict_proba[switch_trial['id']-20:switch_trial['id']+30]*multiplier)
                decoder_confidence_by_switch['predict_proba_null'].append(additive+null_predict_proba[switch_trial['id']-20:switch_trial['id']+30]*multiplier)
                decoder_confidence_by_switch['predict_proba_null_min'].append(additive+null_min_predict_proba[switch_trial['id']-20:switch_trial['id']+30]*multiplier)

                # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
                if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                    decoder_confidence_by_switch['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                    decoder_confidence_by_switch['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                    decoder_confidence_by_switch['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])
                    decoder_confidence_by_switch['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])
                else:
                    decoder_confidence_by_switch['ccf_ap_mean'].append(np.nan)
                    decoder_confidence_by_switch['ccf_dv_mean'].append(np.nan)
                    decoder_confidence_by_switch['ccf_ml_mean'].append(np.nan)
                    decoder_confidence_by_switch['n_units'].append(np.nan)

            #decoder confidence before/after rewarded target, response to non-rewarded target, non-response to non-rewarded target
            sign_corrected_decision_function=corrected_decision_function.copy()
            for tt,trial in trials_middle.iterrows():
                if trial['is_aud_context']:
                    sign_corrected_decision_function[tt]=-sign_corrected_decision_function[tt]
            #find trial and trial+1 of rewarded targets
            rewarded_target_trials=trials_middle.query('is_rewarded==True and is_target==True and is_response==True and is_reward_scheduled==False').index.values
            rewarded_target_trials_plus_one=rewarded_target_trials+1
            if len(rewarded_target_trials_plus_one)>0:
                if rewarded_target_trials_plus_one[-1]>=len(corrected_decision_function):
                    rewarded_target_trials=rewarded_target_trials[:-1]
                    rewarded_target_trials_plus_one=rewarded_target_trials_plus_one[:-1]

            #find trials and trials+1 of responses to non-rewarded targets
            non_rewarded_target_trials=trials_middle.query('is_rewarded==False and is_target==True and is_response==True').index.values
            non_rewarded_target_trials_plus_one=non_rewarded_target_trials+1
            if len(non_rewarded_target_trials_plus_one)>0:
                if non_rewarded_target_trials_plus_one[-1]>=len(corrected_decision_function):
                    non_rewarded_target_trials=non_rewarded_target_trials[:-1]
                    non_rewarded_target_trials_plus_one=non_rewarded_target_trials_plus_one[:-1]

            #find trials and trials+1 of non-response to non-rewarded targets
            non_response_non_rewarded_target_trials=trials_middle.query('is_rewarded==False and is_target==True and is_response==False').index.values
            non_response_non_rewarded_target_trials_plus_one=non_response_non_rewarded_target_trials+1
            if len(non_response_non_rewarded_target_trials_plus_one)>0:
                if non_response_non_rewarded_target_trials_plus_one[-1]>=len(corrected_decision_function):
                    non_response_non_rewarded_target_trials=non_response_non_rewarded_target_trials[:-1]
                    non_response_non_rewarded_target_trials_plus_one=non_response_non_rewarded_target_trials_plus_one[:-1]

            non_response_non_target_trials=trials_middle.query('is_rewarded==False and is_target==False and is_response==False').index.values
            non_response_non_target_trials_plus_one=non_response_non_target_trials+1
            if len(non_response_non_target_trials_plus_one)>0:
                if non_response_non_target_trials_plus_one[-1]>=len(corrected_decision_function):
                    non_response_non_target_trials=non_response_non_target_trials[:-1]
                    non_response_non_target_trials_plus_one=non_response_non_target_trials_plus_one[:-1]

            #append to table
            decoder_confidence_before_after_target['session'].append(session_id_str)
            decoder_confidence_before_after_target['area'].append(area_name)
            decoder_confidence_before_after_target['project'].append(project)
            decoder_confidence_before_after_target['probe'].append(probe_name)
            decoder_confidence_before_after_target['cross_modal_dprime'].append(performance['cross_modal_dprime'].mean())
            decoder_confidence_before_after_target['n_good_blocks'].append(np.sum(performance['cross_modal_dprime']>=1.0))
            decoder_confidence_before_after_target['rewarded_target'].append(sign_corrected_decision_function[rewarded_target_trials])
            decoder_confidence_before_after_target['rewarded_target_plus_one'].append(sign_corrected_decision_function[rewarded_target_trials_plus_one])
            if len(non_rewarded_target_trials)>0:
                decoder_confidence_before_after_target['non_rewarded_target'].append(sign_corrected_decision_function[non_rewarded_target_trials])
                decoder_confidence_before_after_target['non_rewarded_target_plus_one'].append(sign_corrected_decision_function[non_rewarded_target_trials_plus_one])
            else:
                decoder_confidence_before_after_target['non_rewarded_target'].append(np.nan)
                decoder_confidence_before_after_target['non_rewarded_target_plus_one'].append(np.nan)
            if len(non_response_non_rewarded_target_trials)>0:
                decoder_confidence_before_after_target['non_response_non_rewarded_target'].append(sign_corrected_decision_function[non_response_non_rewarded_target_trials])
                decoder_confidence_before_after_target['non_response_non_rewarded_target_plus_one'].append(sign_corrected_decision_function[non_response_non_rewarded_target_trials_plus_one])
            else:
                decoder_confidence_before_after_target['non_response_non_rewarded_target'].append(np.nan)
                decoder_confidence_before_after_target['non_response_non_rewarded_target_plus_one'].append(np.nan)
            decoder_confidence_before_after_target['non_response_non_target_trials'].append(sign_corrected_decision_function[non_response_non_target_trials])
            decoder_confidence_before_after_target['non_response_non_target_trials_plus_one'].append(sign_corrected_decision_function[non_response_non_target_trials_plus_one])

            # 'ccf_ap_mean', 'ccf_dv_mean', 'ccf_ml_mean'
            if 'ccf_ap_mean' in decoder_results[session_id]['results'][aa].keys():
                decoder_confidence_before_after_target['ccf_ap_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ap_mean'])
                decoder_confidence_before_after_target['ccf_dv_mean'].append(decoder_results[session_id]['results'][aa]['ccf_dv_mean'])
                decoder_confidence_before_after_target['ccf_ml_mean'].append(decoder_results[session_id]['results'][aa]['ccf_ml_mean'])
                decoder_confidence_before_after_target['n_units'].append(decoder_results[session_id]['results'][aa]['n_units'])
            else:
                decoder_confidence_before_after_target['ccf_ap_mean'].append(np.nan)
                decoder_confidence_before_after_target['ccf_dv_mean'].append(np.nan)
                decoder_confidence_before_after_target['ccf_ml_mean'].append(np.nan)
                decoder_confidence_before_after_target['n_units'].append(np.nan)

        total_time=time.time()-start_time
        session_time=time.time()-session_start_time
        logger.info(f'{session_id} | finished session')
        logger.info(f'{session_id} | session time: {session_time} seconds;  total time: {total_time} seconds')

    decoder_confidence_versus_response_type_dict=decoder_confidence_versus_response_type.copy()
    decoder_confidence_dprime_by_block_dict=decoder_confidence_dprime_by_block.copy()
    decoder_confidence_by_switch_dict=decoder_confidence_by_switch.copy()
    decoder_confidence_versus_trials_since_rewarded_target_dict=decoder_confidence_versus_trials_since_rewarded_target.copy()
    decoder_confidence_all_trials_dict=decoder_confidence_all_trials.copy()
    decoder_confidence_before_after_target_dict=decoder_confidence_before_after_target.copy()

    decoder_confidence_versus_response_type=pd.DataFrame(decoder_confidence_versus_response_type)
    decoder_confidence_dprime_by_block=pd.DataFrame(decoder_confidence_dprime_by_block)
    decoder_confidence_by_switch=pd.DataFrame(decoder_confidence_by_switch)
    decoder_confidence_versus_trials_since_rewarded_target=pd.DataFrame(decoder_confidence_versus_trials_since_rewarded_target)

    try:
        decoder_confidence_all_trials=pd.DataFrame(decoder_confidence_all_trials)
    except:
        k = {key: len(decoder_confidence_all_trials[key]) for key in decoder_confidence_all_trials.keys()}
        logger.error(f'error converting decoder_confidence_all_trials to dataframe: {k}')
        raise
    decoder_confidence_before_after_target=pd.DataFrame(decoder_confidence_before_after_target)

    if savepath is not None:
        if not upath.UPath(savepath).is_dir():
            upath.UPath(savepath).mkdir(parents=True)
        if n_units is not None:
            n_units_str='_'+str(n_units)+'_units'
        else:
            n_units_str=''

        if single_session:
            temp_session_str=session_id_str+'_'
        else:
            temp_session_str=''

        if use_zarr==True and single_session==True:
            results={
                session_id:{
                    'decoder_confidence_versus_response_type'+n_units_str:decoder_confidence_versus_response_type_dict,
                    'decoder_confidence_dprime_by_block'+n_units_str:decoder_confidence_dprime_by_block_dict,
                    'decoder_confidence_by_switch'+n_units_str:decoder_confidence_by_switch_dict,
                    'decoder_confidence_versus_trials_since_rewarded_target'+n_units_str:decoder_confidence_versus_trials_since_rewarded_target_dict,
                    'decoder_confidence_all_trials'+n_units_str:decoder_confidence_all_trials_dict,
                    'decoder_confidence_before_after_target'+n_units_str:decoder_confidence_before_after_target_dict,
                    },
            }

            zarr_file = zarr.open(files, mode='w')

            dump_dict_to_zarr(zarr_file, results)

        elif use_zarr==False:

            decoder_confidence_versus_response_type.to_csv(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_versus_response_type'+n_units_str+'.csv'),index=False)
            decoder_confidence_dprime_by_block.to_csv(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_dprime_by_block'+n_units_str+'.csv'),index=False)
            decoder_confidence_by_switch.to_csv(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_by_switch'+n_units_str+'.csv'),index=False)
            decoder_confidence_versus_trials_since_rewarded_target.to_csv(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_versus_trials_since_rewarded_target'+n_units_str+'.csv'),index=False)
            decoder_confidence_all_trials.to_csv(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_all_trials'+n_units_str+'.csv'),index=False)
            decoder_confidence_before_after_target.to_csv(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_before_after_target'+n_units_str+'.csv'),index=False)

            decoder_confidence_versus_response_type.to_pickle(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_versus_response_type'+n_units_str+'.pkl'))
            decoder_confidence_dprime_by_block.to_pickle(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_dprime_by_block'+n_units_str+'.pkl'))
            decoder_confidence_by_switch.to_pickle(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_by_switch'+n_units_str+'.pkl'))
            decoder_confidence_versus_trials_since_rewarded_target.to_pickle(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_versus_trials_since_rewarded_target'+n_units_str+'.pkl'))
            decoder_confidence_all_trials.to_pickle(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_all_trials'+n_units_str+'.pkl'))
            decoder_confidence_before_after_target.to_pickle(upath.UPath(savepath) / (temp_session_str+'decoder_confidence_before_after_target'+n_units_str+'.pkl'))

            logger.info(f'saved {n_units_str} decoder confidence tables to: {savepath}')

        del decoder_results
        gc.collect()

        if return_table:
            return decoder_confidence_versus_response_type,decoder_confidence_dprime_by_block,decoder_confidence_by_switch,decoder_confidence_versus_trials_since_rewarded_target,decoder_confidence_before_after_target


def concat_decoder_summary_tables(dir,savepath):

    LP_keys = ['session_id', 'project', 'n_good_blocks','cross_modal_dprime', 'true_accuracy_all',
       'null_accuracy_mean_all', 'null_accuracy_median_all',
       'null_accuracy_std_all', 'p_value_all',
       'true_accuracy_all_trials_no_shift_all']

    #create summary folder if does not exist
    if not upath.UPath(savepath).is_dir():
            upath.UPath(savepath).mkdir(parents=True)

    decoder_results_summary_files=[]
    for p in dir.glob('*decoding_results.csv'):
        session_file = pd.read_csv(p)
        if len(session_file) == 1:
            decoder_results_summary_files.append(session_file[LP_keys])
        else:
            decoder_results_summary_files.append(session_file)

    decoder_results_summary=pd.concat(decoder_results_summary_files)
    decoder_results_summary.to_csv(upath.UPath(savepath) / 'decoder_results_summary.csv')

    #find different n_units
    n_units=[]
    for col in decoder_results_summary.filter(like='true_accuracy_').columns.values:
        if 'no_shift' in col:
            continue
        if len(col.split('_'))==3:
            temp_n_units=col.split('_')[2]
            try:
                n_units.append(int(temp_n_units))
            except:
                n_units.append(temp_n_units)
        else:
            n_units.append(None)

    for nu in n_units:
        decoder_confidence_versus_response_type_files=[]
        for p in dir.glob('*decoder_confidence_versus_response_type_'+str(nu)+'_units.pkl'):
            decoder_confidence_versus_response_type_files.append(pd.read_pickle(p))
        decoder_confidence_versus_response_type=pd.concat(decoder_confidence_versus_response_type_files)
        decoder_confidence_versus_response_type.to_pickle(upath.UPath(savepath) / ('decoder_confidence_versus_response_type_'+str(nu)+'_units.pkl'))

        decoder_confidence_dprime_by_block_files=[]
        for p in dir.glob('*decoder_confidence_dprime_by_block_'+str(nu)+'_units.pkl'):
            decoder_confidence_dprime_by_block_files.append(pd.read_pickle(p))
        decoder_confidence_dprime_by_block=pd.concat(decoder_confidence_dprime_by_block_files)
        decoder_confidence_dprime_by_block.to_pickle(upath.UPath(savepath) / ('decoder_confidence_dprime_by_block_'+str(nu)+'_units.pkl'))

        decoder_confidence_by_switch_files=[]
        for p in dir.glob('*decoder_confidence_by_switch_'+str(nu)+'_units.pkl'):
            decoder_confidence_by_switch_files.append(pd.read_pickle(p))
        decoder_confidence_by_switch=pd.concat(decoder_confidence_by_switch_files)
        decoder_confidence_by_switch.to_pickle(upath.UPath(savepath) / ('decoder_confidence_by_switch_'+str(nu)+'_units.pkl'))

        decoder_confidence_versus_trials_since_rewarded_target_files=[]
        for p in dir.glob('*decoder_confidence_versus_trials_since_rewarded_target_'+str(nu)+'_units.pkl'):
            decoder_confidence_versus_trials_since_rewarded_target_files.append(pd.read_pickle(p))
        decoder_confidence_versus_trials_since_rewarded_target=pd.concat(decoder_confidence_versus_trials_since_rewarded_target_files)
        decoder_confidence_versus_trials_since_rewarded_target.to_pickle(upath.UPath(savepath) / ('decoder_confidence_versus_trials_since_rewarded_target_'+str(nu)+'_units.pkl'))

        decoder_confidence_all_trials_files=[]
        for p in dir.glob('*decoder_confidence_all_trials_'+str(nu)+'_units.pkl'):
            decoder_confidence_all_trials_files.append(pd.read_pickle(p))
        decoder_confidence_all_trials=pd.concat(decoder_confidence_all_trials_files)
        decoder_confidence_all_trials.to_pickle(upath.UPath(savepath) / ('decoder_confidence_all_trials_'+str(nu)+'_units.pkl'))

        decoder_confidence_before_after_target_files=[]
        for p in dir.glob('*decoder_confidence_before_after_target_'+str(nu)+'_units.pkl'):
            decoder_confidence_before_after_target_files.append(pd.read_pickle(p))
        decoder_confidence_before_after_target=pd.concat(decoder_confidence_before_after_target_files)
        decoder_confidence_before_after_target.to_pickle(upath.UPath(savepath) / ('decoder_confidence_before_after_target_'+str(nu)+'_units.pkl'))

