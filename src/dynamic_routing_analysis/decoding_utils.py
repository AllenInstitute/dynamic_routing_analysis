import os
import pickle

import npc_lims
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import ensemble, svm
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from dynamic_routing_analysis import spike_utils


def linearSVC_decoder(input_data,labels,crossval='5_fold',crossval_index=None,labels_as_index=False):
    
    output={}

    # scaler = StandardScaler()
    scaler = RobustScaler()

    scaler.fit(input_data)
    X = scaler.transform(input_data)
    # X = input_data
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
    models=[]

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


    for train,test in train_test_split:
        clf=svm.LinearSVC(max_iter=5000,dual='auto',class_weight='balanced')
        # clf=svm.SVC(class_weight='balanced',kernel='linear',probability=True)
        
        clf.fit(X[train],y[train])
        prediction=clf.predict(X[test])
        decision_function=clf.decision_function(X[test])
        ypred_all.append(prediction)
        ypred[test] = prediction
        ypred_train.append(clf.predict(X[train]))
        ytrue_train.append(y[train])
        y_dec_func[test] = decision_function
        dec_func_all.append(decision_function)
        tidx_used.append([test])
        # coefs.append(clf.dual_coef_)
        coefs.append(clf.coef_) #for linear SVC only
        classes.append(clf.classes_)
        intercept.append(clf.intercept_)
        params.append(clf.get_params())
        train_trials.append(train)
        test_trials.append(test)

        models.append(clf)

    cr_dict=classification_report(y, ypred, output_dict=True)
    balanced_accuracy=balanced_accuracy_score(y, ypred, sample_weight=None, adjusted=False)

    cr_dict_train=classification_report(np.hstack(ytrue_train), np.hstack(ypred_train), output_dict=True)
    balanced_accuracy_train=balanced_accuracy_score(np.hstack(ytrue_train), np.hstack(ypred_train), sample_weight=None, adjusted=False)

    output['cr']=cr_dict
    output['pred_label']=ypred
    output['true_label']=y
    output['pred_label_all']=ypred_all
    output['trials_used']=tidx_used
    output['decision_function']=y_dec_func
    output['decision_function_all']=dec_func_all
    output['coefs']=coefs
    output['classes']=classes
    output['intercept']=intercept
    output['params']=params
    output['balanced_accuracy']=balanced_accuracy
    
    output['pred_label_train']=np.hstack(ypred_train)
    output['true_label_train']=np.hstack(ytrue_train)
    output['cr_train']=cr_dict_train
    output['balanced_accuracy_train']=balanced_accuracy_train
    output['train_trials']=train_trials
    output['test_trials']=test_trials
    output['models']=models
    output['scaler']=scaler
    output['label_names']=unique_labels
    # output['input_data']=input_data
    output['labels']=labels

    return output


def random_forest_decoder(input_data,labels,crossval='5_fold',crossval_index=None,labels_as_index=False):
    
    output={}

    # scaler = StandardScaler()
    # scaler = RobustScaler()

    # scaler.fit(input_data)
    # X = scaler.transform(input_data)
    X = input_data
    unique_labels=np.unique(labels)
    if labels_as_index==True:
        labels=np.array([np.where(unique_labels==x)[0][0] for x in labels])

    y = labels

    # if len(np.unique(labels))>2:
    #     y_dec_func=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    # else:
    #     y_dec_func=np.full(len(y), fill_value=np.nan)

    y_predict_proba=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    # feature_importances=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
 
    if type(y[0])==bool:
        ypred=np.full(len(y), fill_value=False)
    elif type(y[0])==str:
        ypred=np.full(len(y), fill_value='       ')
    else:
        ypred=np.full(len(y), fill_value=np.nan)

    tidx_used=[]
    
    coefs=[]
    classes=[]
    # intercept=[]
    params=[]
    ypred_train=[]
    ytrue_train=[]
    train_trials=[]
    test_trials=[]
    models=[]


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
            # train.append(not_block_inds)

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
    elif crossval=='5_fold':
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        train_test_split = skf.split(input_data, labels)


    for train,test in train_test_split:
        # clf=svm.LinearSVC(max_iter=5000,dual='auto',class_weight='balanced')
        # clf=svm.SVC(class_weight='balanced',kernel='linear',probability=True)
        clf=ensemble.RandomForestClassifier(class_weight='balanced',)
        
        clf.fit(X[train],y[train])
        ypred[test] = clf.predict(X[test])
        ypred_train.append(clf.predict(X[train]))
        ytrue_train.append(y[train])
        # y_dec_func[test] = clf.decision_function(X[test])
        y_predict_proba[test,:] = clf.predict_proba(X[test])
        tidx_used.append([test])
        # coefs.append(clf.dual_coef_)
        # coefs.append(clf.coef_) #for linear SVC only
        classes.append(clf.classes_)
        # intercept.append(clf.intercept_)
        params.append(clf.get_params())
        train_trials.append(train)
        test_trials.append(test)

        models.append(clf)

    cr_dict=classification_report(y, ypred, output_dict=True)
    balanced_accuracy=balanced_accuracy_score(y, ypred, sample_weight=None, adjusted=False)

    cr_dict_train=classification_report(np.hstack(ytrue_train), np.hstack(ypred_train), output_dict=True)
    balanced_accuracy_train=balanced_accuracy_score(np.hstack(ytrue_train), np.hstack(ypred_train), sample_weight=None, adjusted=False)

    output['cr']=cr_dict
    output['pred_label']=ypred
    output['true_label']=y
    output['trials_used']=tidx_used
    # output['decision_function']=y_dec_func
    output['predict_proba']=y_predict_proba
    output['coefs']=coefs
    output['classes']=classes
    # output['intercept']=intercept
    output['params']=params
    output['balanced_accuracy']=balanced_accuracy
    
    output['pred_label_train']=np.hstack(ypred_train)
    output['true_label_train']=np.hstack(ytrue_train)
    output['cr_train']=cr_dict_train
    output['balanced_accuracy_train']=balanced_accuracy_train
    output['train_trials']=train_trials
    output['test_trials']=test_trials
    output['models']=models
    # output['scaler']=scaler
    output['label_names']=unique_labels
    # output['input_data']=input_data
    output['labels']=labels

    return output


def decode_context_from_units(session,params):

    trnum=params['trnum']
    n_units=params['n_units']
    u_min=params['u_min']
    n_repeats=params['n_repeats']
    spikes_binsize=params['spikes_binsize']
    spikes_time_before=params['spikes_time_before']
    spikes_time_after=params['spikes_time_after']
    decoder_binsize=params['decoder_binsize']
    decoder_time_before=params['decoder_time_before']
    decoder_time_after=params['decoder_time_after']
    balance_labels=params['balance_labels']
    savepath=params['savepath']
    filename=params['filename']
    use_structure_probe=params['use_structure_probe']
    crossval=params['crossval']
    all_areas=params['all_areas']
    labels_as_index=params['labels_as_index']
    decoder_type=params['decoder_type']
    generate_labels=params['generate_labels']
    
    time_bins=np.arange(-decoder_time_before,decoder_time_after,decoder_binsize)

    svc_results={}
    
    # make unit xarrays
    # time_before = 0.2
    # time_after = 0.5

    trials=pd.read_parquet(
                npc_lims.get_cache_path('trials',session.id,version='v0.0.173')
            )
    units=pd.read_parquet(
                npc_lims.get_cache_path('units',session.id,version='v0.0.173')
            )

    trial_da = spike_utils.make_neuron_time_trials_tensor(units, trials, spikes_time_before, spikes_time_after, spikes_binsize)

    if use_structure_probe:
        structure_probe=spike_utils.get_structure_probe(session)
        area_counts=structure_probe['structure_probe'].value_counts()
    else:
        area_counts=units['structure'].value_counts()
    
    # predict=['stim_ids','block_ids','trial_response']
    predict=['block_ids']

    #save metadata about this session & decoder params
    # svc_results['metadata']=session.metadata 
    svc_results['trial_numbers']=trnum
    svc_results['unit_numbers']=n_units
    svc_results['min_n_units']=u_min
    svc_results['n_repeats']=n_repeats
    svc_results['time_bins']=time_bins
    svc_results['spikes_time_before']=spikes_time_before
    svc_results['spikes_time_after']=spikes_time_after
    svc_results['spikes_binsize']=spikes_binsize
    svc_results['decoder_time_before']=decoder_time_before
    svc_results['decoder_time_after']=decoder_time_after
    svc_results['decoder_binsize']=decoder_binsize
    svc_results['balance_labels']=balance_labels
    svc_results['crossval']=crossval
    svc_results['all_areas']=all_areas
    svc_results['labels_as_index']=labels_as_index
    svc_results['decoder_type']=decoder_type
    svc_results['generate_labels']=generate_labels
    
    #loop through different labels to predict
    for p in predict:
        svc_results[p]={}
    
        #choose what variable to predict
        if p=='stim_ids':
            #exclude any trials that had opto stimulation
            if 'opto_power' in trials[:].columns:
                trial_sel = trials[:].query('opto_power.isnull() and stim_name != "catch"').index
            else:
                trial_sel = trials[:].query('stim_name != "catch"').index
                
            # grab the stimulus ids
            pred_var = trials[:]['stim_name'][trial_sel].values
    
        elif p=='block_ids':
            #exclude any trials that had opto stimulation
            if crossval=='blockwise':
                if 'opto_power' in trials[:].columns:
                    trial_sel = trials[:].query('opto_power.isnull()').index# and trial_index_in_block>=5').index
                else:
                    trial_sel = trials[:].query('trial_index_in_block>=5').index
            else:
                if 'opto_power' in trials[:].columns:
                    trial_sel = trials[:].query('opto_power.isnull()').index# and trial_index_in_block>=5').index
                else:
                    trial_sel = trials[:].index
                
            # or, use block IDs
            if generate_labels == False:
                pred_var = trials[:]['context_name'][trial_sel].values
            else:
                start_time=trials[:]['start_time'].iloc[0]
                fake_context=np.full(len(trials[:]), fill_value='nan')
                fake_block_nums=np.full(len(trials[:]), fill_value=np.nan)
                block_contexts=['vis','aud','vis','aud','vis','aud']
                for block in range(0,6):
                    block_start_time=start_time+block*10*60
                    block_end_time=start_time+(block+1)*10*60
                    block_trials=trials[:].query('start_time>=@block_start_time').index
                    fake_context[block_trials]=block_contexts[block]
                    fake_block_nums[block_trials]=block
                fake_block_index=fake_block_nums[trial_sel]
                pred_var=fake_context[trial_sel]



        elif p=='trial_response':
            #exclude any trials that had opto stimulation
            if 'opto_power' in trials[:].columns:
                trial_sel = trials[:].query('opto_power.isnull()').index
            else:
                trial_sel = trials[:].index
                
            #or, use whether mouse responded
            pred_var = trials[:]['is_response'][trial_sel].values

        if (crossval=='blockwise') | ('forecast' in crossval):
            if generate_labels == False:
                crossval_index=trials[:]['block_index'][trial_sel].values
            else:
                crossval_index=fake_block_index

            #correct for the first trial of each block actually being the previous context (if using prestim time window)
            # context_switch_trials=trials[:].query('is_context_switch').index
            # for ct in context_switch_trials:
            #     crossval_index[ct]=crossval_index[ct]-1
            svc_results['crossval_index']=crossval_index
        else:
            crossval_index=None
            svc_results['crossval_index']=None

        if all_areas == False:
            area_sel = ['all']
        else:
            area_sel = ['all']+list(area_counts[area_counts>=u_min].index)
        
        #loop through areas
        for aa in area_sel:
            if aa=='all':
                unit_sel = units[:]['unit_id'].values
            elif use_structure_probe:
                unit_sel = structure_probe.query('structure_probe==@aa')['unit_id'].values
            else:
                unit_sel = units[:].query('structure==@aa')['unit_id'].values
            svc_results[p][aa]={}
            svc_results[p][aa]['n_units']=len(unit_sel)
            
            #loop through time bins
            for tt,t_start in enumerate(time_bins[:-1]):
                svc_results[p][aa][tt]={}
                for u_idx,u_num in enumerate(n_units):
                    svc_results[p][aa][tt][u_idx]={}
                    
                    #loop through repeats
                    for nn in range(0,n_repeats):

                        if u_num=='all':
                            unit_subset = unit_sel #np.random.choice(unit_sel,len(unit_sel),replace=False)
                        elif u_num<=len(unit_sel):
                            unit_subset = np.random.choice(unit_sel,u_num,replace=False)
                        else:
                            continue

                        #option to balance number of labels for training
                        if balance_labels:
                            subset_ind=[]
                            conds = np.unique(pred_var)
                            cond_count=[]

                            if trnum=='all':
                                for cc in conds:
                                    cond_count.append(np.sum(pred_var==cc))
                                use_trnum=np.min(cond_count)
                            else:
                                use_trnum = trnum

                            for cc in conds:
                                cond_inds=np.where(pred_var==cc)[0]
                                # if len(cond_inds)<use_trnum:
                                #     use_trnum=len(cond_inds)
                                subset_ind.append(np.random.choice(cond_inds,use_trnum,replace=False))   
                            subset_ind=np.sort(np.hstack(subset_ind))
                        else:
                            subset_ind=np.arange(0,len(trial_sel))


                        sel_data = trial_da.sel(time=slice(t_start,time_bins[tt+1]),
                                                trials=trial_sel[subset_ind],
                                                unit_id=unit_subset).mean(dim='time').values
                        
                        if (crossval=='blockwise') | ('forecast' in crossval):
                            crossval_index_subset=crossval_index[subset_ind]
                        else:
                            crossval_index_subset=None

                        if decoder_type=='linearSVC':
                            svc_results[p][aa][tt][u_idx][nn]=linearSVC_decoder(
                                input_data=sel_data.T,
                                labels=pred_var[subset_ind].flatten(),
                                crossval=crossval,
                                crossval_index=crossval_index_subset,
                                labels_as_index=labels_as_index)

                            svc_results[p][aa][tt][u_idx][nn]['shuffle']=linearSVC_decoder(
                                input_data=sel_data.T,
                                labels=np.random.choice(pred_var[subset_ind],len(pred_var[subset_ind]),replace=False).flatten(),
                                crossval=crossval,
                                crossval_index=crossval_index_subset,
                                labels_as_index=labels_as_index)
                            
                        elif decoder_type=='random_forest':
                            svc_results[p][aa][tt][u_idx][nn]=random_forest_decoder(
                                input_data=sel_data.T,
                                labels=pred_var[subset_ind].flatten(),
                                crossval=crossval,
                                crossval_index=crossval_index_subset,
                                labels_as_index=labels_as_index)

                            svc_results[p][aa][tt][u_idx][nn]['shuffle']=random_forest_decoder(
                                input_data=sel_data.T,
                                labels=np.random.choice(pred_var[subset_ind],len(pred_var[subset_ind]),replace=False).flatten(),
                                crossval=crossval,
                                crossval_index=crossval_index_subset,
                                labels_as_index=labels_as_index)

                        svc_results[p][aa][tt][u_idx][nn]['trial_sel_idx']=trial_sel[subset_ind]
                        svc_results[p][aa][tt][u_idx][nn]['unit_sel_idx']=unit_subset

            print(aa+' done')
            
    print(session.id+' done')
    

    with open(os.path.join(savepath,session.id+'_'+filename), 'wb') as handle:
        pickle.dump(svc_results, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    svc_results={}


def decode_context_from_units_all_timebins(session,params):


    trnum=params['trnum']
    n_units=params['n_units']
    u_min=params['u_min']
    n_repeats=params['n_repeats']
    binsize=params['binsize']
    balance_labels=params['balance_labels']
    savepath=params['savepath']
    filename=params['filename']

    svc_results={}

    trials=pd.read_parquet(
                npc_lims.get_cache_path('trials',session.id,version='v0.0.173')
            )
    units=pd.read_parquet(
                npc_lims.get_cache_path('units',session.id,version='v0.0.173')
            )

    timebin_da,timebins_table=spike_utils.make_neuron_timebins_matrix(units, trials, binsize)

    area_counts=units['structure'].value_counts()
    
    # predict=['stim_ids','block_ids','trial_response']
    predict=['block_ids']

    #save metadata about this session & decoder params
    svc_results['metadata']=session.metadata
    svc_results['trial_numbers']=trnum
    svc_results['unit_numbers']=n_units
    svc_results['min_n_units']=u_min
    svc_results['n_repeats']=n_repeats
    svc_results['balance_labels']=balance_labels
    
    #loop through different labels to predict
    for p in predict:
        svc_results[p]={}

        timebin_context=[]
        for cc in range(0,len(timebins_table)):
            if timebins_table['is_vis_context'].iloc[cc]:
                timebin_context.append('vis')
            else: #timebins_table['is_aud_context'].iloc[cc]:
                timebin_context.append('aud')

        timebins_table['context']=timebin_context
        pred_var = timebins_table['context'].values

        area_sel = ['all']+list(area_counts[area_counts>=u_min].index)
        
        #loop through areas
        for aa in area_sel:
            if aa=='all':
                unit_sel = units[:].index.values
            else:
                unit_sel = units[:].query('structure==@aa').index.values
            svc_results[p][aa]={}
            svc_results[p][aa]['n_units']=len(unit_sel)
            
            # since time bins are observatinos here, no need to loop through diff time bins
            # keep the index for analysis consistency
            tt=0
            svc_results[p][aa][tt]={}
            for u_idx,u_num in enumerate(n_units):
                svc_results[p][aa][tt][u_idx]={}
                
                #loop through repeats
                for nn in range(0,n_repeats):

                    if u_num=='all':
                        unit_subset = unit_sel #np.random.choice(unit_sel,len(unit_sel),replace=False)
                    elif u_num<=len(unit_sel):
                        unit_subset = np.random.choice(unit_sel,u_num,replace=False)
                    else:
                        continue

                    #option to balance number of labels for training
                    if balance_labels:
                        subset_ind=[]
                        conds = np.unique(pred_var)
                        cond_count=[]

                        if trnum=='all':
                            for cc in conds:
                                cond_count.append(np.sum(pred_var==cc))
                            use_trnum=np.min(cond_count)
                        else:
                            use_trnum = trnum

                        for cc in conds:
                            cond_inds=np.where(pred_var==cc)[0]
                            # if len(cond_inds)<use_trnum:
                            #     use_trnum=len(cond_inds)
                            subset_ind.append(np.random.choice(cond_inds,use_trnum,replace=False))   
                        subset_ind=np.sort(np.hstack(subset_ind))
                    else:
                        subset_ind=np.arange(0,len(pred_var))


                    sel_data = timebin_da.sel(timebin=subset_ind,
                                              unit_id=units[:]['unit_id'].loc[unit_subset].values
                                              ).values

                    svc_results[p][aa][tt][u_idx][nn]=linearSVC_decoder(
                        input_data=sel_data.T,
                        labels=pred_var[subset_ind].flatten())

                    svc_results[p][aa][tt][u_idx][nn]['shuffle']=linearSVC_decoder(
                        input_data=sel_data.T,
                        labels=np.random.choice(pred_var[subset_ind],len(pred_var[subset_ind]),replace=False).flatten())

                    svc_results[p][aa][tt][u_idx][nn]['trial_sel_idx']=subset_ind
                    svc_results[p][aa][tt][u_idx][nn]['unit_sel_idx']=unit_subset
                    

            print(aa+' done')
            
    print(session.id+' done')
    

    with open(os.path.join(savepath,session.id+'_'+filename), 'wb') as handle:
        pickle.dump(svc_results, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    svc_results={}