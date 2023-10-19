import os
import pickle

import numpy as np
import xarray as xr
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dynamic_routing_analysis import spike_utils


def linearSVC_decoder(input_data,labels):
    
    output={}

    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    
    scaler.fit(input_data)
    X = scaler.transform(input_data)
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

    for train,test in skf.split(X, y):
        clf=svm.LinearSVC(max_iter=5000)
        clf.fit(X[train],y[train])
        ypred[test] = clf.predict(X[test])
        y_dec_func[test] = clf.decision_function(X[test])
        tidx_used.append([test])
        coefs.append(clf.coef_)
        classes.append(clf.classes_)
        intercept.append(clf.intercept_)
        params.append(clf.get_params())

    cr_dict=classification_report(y, ypred, output_dict=True)

    output['cr']=cr_dict
    output['pred_label']=ypred
    output['true_label']=y
    output['trials_used']=tidx_used
    output['decision_function']=y_dec_func
    output['coefs']=coefs
    output['classes']=classes
    output['intercept']=intercept
    output['params']=params
    
    return output


def decode_context_from_units(session,params):


    trnum=params['trnum']
    n_units=params['n_units']
    u_min=params['u_min']
    n_repeats=params['n_repeats']
    binsize=params['binsize']
    time_bins=params['time_bins']
    balance_labels=params['balance_labels']
    savepath=params['savepath']
    filename=params['filename']

    svc_results={}
    
    # make unit xarrays
    time_before = 0.5
    time_after = 0.5
    binsize = 0.1
    trial_da = spike_utils.make_neuron_time_trials_tensor(session.units, session.trials, time_before, time_after, binsize)

    area_counts=session.units[:]['structure'].value_counts()
    
#     predict=['stim_ids','block_ids','trial_response']
    predict=['block_ids']

    #save metadata about this session & decoder params
    svc_results['metadata']=session.metadata
    svc_results['trial_numbers']=trnum
    svc_results['unit_numbers']=n_units
    svc_results['min_n_units']=u_min
    svc_results['n_repeats']=n_repeats
    svc_results['time_bins']=time_bins
    svc_results['balance_labels']=balance_labels
    
    #loop through different labels to predict
    for p in predict:
        svc_results[p]={}
    
        #choose what variable to predict
        if p=='stim_ids':
            #exclude any trials that had opto stimulation
            if 'opto_power' in session.trials[:].columns:
                trial_sel = session.trials[:].query('opto_power.isnull() and stim_name != "catch"').index
            else:
                trial_sel = session.trials[:].query('stim_name != "catch"').index
                
            # grab the stimulus ids
            pred_var = session.trials[:]['stim_name'][trial_sel].values
    
        elif p=='block_ids':
            #exclude any trials that had opto stimulation
            if 'opto_power' in session.trials[:].columns:
                trial_sel = session.trials[:].query('opto_power.isnull()').index
            else:
                trial_sel = session.trials[:].index
                
            # or, use block IDs
            pred_var = session.trials[:]['context_name'][trial_sel].values
    
        elif p=='trial_response':
            #exclude any trials that had opto stimulation
            if 'opto_power' in session.trials[:].columns:
                trial_sel = session.trials[:].query('opto_power.isnull()').index
            else:
                trial_sel = session.trials[:].index
                
            #or, use whether mouse responded
            pred_var = session.trials[:]['is_rewarded'][trial_sel].values


        area_sel = ['all']+list(area_counts[area_counts>=u_min].index)
        
        #loop through areas
        for aa in area_sel:
            if aa=='all':
                unit_sel = session.units[:].index.values
            else:
                unit_sel = session.units[:].query('structure==@aa').index.values
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

                        svc_results[p][aa][tt][u_idx][nn]=linearSVC_decoder(
                            input_data=sel_data.T,
                            labels=pred_var[subset_ind].flatten())

                        svc_results[p][aa][tt][u_idx][nn]['shuffle']=linearSVC_decoder(
                            input_data=sel_data.T,
                            labels=np.random.choice(pred_var[subset_ind],len(pred_var),replace=False).flatten())

                        svc_results[p][aa][tt][u_idx][nn]['trial_sel_idx']=trial_sel
                        svc_results[p][aa][tt][u_idx][nn]['unit_sel_idx']=unit_subset
                    

            print(aa+' done')
            
    print(session.id+' done')
    

    with open(os.path.join(savepath,session.id+'_'+filename), 'wb') as handle:
        pickle.dump(svc_results, handle, protocol=pickle.HIGHEST_PROTOCOL) 
