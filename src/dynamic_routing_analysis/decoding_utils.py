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
from sklearn.preprocessing import RobustScaler, StandardScaler

from dynamic_routing_analysis import data_utils, spike_utils

logger = logging.getLogger(__name__)

class NotEnoughBlocksError(Exception):
    pass



def decoder_helper(input_data,labels,decoder_type='linearSVC',crossval='5_fold',
                   crossval_index=None,labels_as_index=False,train_test_split_input=None,
                   regularization=None,penalty=None,solver=None,n_jobs=None,set_random_state=None,
                   other_data=None,scaler='robust'):
    
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
                      C=regularization,penalty=penalty)
    elif decoder_type=='nonlinearSVC':
        from sklearn.svm import SVC
        kernel='rbf'
        if regularization is None:
            regularization = 1.0
        if penalty is not None:
            logger.warning('Penalty not used for non-linear SVC')
        if solver is not None:
            logger.warning('Solver not used for non-linear SVC')
        clf=SVC(max_iter=5000,probability=True,class_weight='balanced',
                      C=regularization,kernel=kernel)
    elif decoder_type=='LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        if solver is None:
            solver = 'svd'
        if regularization is not None:
            logger.warning('Regularization not used for LDA')
        if penalty is not None:
            logger.warning('Penalty not used for LDA')
        clf=LinearDiscriminantAnalysis(solver=solver)
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

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()

    if scaler == 'none':
        X = input_data
    else:
        scaler.fit(input_data)
        X = scaler.transform(input_data)

    unique_labels=np.unique(labels)
    if labels_as_index==True:
        labels=np.array([np.where(unique_labels==x)[0][0] for x in labels])
    y = labels

    if other_data is not None:
        if scaler == 'none':
            X_other = other_data
        else:
            X_other = scaler.transform(other_data)

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
    if len(np.unique(labels))>2:
        decision_function=np.full((len(y),len(np.unique(labels))), fill_value=np.nan)
    else:
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
    models=[]
    cr_dict_train = []
    balanced_accuracy_train = []
    coefs_all = []

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

            # #equalize number of trials for each condition
            # subset_ind=[]
            # conds = np.unique(y[not_block_inds])
            # cond_count=[]
            # for cc in conds:
            #     cond_count.append(np.sum(y[not_block_inds]==cc))
            # use_trnum=np.min(cond_count)
            # for cc in conds:
            #     cond_inds=np.where(y[not_block_inds]==cc)[0]
            #     subset_ind.append(np.random.choice(cond_inds,use_trnum,replace=False))
            # subset_ind=np.sort(np.hstack(subset_ind))
            # train.append(not_block_inds[subset_ind])

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

    elif crossval=='5_fold_constant' or crossval=='custom':
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
        tidx_used.append([test])
        classes.append(clf.classes_)
        # intercept.append(clf.intercept_)
        params.append(clf.get_params())
        train_trials.append(train)
        test_trials.append(test)

        if decoder_type == 'LDA' or decoder_type == 'RandomForest' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
            ypred_proba[test,:] = clf.predict_proba(X[test])
        else:
            ypred_proba[test,:] = np.full((len(test),len(np.unique(labels))), fill_value=False)

        if decoder_type=='LDA' or decoder_type=='linearSVC' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
            if len(np.unique(labels))>2:
                decision_function[test,:]=clf.decision_function(X[test])
            else:
                decision_function[test]=clf.decision_function(X[test])
        else:
            if len(np.unique(labels))>2:
                decision_function[test,:]=np.full((len(test),len(np.unique(labels))), fill_value=False)
            else:
                decision_function[test]=np.full((len(test)), fill_value=False)

        if decoder_type == 'LDA' or decoder_type == 'linearSVC' or decoder_type == 'LogisticRegression':
            coefs_all.append(clf.coef_)
        else:
            coefs_all.append(np.full((X.shape[1]), fill_value=False))

        models.append(clf)

    #fit on all trials
    clf.fit(X, y)
    ypred = clf.predict(X)

    if other_data is not None:
        ypred_other = clf.predict(X_other)

    if decoder_type == 'LDA' or decoder_type == 'RandomForest' or decoder_type=='LogisticRegression' or decoder_type=='nonlinearSVC':
        predict_proba_all_trials = clf.predict_proba(X)
        if other_data is not None:
            predict_proba_other = clf.predict_proba(X_other)

    else:
        predict_proba_all_trials = np.full((X.shape[0],len(np.unique(labels))), fill_value=False)
        if other_data is not None:
            predict_proba_other = np.full((X_other.shape[0],len(np.unique(labels))), fill_value=False)

    if decoder_type == 'LDA' or decoder_type == 'linearSVC' or decoder_type == 'LogisticRegression':
        coefs = clf.coef_
        intercept = clf.intercept_
        dec_func_all_trials = clf.decision_function(X)
        if other_data is not None:
            dec_func_other = clf.decision_function(X_other)
    elif decoder_type == 'nonlinearSVC':
        coefs = np.full((X.shape[1]), fill_value=False)
        intercept = np.full((1), fill_value=False)
        dec_func_all_trials = clf.decision_function(X)
        if other_data is not None:
            dec_func_other = clf.decision_function(X_other)
    else:
        coefs = np.full((X.shape[1]), fill_value=False)
        intercept = np.full((1), fill_value=False)
        dec_func_all_trials = np.full((X.shape[0]), fill_value=np.nan)
        if other_data is not None:
            dec_func_other = np.full((X_other.shape[0]), fill_value=np.nan)

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
    output['coefs_all']=coefs_all
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

    #predicted labels for other data input (if provided)
    if other_data is not None:
        output['pred_label_other']=ypred_other
        output['decision_function_other']=dec_func_other
        output['predict_proba_other']=predict_proba_other
    else:
        output['pred_label_other']=None
        output['decision_function_other']=None
        output['predict_proba_other']=None

    return output


