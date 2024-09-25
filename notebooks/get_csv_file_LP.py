import sys
sys.path.append(r"C:\Users\shailaja.akella\Dropbox (Personal)\DR\dynamic_routing_analysis_ethan\src")

import npc_lims
from npc_sessions import DynamicRoutingSession
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from dynamic_routing_analysis import decoding_utils
import pandas as pd


savepath = r"C:\Users\shailaja.akella\Dropbox (Personal)\DR\dynamic_routing_analysis_ethan\results\all_features_LP_updated"

files=glob.glob(os.path.join(savepath,'*LP.pkl'))

# load all trialwise decoding results, option to concatenate all sessions
concat_session_results=True
save_tables=True
return_table=True
if save_tables:
    savepath=savepath
else:
    savepath=None

if concat_session_results:
    combined_results=decoding_utils.concat_trialwise_decoder_results(files,savepath=savepath,return_table=return_table)
    if return_table:
        decoder_confidence_versus_response_type=combined_results[0]
        decoder_confidence_dprime_by_block=combined_results[1]
        decoder_confidence_by_switch=combined_results[2]
        decoder_confidence_versus_trials_since_rewarded_target=combined_results[3]
        decoder_confidence_before_after_target=combined_results[4]
else:
    decoder_confidence_versus_response_type=pd.read_csv(os.path.join(savepath,'decoder_confidence_versus_response_type.csv'))
    decoder_confidence_dprime_by_block=pd.read_csv(os.path.join(savepath,'decoder_confidence_dprime_by_block.csv'))
    decoder_confidence_by_switch=pd.read_csv(os.path.join(savepath,'decoder_confidence_by_switch.csv'))
    decoder_confidence_versus_trials_since_rewarded_target=pd.read_csv(os.path.join(loadpath,'decoder_confidence_versus_trials_since_rewarded_target.csv'))
    decoder_confidence_before_after_target=pd.read_csv(os.path.join(loadpath,'decoder_confidence_before_after_target.csv'))