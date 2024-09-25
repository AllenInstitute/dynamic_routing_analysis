import sys
sys.path.append(r"C:\Users\shailaja.akella\Dropbox (Personal)\DR\dynamic_routing_analysis_ethan\src")
import npc_lims
from dynamic_routing_analysis import decoding_utils
from npc_sessions import DynamicRoutingSession

# 1A get all uploaded & annotated ephys sessions
# ephys_sessions = tuple(s for s in npc_lims.get_session_info(is_ephys=True, is_uploaded=True, is_annotated=True))

# 2 set savepath and filename
savepath = r"C:\Users\shailaja.akella\Dropbox (Personal)\DR\dynamic_routing_analysis_ethan\results"
filename = 'decoding_results_linear_shift_20_units_re_run.pkl'

except_list = {}

# 3 set parameters
# linear shift decoding currently just takes the average firing rate over all bins defined here
spikes_binsize = 0.2  # bin size in seconds
spikes_time_before = 0.2  # time before the stimulus per trial
spikes_time_after = 0.01  # time after the stimulus per trial

# #not used for linear shift decoding, were used in a previous iteration of decoding analysis
# decoder_binsize=0.2
# decoder_time_before=0.2
# decoder_time_after=0.1

params = {
    'n_units': 20,  # number of units to sample for each area
    'n_repeats': 25,  # number of times to repeat decoding with different randomly sampled units
    'input_data_type': 'LP',  # spikes or facemap or LP
    'vid_angle_facemotion': 'face', # behavior, face, eye
    'vid_angle_LP': 'behavior',
    'central_section': '4_blocks_plus',
    # for linear shift decoding, how many trials to use for the shift. '4_blocks_plus' is best
    'exclude_cue_trials': False,  # option to totally exclude autorewarded trials
    'n_unit_threshold': 20,  # minimum number of units to include an area in the analysis
    'keep_n_SVDs': 500,  # number of SVD components to keep for facemap data
    'LP_parts_to_keep': ['ear_base_l', 'eye_bottom_l', 'jaw', 'nose_tip', 'whisker_pad_l_side'],
    'spikes_binsize': spikes_binsize,
    'spikes_time_before': spikes_time_before,
    'spikes_time_after': spikes_time_after,
    # 'decoder_binsize':decoder_binsize,
    # 'decoder_time_before':decoder_time_before,
    # 'decoder_time_after':decoder_time_after,
    'savepath': savepath,
    'filename': filename,
    'use_structure_probe': True,  # if True, appedn probe name to area name when multiple probes in the same area
    'crossval': '5_fold',  # '5_fold' or 'blockwise' - blockwise untested with linear shift
    'labels_as_index': True,  # convert labels (context names) to index [0,1]
    'decoder_type': 'linearSVC',  # 'linearSVC' or 'LDA' or 'RandomForest' or 'LogisticRegression'
}
# ephys_sessions[:]

for ephys_session in ['713655_2024-08-05']:
    try:
        session = DynamicRoutingSession(ephys_session)
        print(session.id + ' loaded')
        if 'structure' in session.electrodes[:].columns:
            decoding_utils.decode_context_with_linear_shift(session, params)
        else:
            print('no structure column found in electrodes table, moving to next recording')
        session = []
    except Exception as e:
        except_list[session.id] = repr(e)
