import os
import re

import io_utils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import pandas as pd
import polars as pl


def get_session_ids():
    session_ids = pl.read_parquet(npc_lims.get_cache_path('session', version='0.0.265')).filter(
                pl.col("keywords").list.contains("production"),
                ~pl.col("keywords").list.contains("issues"),
                pl.col("keywords").list.contains("task"),
                pl.col("keywords").list.contains("ephys"),
                pl.col("keywords").list.contains("ccf"),
                ~pl.col("keywords").list.contains("opto_perturbation"),
                ~pl.col("keywords").list.contains("opto_control"),
                ~pl.col("keywords").list.contains("injection_perturbation"),
                ~pl.col("keywords").list.contains("injection_control"),
                ~pl.col("keywords").list.contains("hab"),
                ~pl.col("keywords").list.contains("training"),
                ~pl.col("keywords").list.contains("context_naive"),
                ~pl.col("keywords").list.contains("templeton"),
)['session_id']
    return session_ids


def coding_score(drop_r2, full_r2):
    score = (1 - drop_r2/full_r2)
    if isinstance(score, np.ndarray) and len(score) > 1:
        score[score < 0] = 0
        score[drop_r2 < 0.005] = 0
        score[full_r2 < 0.005] = 0
    else:
        if score < 0:
            score = 0
        if drop_r2 < 0.005:
            score = 0
        if full_r2 < 0.005:
            score = 0
    return score


def get_df(fit, model_label):
    r2 = np.nanmean(fit[model_label]['cv_var_test'], axis = 1)
    ids = fit['spike_count_arr']['unit_id']
    areas = fit['spike_count_arr']['structure']

    # Create a DataFrame for fullmodel data
    full_df = pd.DataFrame({
        f'r2_{model_label}': r2,
        'unit_id': ids,
        'session_id': fit["session_id"],
        'areas':areas,
        'av_dprime': np.nanmean(fit['dprime']),
        'n_good_blocks': len(fit['dprime'][fit['dprime'] > 1]),
    })

    return full_df


def get_fit_dict(file):
    try:
        fit_dict = np.load(file, allow_pickle=True)
    except FileNotFoundError:
        try:
            fit_dict = np.load(file.replace('.npz', '_0.npz'), allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Neither '{file}' nor its fallback '{file.replace('.npz', '_0.npz')}' exists.")

    fit = {key: fit_dict[key][()] for key in fit_dict.files}
    pattern = r'(\d{6,})_(\d{4}-\d{2}-\d{2})'
    fit['session_id'] = '_'.join(re.findall(pattern, file)[0]) if re.findall(pattern, file) else None
    return fit


def get_session_info(session_id, run_on_qc_units = False, version = '0.0.265', spike_bin_width = 0.025, unit_ids = None):
    # Set up run parameters
    params = io_utils.RunParams(session_id=session_id)
    params.update_multiple_metrics({"cache_version": version,
                                    "time_of_interest": "full_trial",
                                    "run_on_qc_units": run_on_qc_units,
                                    "unit_ids": unit_ids,
                                    "spike_bin_width": spike_bin_width,
                                    "unit_inclusion_criteria": {'presence_ratio': 0.7}})

    params.validate_params()
    run_params = params.get_params()

    # Capsule 1 - GLM_inputs
    session, units_table, behavior_info, _ = io_utils.get_session_data(session_id, version = run_params['cache_version'])
    fit = io_utils.extract_unit_data(run_params, units_table, behavior_info)
    design = io_utils.DesignMatrix(fit)
    design, fit = io_utils.add_kernels(design, run_params, session, fit, behavior_info)
    X = design.get_X()
    return X, fit

def boxcar_filter(size):
    return np.ones(size) / size

def convolve_time_series(time_series, filter_size = 5):
    box_filter = boxcar_filter(filter_size)
    # return np.convolve(time_series, box_filter, mode='full')[1:len(time_series) + 1]
    return np.convolve(time_series, box_filter, mode='same')


def get_psth(rate, event_time_samples, bin_width = 0.025, time_before = 2, time_after = 3):
    psth = []
    for t in event_time_samples:
        start_time = t - int(time_before/bin_width)
        end_time = t + int(time_after/bin_width)
        rate_snippet = rate[start_time:end_time]
        if rate_snippet.size > 0 and len(rate_snippet) == int((time_before + time_after)/bin_width):
            psth.append(rate[t - int(time_before/bin_width): t + int(time_after/bin_width)])
    return np.nanmean(np.array(psth), axis = 0), np.nanstd(np.array(psth), axis = 0)/np.sqrt(len(psth))


def plot_fit(fit, design_matrix, fitted_dict, unit_ids, save_fig = False, save_path = 'figures/context_units_psth/linear_shift/'):
    df = pd.read_pickle('results/linear_shift.pkl')
    spike_counts = fit['spike_count_arr']['spike_counts'].T
    spike_rate = np.array([convolve_time_series(counts, 5)/fit['spike_bin_width'] for counts in spike_counts])

    model_name = 'fullmodel, drop_context, shift_context'
    fullmodel_weights = fitted_dict[model_name]['weights']
    fullmodel_pred = np.dot(design_matrix.data, fullmodel_weights).T
    fullmodel_rate = np.array([convolve_time_series(pred, 5)/fit['spike_bin_width'] for pred in fullmodel_pred])

    sub_select_units = df[(df['unit_id'].isin(unit_ids))]

    for _, row in sub_select_units.iterrows():
        # plots
        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(4, 1, figure=fig)  # Changed from 5 to 4
        unit_pos_spikes = np.where(fit['spike_count_arr']['unit_id'] == row['unit_id'])[0][0]
        unit_pos_fullmodel = np.where(fitted_dict['spike_count_arr']['unit_id'] == row['unit_id'])[0][0]
        # Combine the first row into a single plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(fit['bin_centers'], spike_rate[unit_pos_spikes], color='k', alpha=0.8,
                 label = 'truth', lw = 0.5)
        ax1.plot(fit['bin_centers'], fullmodel_rate[unit_pos_fullmodel], color='r',
                 alpha=0.8, label = 'fullmodel', lw = 0.5)
        ax1.set_title(f'Unit id: {row["unit_id"]}, area: {row["structure"]}\ncontext-score-linear-shift: {row["context_score_LS"]:.3f}, delta_linear_shift: {row["relative_diff_LS"]:.4f}, \ncontext-score-dropout: {row["context_score_DO"]:.3f}, delta_dropout: {row["relative_diff_DO"]:.4f}, \nFullmodel, cvr2: {row["test_cvr2"]:.3f}, Context-free: {row["dropout_cvr2"]: .3f}, activity-drift: {row["activity_drift"]:0.2f}\ndprimes: {fitted_dict["dprime"]}')
        ax1.set_ylabel('Spike rate (spikes/s)')
        ax1.legend()
        licks = design_matrix.sel(weights = 'licks_0').data.copy()
        licks[licks == 0] = np.nan
        y_max = np.nanmax(spike_rate[unit_pos_spikes])
        y_min = np.nanmin(spike_rate[unit_pos_spikes])
        ax1.plot(fit['bin_centers'], licks*y_max, color='k',  lw = 0.3, marker = '|')
        hits = design_matrix.sel(weights = 'hit_0').data.copy()
        hits[hits == 0] = np.nan
        ax1.plot(fit['bin_centers'], hits*(y_max + y_max/10), color='k', marker = 'o',
                 ms = 2,mfc = 'None')
        context = design_matrix.sel(weights = 'context_0').data.copy()
        context[context == 1] = np.nan
        ax1.fill_between(fit['bin_centers'], 0, y_max, where=np.isnan(context),
                         color='dimgray', alpha=0.2)
        ax1.set_xlim(fit['bin_centers'][0], fit['bin_centers'][-1])
        ax1.set_ylim(y_min, y_max+y_max/5)

        diffs = np.abs(np.diff(design_matrix.sel(weights = 'context_0').data.copy()))
        switch_trials = [0] + list(np.where(diffs)[0]) + [len(diffs)]
        # Combine the second row into a single plot
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(fit['bin_centers'], spike_rate[unit_pos_spikes], color='k', alpha=0.8,
                 label = 'truth', lw = 0.5)
        ax2.plot(fit['bin_centers'], fullmodel_rate[unit_pos_fullmodel], color='r',
                 alpha=0.8, label = 'fullmodel', lw = 0.5)
        ax2.set_ylabel('Spike rate (spikes/s)')
        ax2.plot(fit['bin_centers'], licks*y_max, color='k',  lw = 0.3, marker = '|')
        ax2.plot(fit['bin_centers'], hits*(y_max + y_max/10), color='k', marker = 'o',
                 ms = 2,mfc = 'None')
        blk_start = switch_trials[0]
        blk_end = switch_trials[1]
        for stim in ['vis1', 'sound1', 'sound2', 'vis2']:
            stim_vector = design_matrix.sel(weights = f'{stim}_0').data.copy()
            stim_locs = fit['bin_centers'][stim_vector == 1]
            ax2.vlines(stim_locs, ymin=0, ymax=y_max, color='silver', lw=0.5)
        ax2.set_xlim(fit['bin_centers'][blk_start], fit['bin_centers'][blk_end])
        ax2.set_ylim(y_min, y_max+y_max/5)
        context_blk = 'Visual block' if design_matrix.sel(weights = 'context_0').data.copy()[blk_start+1] == 1 else 'Auditory block'
        ax2.set_title(f"Block 1, {context_blk}")

        # New plot between ax2 and ax3
        ax_new = fig.add_subplot(gs[2, :])
        ax_new.plot(fit['bin_centers'], spike_rate[unit_pos_spikes], color='k',
                    alpha=0.8, label = 'truth', lw = 0.5)
        ax_new.plot(fit['bin_centers'], fullmodel_rate[unit_pos_fullmodel], color='r',
                    alpha=0.8, label = 'fullmodel', lw = 0.5)
        ax_new.set_ylabel('Spike rate (spikes/s)')
        ax_new.plot(fit['bin_centers'], licks*y_max, color='k',  lw = 0.3, marker = '|')
        ax_new.plot(fit['bin_centers'], hits*(y_max + y_max/10), color='k', marker = 'o',
                    ms = 2,mfc = 'None')
        blk_start = switch_trials[1]
        blk_end = switch_trials[2]
        for stim in ['vis1', 'sound1', 'sound2', 'vis2']:
                stim_vector = design_matrix.sel(weights = f'{stim}_0').data.copy()
                stim_locs = fit['bin_centers'][stim_vector == 1]
                ax_new.vlines(stim_locs, ymin=0, ymax=y_max, color='silver', lw=0.5)
        ax_new.set_xlim(fit['bin_centers'][blk_start], fit['bin_centers'][blk_end])
        ax_new.set_ylim(y_min, y_max+y_max/5)
        context_blk = 'Visual block' if design_matrix.sel(weights = 'context_0').data.copy()[blk_start+1] == 1 else 'Auditory block'
        ax_new.set_title(f"Block 2, {context_blk}")

        # Fourth row with two plots
        gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, :])
        ax4 = fig.add_subplot(gs4[0, 0])
        ax5 = fig.add_subplot(gs4[0, 1])

        cntxt = 1 # vis
        context = design_matrix.sel(weights = 'context_0').data.copy()
        c = ['tab:green', 'tab:blue']
        for s_no, stim in enumerate(['vis1', 'sound1']):
                stim_vector = design_matrix.sel(weights = f'{stim}_0').data.copy()
                stim_vector = stim_vector * context
                stim_locs_sample = np.where(stim_vector == cntxt)[0]
                spike_rate_stim, spike_sem = get_psth(spike_rate[unit_pos_spikes],
                                                      stim_locs_sample)
                if stim == 'vis1':
                    ax4.plot(np.linspace(-2, 3, len(spike_rate_stim)), spike_rate_stim,
                             label = 'visual context', color = 'tab:green',lw = 1)
                    ax4.fill_between(np.linspace(-2, 3, len(spike_rate_stim)),
                                     spike_rate_stim - spike_sem,
                                     spike_rate_stim + spike_sem,
                                     alpha = 0.2, color = 'tab:green', ls = 'None')
                else:
                    ax5.plot(np.linspace(-2, 3, len(spike_rate_stim)), spike_rate_stim,
                             label = 'visual context', color = 'tab:green' ,lw = 1)
                    ax5.fill_between(np.linspace(-2, 3, len(spike_rate_stim)),
                                     spike_rate_stim - spike_sem,
                                     spike_rate_stim + spike_sem,
                                     alpha = 0.2, color = 'tab:green', ls = 'None')

        context = design_matrix.sel(weights = 'context_0').data.copy()
        cntxt =  -1
        for _, stim in enumerate(['vis1', 'sound1']):
                stim_vector = design_matrix.sel(weights = f'{stim}_0').data.copy()
                stim_vector = stim_vector * context
                stim_locs_sample = np.where(stim_vector == cntxt)[0]
                spike_rate_stim, spike_sem = get_psth(spike_rate[unit_pos_spikes],
                                                      stim_locs_sample)
                if stim == 'vis1':
                    ax4.plot(np.linspace(-2, 3, len(spike_rate_stim)), spike_rate_stim,
                             label = 'auditory context', color = "tab:blue" ,lw = 1)
                    ax4.fill_between(np.linspace(-2, 3, len(spike_rate_stim)),
                                     spike_rate_stim - spike_sem,
                                     spike_rate_stim + spike_sem,
                                     alpha = 0.2, color = "tab:blue", ls = 'None')
                else:
                    ax5.plot(np.linspace(-2, 3, len(spike_rate_stim)), spike_rate_stim,
                             label = 'auditory context', color = "tab:blue",lw = 1)
                    ax5.fill_between(np.linspace(-2, 3, len(spike_rate_stim)),
                                     spike_rate_stim - spike_sem, spike_rate_stim + spike_sem,
                                     alpha = 0.2, color = "tab:blue", ls = 'None')

        ax4.set_ylabel('Spike rate (spikes/s)', fontsize=12)
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_title('Visual stimulus', fontsize=12)
        ax4.set_xlim(-2, 3)
        ax4.legend()
        ax4.tick_params(axis='both', which='major', labelsize=12)

        ax5.set_ylabel('Spike rate (spikes/s)', fontsize=12)
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_title('Sound stimulus', fontsize=12)
        ax5.set_xlim(-2, 3)
        handles, labels = ax5.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax5.legend(by_label.values(), by_label.keys())
        ax5.tick_params(axis='both', which='major', labelsize=12)


        # Get the y-limits from both axes
        ymin_ax4, ymax_ax4 = ax4.get_ylim()
        ymin_ax5, ymax_ax5 = ax5.get_ylim()

        # Calculate the combined y-limits
        ymin = min(ymin_ax4, ymin_ax5)
        ymax = max(ymax_ax4, ymax_ax5)

        # Set the same y-limits for both axes
        ax4.set_ylim(ymin, ymax)
        ax5.set_ylim(ymin, ymax)

        axes = [ax1, ax2, ax_new, ax4, ax5]
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_fig:
            os.makedirs(save_path, exist_ok = True)
            plt.savefig(f"{save_path}/{row['unit_id']}.png", dpi = 300)
            plt.close()
