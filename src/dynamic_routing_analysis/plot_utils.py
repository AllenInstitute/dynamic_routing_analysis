from __future__ import annotations

import functools
import logging
import os
import pickle
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import altair as alt
import geopandas as gpd
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt
import npc_lims
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import rasterio.features
import shapely
import shapely.geometry
from matplotlib import patches

from dynamic_routing_analysis import ccf_utils, spike_utils

logger = logging.getLogger(__name__)


def plot_unit_by_id(sel_unit, save_path=None, show_metric=None):

    unit_df = (
        ds.dataset(
            npc_lims.get_cache_path(
                "units", session_id=sel_unit[:17], version="0.0.214"
            )
        )
        .to_table(filter=(ds.field("unit_id") == sel_unit))
        .to_pandas()
    )
    session_id = (
        str(unit_df["subject_id"].values[0]) + "_" + str(unit_df["date"].values[0])
    )

    trials = pd.read_parquet(
        npc_lims.get_cache_path("trials", session_id, version="any")
    )

    time_before = 0.5
    time_after = 1.0
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(
        unit_df, trials, time_before, time_after, binsize
    )

    ##plot PSTH with context differences -- subplot for each stimulus
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax = ax.flatten()

    stims = ["vis1", "vis2", "sound1", "sound2"]

    for st, stim in enumerate(stims):

        stim_trials = trials[:].query("stim_name==@stim")

        vis_context_spikes = trial_da.sel(
            trials=stim_trials.query("is_vis_context").index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        aud_context_spikes = trial_da.sel(
            trials=stim_trials.query("is_aud_context").index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        ax[st].plot(
            vis_context_spikes.time,
            vis_context_spikes.values,
            label="vis context",
            color="tab:green",
        )
        ax[st].plot(
            aud_context_spikes.time,
            aud_context_spikes.values,
            label="aud context",
            color="tab:blue",
        )
        ax[st].axvline(0, color="k", linestyle="--", alpha=0.5)
        ax[st].axvline(0.5, color="k", linestyle="--", alpha=0.5)
        ax[st].set_title(stim)
        ax[st].legend()
        ax[st].set_xlim([-0.5, 1.0])

        if st > 1:
            ax[st].set_xlabel("time (s)")
        if st == 0 or st == 2:
            ax[st].set_ylabel("spikes/s")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_context_modulation.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()

    # plot lick vs. not lick
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3), sharex=True, sharey=True)
    ax = ax.flatten()
    stims = ["vis1", "sound1"]
    for st, stim in enumerate(stims):

        if stim == "vis1":
            sel_context = "aud"
        elif stim == "sound1":
            sel_context = "vis"
        stim_trials = trials[:].query("stim_name==@stim and context_name==@sel_context")

        lick_spikes = trial_da.sel(
            trials=stim_trials.query("is_response").index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        non_lick_spikes = trial_da.sel(
            trials=stim_trials.query("not is_response").index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        ax[st].plot(
            lick_spikes.time, lick_spikes.values, label="licks", color="tab:red"
        )
        ax[st].plot(
            non_lick_spikes.time,
            non_lick_spikes.values,
            label="non licks",
            color="tab:purple",
        )
        ax[st].axvline(0, color="k", linestyle="--", alpha=0.5)
        ax[st].axvline(0.5, color="k", linestyle="--", alpha=0.5)
        ax[st].set_title(stim + "; " + sel_context + " context")
        ax[st].legend()
        ax[st].set_xlim([-0.5, 1.0])

        if st > 1:
            ax[st].set_xlabel("time (s)")
        if st == 0 or st == 2:
            ax[st].set_ylabel("spikes/s")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_lick_modulation.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()


def plot_stim_response_by_unit_id(sel_unit, save_path=None, show_metric=None):
    unit_df = (
        ds.dataset(
            npc_lims.get_cache_path("units", session_id=sel_unit[:17], version="any")
        )
        .to_table(filter=(ds.field("unit_id") == sel_unit))
        .to_pandas()
    )
    session_id = (
        str(unit_df["subject_id"].values[0]) + "_" + str(unit_df["date"].values[0])
    )

    trials = pd.read_parquet(
        npc_lims.get_cache_path("trials", session_id, version="any")
    )

    ##plot rasters - target only
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
    ax = ax.flatten()

    stims = ["vis1", "sound1"]
    stim_names = ["vis", "aud"]

    for st, _stim in enumerate(stims):

        stim_trials = trials[:].query("stim_name==@stim")
        ax[st].axvline(0, color="k", linestyle="-", alpha=0.5)
        ax[st].set_title(stim_names[st])
        ax[st].set_xlim([-0.2, 0.5])
        ax[st].set_xlabel("time (s)")
        if st == 0:
            ax[st].set_ylabel("trials")

        for it, (_tt, trial) in enumerate(stim_trials.iterrows()):
            stim_start = trial["stim_start_time"]
            spikes = unit_df["spike_times"].iloc[0] - stim_start

            trial_spike_inds = (spikes >= -0.2) & (spikes <= 0.5)
            if sum(trial_spike_inds) == 0:
                continue
            else:
                spikes = spikes[trial_spike_inds]

            ax[st].vlines(spikes, ymin=it, ymax=it + 1, linewidth=0.75, color="k")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_target_stim_rasters.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()

    time_before = 0.2
    time_after = 0.5
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(
        unit_df, trials, time_before, time_after, binsize
    )

    ##plot PSTHs - target only
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), sharex=True, sharey=True)
    ax = ax.flatten()

    stims = ["vis1", "sound1"]
    stim_names = ["vis", "aud"]

    for st, stim in enumerate(stims):

        stim_trials = trials[:].query("stim_name==@stim").index.values

        spikes = trial_da.sel(unit_id=sel_unit, trials=stim_trials).mean(dim=["trials"])

        ax[st].plot(spikes.time, spikes.values, label=stim)
        ax[st].axvline(0, color="k", linestyle="--", alpha=0.5)
        # ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[st].set_title(stim_names[st])
        ax[st].legend()
        ax[st].set_xlim([-0.2, 0.5])
        ax[st].set_xlabel("time (s)")

        if st > 1:
            ax[st].set_xlabel("time (s)")
        if st == 0 or st == 2:
            ax[st].set_ylabel("spikes/s")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_target_stim_PSTHs.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()


def plot_motor_response_by_unit_id(sel_unit, save_path=None, show_metric=None):
    unit_df = (
        ds.dataset(
            npc_lims.get_cache_path("units", session_id=sel_unit[:17], version="any")
        )
        .to_table(filter=(ds.field("unit_id") == sel_unit))
        .to_pandas()
    )
    session_id = (
        str(unit_df["subject_id"].values[0]) + "_" + str(unit_df["date"].values[0])
    )

    trials = pd.read_parquet(
        npc_lims.get_cache_path("trials", session_id, version="any")
    )

    ##plot rasters - vis stim, aud context
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharex=True)
    ax = ax.flatten()

    responses = [True, False]
    response_names = ["lick", "no lick"]
    stim = "vis1"

    for rr, _resp in enumerate(responses):

        if stim == "vis1":
            pass
        elif stim == "sound1":
            pass
        resp_trials = trials[:].query(
            "stim_name==@stim and context_name==@sel_context and is_response==@resp"
        )
        ax[rr].axvline(0, color="k", linestyle="-", alpha=0.5)
        ax[rr].set_title(response_names[rr])
        ax[rr].set_xlim([-0.2, 0.5])
        ax[rr].set_xlabel("time (s)")
        if rr == 0:
            ax[rr].set_ylabel("trials")

        for it, (_tt, trial) in enumerate(resp_trials.iterrows()):
            stim_start = trial["stim_start_time"]
            spikes = unit_df["spike_times"].iloc[0] - stim_start

            trial_spike_inds = (spikes >= -0.2) & (spikes <= 0.5)
            if sum(trial_spike_inds) == 0:
                continue
            else:
                spikes = spikes[trial_spike_inds]

            ax[rr].vlines(spikes, ymin=it, ymax=it + 1, linewidth=0.75, color="k")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, unit_df["unit_id"].values[0] + "_lick_rasters.png"),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()

    # PSTHs
    time_before = 0.2
    time_after = 0.5
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(
        unit_df, trials, time_before, time_after, binsize
    )

    ##plot PSTHs - target only
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), sharex=True, sharey=True)
    ax = ax.flatten()

    responses = [True, False]
    response_names = ["lick", "no lick"]
    # stim='vis1'

    for rr, _resp in enumerate(responses):

        if stim == "vis1":
            pass
        elif stim == "sound1":
            pass
        resp_trials = (
            trials[:]
            .query(
                "stim_name==@stim and context_name==@sel_context and is_response==@resp"
            )
            .index.values
        )

        spikes = trial_da.sel(unit_id=sel_unit, trials=resp_trials).mean(dim=["trials"])

        ax[rr].plot(spikes.time, spikes.values, label=stim)
        ax[rr].axvline(0, color="k", linestyle="--", alpha=0.5)
        # ax[st].axvline(0.5, color='k', linestyle='--',alpha=0.5)
        ax[rr].set_title(response_names[rr])
        ax[rr].legend()
        ax[rr].set_xlim([-0.2, 0.5])
        ax[rr].set_xlabel("time (s)")

        if rr > 1:
            ax[rr].set_xlabel("time (s)")
        if rr == 0:
            ax[rr].set_ylabel("spikes/s")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, unit_df["unit_id"].values[0] + "_lick_PSTHs.png"),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()


def plot_context_offset_by_unit_id(sel_unit, save_path=None, show_metric=None):
    unit_df = (
        ds.dataset(
            npc_lims.get_cache_path("units", session_id=sel_unit[:17], version="any")
        )
        .to_table(filter=(ds.field("unit_id") == sel_unit))
        .to_pandas()
    )
    session_id = (
        str(unit_df["subject_id"].values[0]) + "_" + str(unit_df["date"].values[0])
    )

    trials = pd.read_parquet(
        npc_lims.get_cache_path("trials", session_id, version="any")
    )

    ##plot rasters - target only
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
    ax = ax.flatten()

    stims = ["vis1", "sound1"]
    stim_names = ["vis", "aud"]

    for st, _stim in enumerate(stims):

        stim_trials = trials[:].query("stim_name==@stim").reset_index()

        colors = np.full(len(stim_trials), "k                 ")
        colors[stim_trials.query("is_vis_context").index] = "tab:green"
        colors[stim_trials.query("is_aud_context").index] = "tab:blue"

        # aud context trials
        stim_trials.query("is_aud_context").index.values

        # make patches during aud context trials
        for it, trial in stim_trials.query("is_aud_context").iterrows():
            stim_start = trial["stim_start_time"]
            ax[st].add_patch(
                patches.Rectangle(
                    (-0.2, it),
                    0.7,
                    0.5,
                    fill=True,
                    color="grey",
                    edgecolor=None,
                    alpha=0.1,
                )
            )

        ax[st].axvline(0, color="k", linestyle="-", alpha=0.5)
        ax[st].set_title(stim_names[st])
        ax[st].set_xlim([-0.2, 0.5])
        ax[st].set_xlabel("time (s)")
        if st == 0:
            ax[st].set_ylabel("trials")

        for it, (_tt, trial) in enumerate(stim_trials.iterrows()):
            stim_start = trial["stim_start_time"]
            spikes = unit_df["spike_times"].iloc[0] - stim_start

            trial_spike_inds = (spikes >= -0.2) & (spikes <= 0.5)
            if sum(trial_spike_inds) == 0:
                continue
            else:
                spikes = spikes[trial_spike_inds]

            # ax[st].vlines(spikes,ymin=it,ymax=it+1,linewidth=0.75,color='k')
            ax[st].vlines(
                spikes, ymin=it, ymax=it + 1, linewidth=0.75, color=colors[it]
            )

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_context_diff_rasters.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()

    time_before = 0.2
    time_after = 0.5
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(
        unit_df, trials, time_before, time_after, binsize
    )

    ##plot PSTHs - target only
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), sharex=True, sharey=True)
    ax = ax.flatten()

    stims = ["vis1", "sound1"]
    stim_names = ["vis", "aud"]

    for st, stim in enumerate(stims):

        stim_trials = trials[:].query("stim_name==@stim")

        vis_context_spikes = trial_da.sel(
            trials=stim_trials.query("is_vis_context").index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        aud_context_spikes = trial_da.sel(
            trials=stim_trials.query("is_aud_context").index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        ax[st].plot(
            vis_context_spikes.time,
            vis_context_spikes.values,
            label="vis context",
            color="tab:green",
        )
        ax[st].plot(
            aud_context_spikes.time,
            aud_context_spikes.values,
            label="aud context",
            color="tab:blue",
        )
        ax[st].axvline(0, color="k", linestyle="--", alpha=0.5)
        ax[st].axvline(0.5, color="k", linestyle="--", alpha=0.5)
        ax[st].set_title(stim)
        if st == 1:
            ax[st].legend()
        ax[st].set_xlim([-0.2, 0.5])

        if st > 1:
            ax[st].set_xlabel("time (s)")
        if st == 0 or st == 2:
            ax[st].set_ylabel("spikes/s")

    if show_metric is not None:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
            + "; "
            + show_metric
        )
    else:
        fig.suptitle(
            "unit "
            + unit_df["unit_id"].values[0]
            + "; "
            + unit_df["structure"].values[0]
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_context_diff_PSTHs.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()


def plot_unit_response_by_task_performance(sel_unit, save_path=None):
    unit_df = (
        ds.dataset(npc_lims.get_cache_path("units"))
        .to_table(filter=(ds.field("unit_id") == sel_unit))
        .to_pandas()
    )
    session_id = (
        str(unit_df["subject_id"].values[0]) + "_" + str(unit_df["date"].values[0])
    )

    trials = pd.read_parquet(
        npc_lims.get_cache_path("trials", session_id, version="any")
    )

    time_before = 0.5
    time_after = 1
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(
        unit_df, trials, time_before, time_after, binsize
    )

    ## plot PSTH with context differences -- subplot for each performance
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))

    contexts = ["vis", "aud"]
    cols = ["tab:green", "tab:blue", "tab:red"]
    # target - hits
    for c, context in enumerate(contexts):
        stim_trials = trials[(trials.is_hit) & (trials["is_" + context + "_context"])]
        context_spikes = trial_da.sel(
            trials=stim_trials.index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        rate = np.round(
            len(trials[(trials.is_hit) & (trials["is_" + context + "_context"])])
            / len(trials[(trials.is_go) & (trials["is_" + context + "_context"])]),
            2,
        )
        ax[0, c].plot(
            context_spikes.time,
            context_spikes.values,
            label="hits, rate:" + str(rate),
            color=cols[0],
        )
        ax[0, c].set_title(context + " context")

    conditions_set = ["correct_reject", "false_alarm"]
    for c1, context in enumerate(contexts):
        for c2, condition in enumerate(conditions_set):
            # target
            stim_trials = trials[
                (trials["is_" + condition])
                & (trials["is_" + context + "_context"])
                & (trials.is_target)
            ]
            target_spikes = trial_da.sel(
                trials=stim_trials.index,
                unit_id=sel_unit,
            ).mean(dim=["trials"])

            rate = np.round(
                len(
                    trials[
                        (trials["is_" + condition])
                        & (trials["is_" + context + "_context"])
                        & (trials.is_target)
                    ]
                )
                / len(
                    trials[
                        (trials.is_nogo)
                        & (trials["is_" + context + "_context"])
                        & (trials.is_target)
                    ]
                ),
                2,
            )
            ax[0, c1].plot(
                target_spikes.time,
                target_spikes.values,
                label=condition + ", rate: " + str(rate),
                color=cols[c2 + 1],
            )

            # non-target
            stim_trials = trials[
                (trials["is_" + condition])
                & (trials["is_" + context + "_context"])
                & (trials.is_nontarget)
            ]
            target_spikes = trial_da.sel(
                trials=stim_trials.index,
                unit_id=sel_unit,
            ).mean(dim=["trials"])

            rate = np.round(
                len(
                    trials[
                        (trials["is_" + condition])
                        & (trials["is_" + context + "_context"])
                        & (trials.is_nontarget)
                    ]
                )
                / len(
                    trials[
                        (trials.is_nogo)
                        & (trials["is_" + context + "_context"])
                        & (trials.is_nontarget)
                    ]
                ),
                2,
            )
            ax[1, c1].plot(
                target_spikes.time,
                target_spikes.values,
                label=condition + ", rate: " + str(rate),
                color=cols[c2 + 1],
            )

    ax = ax.flatten()
    for c in range(4):
        ax[c].axvline(0, color="k", linestyle="--", alpha=0.5)
        ax[c].axvline(0.5, color="k", linestyle="--", alpha=0.5)
        ax[c].legend()
        ax[c].set_xlim([-0.5, 1.0])

        if c > 1:
            ax[c].set_xlabel("time (s)")
        if c == 0:
            ax[c].set_ylabel("Target - spikes/s")
        if c == 2:
            ax[c].set_ylabel("Non-target - spikes/s")

    fig.suptitle(
        "unit " + unit_df["unit_id"].values[0] + "; " + unit_df["structure"].values[0]
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, unit_df["unit_id"].values[0] + "_performance.png"),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()


def plot_unit_response_by_task_performance_stim_aligned(sel_unit, save_path=None):
    unit_df = (
        ds.dataset(npc_lims.get_cache_path("units"))
        .to_table(filter=(ds.field("unit_id") == sel_unit))
        .to_pandas()
    )
    session_id = (
        str(unit_df["subject_id"].values[0]) + "_" + str(unit_df["date"].values[0])
    )

    trials = pd.read_parquet(
        npc_lims.get_cache_path("trials", session_id, version="any")
    )

    time_before = 0.5
    time_after = 1
    binsize = 0.025
    trial_da = spike_utils.make_neuron_time_trials_tensor(
        unit_df, trials, time_before, time_after, binsize
    )

    ## plot PSTH with context differences -- subplot for each performance
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(4, 6))

    stims = ["vis", "aud"]
    cols = ["tab:green", "tab:blue", "tab:red"]
    # target - hits
    for s, stim in enumerate(stims):
        stim_trials = trials[(trials.is_hit) & (trials["is_" + stim + "_target"])]
        stim_spikes = trial_da.sel(
            trials=stim_trials.index,
            unit_id=sel_unit,
        ).mean(dim=["trials"])

        rate = np.round(
            len(trials[(trials.is_hit) & (trials["is_" + stim + "_target"])])
            / len(trials[(trials.is_go) & (trials["is_" + stim + "_target"])]),
            2,
        )
        ax[s].plot(
            stim_spikes.time,
            stim_spikes.values,
            label="hits, rate:" + str(rate),
            color=cols[0],
        )
        ax[s].set_title(stim + " stimulus")

    conditions_set = ["correct_reject", "false_alarm"]
    for s, stim in enumerate(stims):
        for c, condition in enumerate(conditions_set):
            # target
            stim_trials = trials[
                (trials["is_" + condition]) & (trials["is_" + stim + "_target"])
            ]
            target_spikes = trial_da.sel(
                trials=stim_trials.index,
                unit_id=sel_unit,
            ).mean(dim=["trials"])

            rate = np.round(
                len(
                    trials[
                        (trials["is_" + condition]) & (trials["is_" + stim + "_target"])
                    ]
                )
                / len(trials[(trials.is_nogo) & (trials["is_" + stim + "_target"])]),
                2,
            )
            ax[s].plot(
                target_spikes.time,
                target_spikes.values,
                label=condition + ", rate: " + str(rate),
                color=cols[c + 1],
            )

    ax = ax.flatten()
    for s in range(2):
        ax[s].axvline(0, color="k", linestyle="--", alpha=0.5)
        ax[s].axvline(0.5, color="k", linestyle="--", alpha=0.5)
        ax[s].legend()
        ax[s].set_xlim([-0.5, 1.0])
        ax[s].set_ylabel("Target - spikes/s")
    ax[1].set_xlabel("time (s)")
    fig.suptitle(
        "unit " + unit_df["unit_id"].values[0] + "; " + unit_df["structure"].values[0]
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            os.path.join(
                save_path, unit_df["unit_id"].values[0] + "_performance_stim_target.png"
            ),
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )
        plt.close()


def plot_stimulus_modulation_pie_chart(adj_pvals, sel_project, savepath=None):
    # stimulus modulation across all units
    # each stim only
    vis1_stim_resp = adj_pvals.query(
        "vis1<0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05"
    )
    vis2_stim_resp = adj_pvals.query(
        "vis2<0.05 and vis1>=0.05 and sound1>=0.05 and sound2>=0.05"
    )
    sound1_stim_resp = adj_pvals.query(
        "sound1<0.05 and sound2>=0.05 and vis1>=0.05 and vis2>=0.05"
    )
    sound2_stim_resp = adj_pvals.query(
        "sound2<0.05 and sound1>=0.05 and vis1>=0.05 and vis2>=0.05"
    )

    # both vis
    both_vis_stim_resp = adj_pvals.query(
        "vis1<0.05 and vis2<0.05 and sound1>=0.05 and sound2>=0.05"
    )
    # both aud
    both_sound_stim_resp = adj_pvals.query(
        "sound1<0.05 and sound2<0.05 and vis1>=0.05 and vis2>=0.05"
    )

    # at least one vis and one aud
    mixed_stim_resp = adj_pvals.query(
        "((vis1<0.05 or vis2<0.05) and (sound1<0.05 and sound2<0.05))"
    )

    # any stim
    # any_stim_resp=adj_pvals.query('vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05')
    adj_pvals.query("any_stim<0.05")
    adj_pvals.query("any_stim<0.05 and context<0.05")

    # catch
    catch_stim_resp = adj_pvals.query("catch<0.05")

    # none
    no_stim_resp = adj_pvals.query(
        "vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and catch>=0.05"
    )

    # stimulus responses
    labels = [
        "vis1 only",
        "vis2 only",
        "both vis",
        "sound1 only",
        "sound2 only",
        "both sound",
        "mixed",
        "none",
        "catch",
    ]
    sizes = [
        len(vis1_stim_resp),
        len(vis2_stim_resp),
        len(both_vis_stim_resp),
        len(sound1_stim_resp),
        len(sound2_stim_resp),
        len(both_sound_stim_resp),
        len(mixed_stim_resp),
        len(no_stim_resp),
        len(catch_stim_resp),
    ]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("n = " + str(len(adj_pvals)) + " units")
    fig.suptitle("stimulus responsiveness")
    fig.tight_layout()

    if savepath is not None:
        if "Templeton" in sel_project:
            temp_savepath = os.path.join(
                savepath, "stimulus_responsiveness_Templeton.png"
            )
        else:
            temp_savepath = os.path.join(savepath, "stimulus_responsiveness_DR.png")

        fig.savefig(
            temp_savepath,
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )


def plot_context_stim_lick_modulation_pie_chart(adj_pvals, sel_project, savepath=None):

    # lick modulation only
    lick_resp = adj_pvals.query(
        "lick<0.05 and context>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05"
    )

    # lick and context
    lick_and_context_resp = adj_pvals.query(
        "context<0.05 and lick<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05"
    )

    # lick and stimulus
    lick_and_stim_resp = adj_pvals.query(
        "lick<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and context>=0.05"
    )

    # all three
    all_resp = adj_pvals.query(
        "context<0.05 and lick<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05)"
    )

    # stimulus modulation only
    only_stim_resp = adj_pvals.query(
        "(vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and context>=0.05 and lick>=0.05"
    )

    # context modulation only
    context_resp = adj_pvals.query(
        "context<0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05"
    )

    # stim and context modulation
    stim_and_context_resp = adj_pvals.query(
        "context<0.05 and (vis1<0.05 or vis2<0.05 or sound1<0.05 or sound2<0.05) and lick>=0.05"
    )

    neither_stim_nor_context_resp = adj_pvals.query(
        "context>=0.05 and vis1>=0.05 and vis2>=0.05 and sound1>=0.05 and sound2>=0.05 and lick>=0.05"
    )

    labels = [
        "stimulus only",
        "stimulus and context",
        "context only",
        "context and lick",
        "lick only",
        "lick & stimulus & context",
        "lick and stimulus",
        "none",
    ]
    sizes = [
        len(only_stim_resp),
        len(stim_and_context_resp),
        len(context_resp),
        len(lick_and_context_resp),
        len(lick_resp),
        len(all_resp),
        len(lick_and_stim_resp),
        len(neither_stim_nor_context_resp),
    ]

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=[
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "grey",
        ],
    )
    ax.set_title("n = " + str(len(adj_pvals)) + " units")
    fig.suptitle("context, lick, and stim modulation")
    fig.tight_layout()

    if savepath is not None:
        if "Templeton" in sel_project:
            temp_savepath = os.path.join(savepath, "context_stim_lick_Templeton.png")
        else:
            temp_savepath = os.path.join(savepath, "context_stim_lick_DR.png")

        fig.savefig(
            temp_savepath,
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )


def plot_context_mod_stim_resp_pie_chart(adj_pvals, sel_project, savepath=None):

    # stimulus context modulation
    vis1_context_stim_mod = adj_pvals.query(
        "vis1_context<0.05 and vis2_context>=0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05"
    )
    vis2_context_stim_mod = adj_pvals.query(
        "vis2_context<0.05 and vis1_context>=0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05"
    )
    sound1_context_stim_mod = adj_pvals.query(
        "sound1_context<0.05 and sound2_context>=0.05 and vis1_context>=0.05 and vis2_context>=0.05 and any_stim<0.05"
    )
    sound2_context_stim_mod = adj_pvals.query(
        "sound2_context<0.05 and sound1_context>=0.05 and vis1_context>=0.05 and vis2_context>=0.05 and any_stim<0.05"
    )

    both_vis_context_stim_mod = adj_pvals.query(
        "vis1_context<0.05 and vis2_context<0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05"
    )
    both_aud_context_stim_mod = adj_pvals.query(
        "sound1_context<0.05 and sound2_context<0.05 and vis1_context>=0.05 and vis2_context>=0.05 and any_stim<0.05"
    )
    multi_modal_context_stim_mod = adj_pvals.query(
        "((vis1_context<0.05 or vis2_context<0.05) and (sound1_context<0.05 or sound2_context<0.05)) and any_stim<0.05"
    )

    no_context_stim_mod = adj_pvals.query(
        "vis1_context>=0.05 and vis2_context>=0.05 and sound1_context>=0.05 and sound2_context>=0.05 and any_stim<0.05"
    )

    # evoked stimulus context modulation
    vis1_context_evoked_stim_mod = adj_pvals.query(
        "vis1_context_evoked<0.05 and vis2_context_evoked>=0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05"
    )
    vis2_context_evoked_stim_mod = adj_pvals.query(
        "vis2_context_evoked<0.05 and vis1_context_evoked>=0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05"
    )
    sound1_context_evoked_stim_mod = adj_pvals.query(
        "sound1_context_evoked<0.05 and sound2_context_evoked>=0.05 and vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and any_stim<0.05"
    )
    sound2_context_evoked_stim_mod = adj_pvals.query(
        "sound2_context_evoked<0.05 and sound1_context_evoked>=0.05 and vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and any_stim<0.05"
    )

    both_vis_context_evoked_stim_mod = adj_pvals.query(
        "vis1_context_evoked<0.05 and vis2_context_evoked<0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05"
    )
    both_aud_context_evoked_stim_mod = adj_pvals.query(
        "sound1_context_evoked<0.05 and sound2_context_evoked<0.05 and vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and any_stim<0.05"
    )
    multi_modal_context_evoked_stim_mod = adj_pvals.query(
        "((vis1_context_evoked<0.05 or vis2_context_evoked<0.05) and (sound1_context_evoked<0.05 or sound2_context_evoked<0.05)) and any_stim<0.05"
    )

    no_context_evoked_stim_mod = adj_pvals.query(
        "vis1_context_evoked>=0.05 and vis2_context_evoked>=0.05 and sound1_context_evoked>=0.05 and sound2_context_evoked>=0.05 and any_stim<0.05"
    )

    labels = [
        "vis1 only",
        "vis2 only",
        "both vis",
        "sound1 only",
        "sound2 only",
        "both sound",
        "mixed",
        "none",
    ]
    sizes = [
        len(vis1_context_stim_mod),
        len(vis2_context_stim_mod),
        len(both_vis_context_stim_mod),
        len(sound1_context_stim_mod),
        len(sound2_context_stim_mod),
        len(both_aud_context_stim_mod),
        len(multi_modal_context_stim_mod),
        len(no_context_stim_mod),
    ]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("n = " + str(len(adj_pvals)) + " units")
    # ax.set_title('n = '+str(len(stim_and_context))+' units')

    fig.suptitle("context modulation of stimulus response")
    fig.tight_layout()

    if savepath is not None:
        if "Templeton" in sel_project:
            temp_savepath = os.path.join(
                savepath, "context_mod_stim_resp_Templeton.png"
            )
        else:
            temp_savepath = os.path.join(savepath, "context_mod_stim_resp_DR.png")

        fig.savefig(
            temp_savepath,
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )

    # evoked
    labels = [
        "vis1 only",
        "vis2 only",
        "both vis",
        "sound1 only",
        "sound2 only",
        "both sound",
        "mixed",
        "none",
    ]
    sizes = [
        len(vis1_context_evoked_stim_mod),
        len(vis2_context_evoked_stim_mod),
        len(both_vis_context_evoked_stim_mod),
        len(sound1_context_evoked_stim_mod),
        len(sound2_context_evoked_stim_mod),
        len(both_aud_context_evoked_stim_mod),
        len(multi_modal_context_evoked_stim_mod),
        len(no_context_evoked_stim_mod),
    ]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("n = " + str(len(adj_pvals)) + " units")

    fig.suptitle("context modulation of evoked stimulus response")
    fig.tight_layout()

    if savepath is not None:
        if "Templeton" in sel_project:
            temp_savepath = os.path.join(
                savepath, "evoked_context_mod_stim_resp_Templeton.png"
            )
        else:
            temp_savepath = os.path.join(
                savepath, "evoked_context_mod_stim_resp_DR.png"
            )

        fig.savefig(
            temp_savepath,
            dpi=300,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=None,
        )


def plot_single_session_decoding_results(path):

    use_half_shifts = False

    decoder_results = pickle.load(open(path, "rb"))
    session_id = list(decoder_results.keys())[0]

    shifts = decoder_results[session_id]["shifts"]
    areas = decoder_results[session_id]["areas"]
    n_repeats = 25

    half_neg_shift = np.round(shifts.min() / 2)
    half_pos_shift = np.round(shifts.max() / 2)

    half_neg_shift_ind = np.where(shifts == half_neg_shift)[0][0]
    half_pos_shift_ind = np.where(shifts == half_pos_shift)[0][0]
    half_shift_inds = np.arange(half_neg_shift_ind, half_pos_shift_ind + 1)

    if use_half_shifts is False:
        half_shift_inds = np.arange(len(shifts))

    bal_acc = {}
    for aa in areas:
        if aa in decoder_results[session_id]["results"]:
            bal_acc[aa] = []
            for rr in range(n_repeats):
                temp_bal_acc = []
                for sh in half_shift_inds:
                    if sh in list(
                        decoder_results[session_id]["results"][aa]["shift"][rr].keys()
                    ):
                        temp_bal_acc.append(
                            decoder_results[session_id]["results"][aa]["shift"][rr][sh][
                                "balanced_accuracy_test"
                            ]
                        )
                if len(temp_bal_acc) > 0:
                    bal_acc[aa].append(np.array(temp_bal_acc))
            bal_acc[aa] = np.vstack(bal_acc[aa])

    for aa in areas:
        if aa in decoder_results[session_id]["results"]:
            mean_acc = np.nanmean(bal_acc[aa], axis=0)

            true_acc = mean_acc[shifts[half_shift_inds] == 1]
            pval = np.round(np.nanmean(mean_acc >= true_acc), decimals=4)

            fig, ax = plt.subplots(1, 1)
            ax.axhline(true_acc, color="k", linestyle="--", alpha=0.5)
            ax.axvline(1, color="k", linestyle="--", alpha=0.5)
            ax.plot(shifts[half_shift_inds], bal_acc[aa].T, alpha=0.5, color="gray")
            ax.plot(shifts[half_shift_inds], mean_acc, color="k", linewidth=2)
            ax.set_xlabel("trial shift")
            ax.set_ylabel("balanced accuracy")
            ax.set_title(str(aa) + " p=" + str(pval))


def _aggregate_top_layer(
    regions: npt.ArrayLike,
    values: npt.ArrayLike,
    agg_func_name: str,
) -> tuple[npt.NDArray[str], npt.NDArray[float]]:
    """
    For the 'top' view of cortex, layer 1 is in view, but typically not informative.
    Find all the layers for an area, aggregate and update layer 1's value

    >>> regions, values = _aggregate_top_layer(np.array(["VISp2/3", "VISp4"]), np.array([[2, np.nan], [np.nan, 3]]), "max")
    >>> list(regions)
    [np.str_('VISp1')]
    >>> list(values)
    [array([2., 3.])]
    """
    top_values = []
    top_regions = []

    def get_agg_layer_df(regions, values):
        return (
            # create a mapping of layer acronym to corresponding top-level layer and an aggregate value across all layers
            ccf_utils.get_ccf_structure_tree_df()
            # get values for any matching areas, but operate on all areas until very end:
            .join(
                pl.DataFrame({"acronym": regions, "value": values}),
                on="acronym",
                how="left",
            )  # preserve all rows
            # exclude nan values from calculations:
            .with_columns(
                pl.col("value").fill_nan(None),
            )
            .filter(
                pl.col("name").str.to_lowercase().str.contains("layer"),
            )
            .group_by("parent_structure_id")
            .agg("acronym", "value")
            .filter(
                pl.col("acronym").list.join("").str.contains("1"),
                pl.col("acronym").list.len()
                > 1,  # if only one acronym in group, it has a 1 in the name but no layers (e.g. CA1)
            )
            .with_columns(
                pl.col("acronym").list.first().alias("top_layer"),
                # if all values are null, agg = 0.0, which is incorrect
                # - we also don't want to drop nulls completely, so apply selectively:
                pl.when(pl.col("value").list.drop_nulls().list.len() != 0)
                .then(
                    pl.col("value")
                    .list.drop_nulls()
                    .list.eval(getattr(pl.element(), agg_func_name)())
                    .list.get(0)
                )
                .otherwise(pl.lit(np.nan)),
            )
            .explode("acronym")
            # filter rows with matches in actual data
            .join(
                pl.DataFrame({"acronym": regions, "value": values}),
                on="acronym",
                how="semi",
            )
        )

    df_left = get_agg_layer_df(regions, values[:, 0])
    df_right = get_agg_layer_df(regions, values[:, 1])
    for idx, r in enumerate(regions):
        if r not in df_left["acronym"] and r not in df_right["acronym"]:
            top_regions.append(r)
            top_values.append(values[idx, :])
        else:
            top_layer = df_left.filter(pl.col("acronym") == r)["top_layer"][
                0
            ]  # doesn't matter which df we use here
            if top_layer in top_regions:
                continue  # already added
            lr_values = [
                df.filter(pl.col("acronym") == r)["value"][0]
                for df in (df_left, df_right)
            ]
            top_regions.append(top_layer)
            top_values.append(lr_values)
    assert len(top_values) == len(top_regions)
    return np.array(top_regions), np.array(top_values)


def get_heatmap_gdf(
    regions: Iterable[str] | npt.ArrayLike,
    values: Iterable[float] | npt.ArrayLike,
    projection: str | Literal["sagittal", "coronal", "horizontal", "top"] = "top",
    position: float | None = None,
    remove_redundant_parents: bool = True,
    combine_child_patches: bool = True,
    top_layer_agg_func: str | None = None,
    horizontal_upright: bool = False,
) -> gpd.GeoDataFrame:
    t0 = time.time()
    # clean up inputs
    if position is None and projection != "top":
        raise ValueError("position must be specified for non-top projections")
    if projection == "top" and position is not None:
        logger.warning("position is ignored for top view")
    # check for duplicate regions
    if len(set(regions)) != len(regions):
        raise ValueError(
            f"Provide only one value per area acronym: {Counter(regions).most_common(3)=}"
        )
    regions = np.array(regions)
    values = np.array(values)
    if values.ndim == 1:
        values = np.array([values, np.full(len(values), np.nan)]).T
    if values.ndim > 1 and values.shape[0] == 2 and values.shape[1] != 2:
        values = values.T
    if values.shape[0] != regions.shape[0]:
        raise ValueError(f"{values.shape[0]=} does not match {regions.shape[0]=}")

    if top_layer_agg_func is not None:
        if not isinstance(top_layer_agg_func, str):
            raise ValueError(
                f"Layer aggregation function should be specified as a string, e.g. 'max', 'mean', not {top_layer_agg_func!r}"
            )
        regions, values = _aggregate_top_layer(regions, values, top_layer_agg_func)
    assert values.shape[1] == 2
    assert values.shape[0] == regions.shape[0]

    if not regions.size:
        regions = ccf_utils.get_ccf_structure_tree_df()["acronym"].to_numpy()
        values = np.full((len(regions), 2), np.nan)
    user_df = pl.DataFrame({"acronym": regions, "value": values}).join(
        ccf_utils.get_ccf_structure_tree_df(),
        on="acronym",
        how="left",
    )

    missing_ccf = set(regions) - set(user_df["acronym"])
    if missing_ccf:
        logger.warning(
            f"{len(missing_ccf)} acronyms specified in 'regions' have no match in CCF tree: {missing_ccf}"
        )

    expr = pl.col("id").is_in(user_df["parent_ids"].explode())
    redundant_parents = user_df.filter(expr)
    if len(redundant_parents):
        if remove_redundant_parents:
            logger.debug(
                f"Removing {len(redundant_parents)} regions as they are parents of other regions ({remove_redundant_parents=!r}): {redundant_parents['acronym'].to_numpy()}"
            )
            user_df = user_df.filter(~expr)
        else:
            raise ValueError(
                f"Found {len(redundant_parents)} regions that are parents of other regions: {redundant_parents['acronym'].to_numpy()} (try setting `remove_redundant_parents=True`)"
            )

    if not combine_child_patches:
        # only the deepest children in the tree are labelled in the volume:
        # replace any parents in user-specified 'regions' with their children
        #  - this is a requirement for plotting unless child patch polygons are combined below
        #! note: this will apply to both left and right hemispheres
        logger.info(
            f"Converting each of {len(regions)} regions to deepest children in CCF tree for plotting purposes ({combine_child_patches=!r})"
        )
        values_ = []
        child_ids = []
        for row in user_df.iter_rows(named=True):
            row_child_ids = row["deepest_child_ids"]
            if set(row_child_ids) & set(child_ids):
                raise ValueError(
                    f"In trying to add children for {row['acronym']!r}, {set(row_child_ids) & set(child_ids)} already have values: not sure how to continue"
                )
            child_ids.extend(row_child_ids)
            values_.extend([row["value"]] * len(row_child_ids))
        user_df = pl.DataFrame({"id": child_ids, "value": values_}).join(
            other=ccf_utils.get_ccf_structure_tree_df(),
            on="id",
            how="left",
        )

    # get slice/projection img:
    vol = ccf_utils.get_ccf_volume(left_hemisphere=True, right_hemisphere=True)
    if projection == "top":
        img = ccf_utils.project_first_nonzero_labels(vol)
        img[np.isnan(img)] = 0
    else:
        assert position is not None
        p = ccf_utils.ccf_to_volume_index(position)
        if projection == "sagittal":
            img = vol[p, :, :]
        elif projection == "horizontal":
            img = vol[:, p, :]
        elif projection == "coronal":
            img = vol[:, :, p].T

    mirror_lr = True
    dtype = np.int32
    volume_ml_midline = round(vol.shape[0] / 2)

    def _split_img_on_midline():
        img_l = img.copy().astype(dtype)
        img_r = img.copy().astype(dtype)
        assert len(np.unique(img_l)) == len(np.unique(img))
        if projection in ("top", "horizontal"):
            img_l[volume_ml_midline:, :] = 0
            if mirror_lr:
                img_r = img_l[::-1, :]
            else:
                img_r[:volume_ml_midline, :] = 0
        elif projection == "sagittal":
            assert position is not None
            if ccf_utils.ccf_to_volume_index(position) <= volume_ml_midline:
                img_r = np.zeros_like(img).astype(dtype)
            else:
                img_l = np.zeros_like(img).astype(dtype)
        elif projection == "coronal":
            img_l[:, volume_ml_midline:] = 0
            if mirror_lr:
                img_r = img_l[:, ::-1]
            else:
                img_r[:, :volume_ml_midline] = 0
        return img_l, img_r

    img_left, img_right = _split_img_on_midline()
    # get shapely polygons from connected labelled regions:
    transform = rasterio.Affine(
        ccf_utils.RESOLUTION_UM, 0, 0, 0, ccf_utils.RESOLUTION_UM, 0
    )

    for img in (img_left, img_right):
        is_left_img = img is img_left
        ids = []
        geometry = []

        if projection in ("top", "horizontal") and horizontal_upright:
            img = img.T

        # find connected regions in the image:
        assert img.dtype in (np.int32, np.float32)
        for polygon, label in rasterio.features.shapes(
            img, connectivity=4, transform=transform
        ):
            if label == 0 or np.isnan(label):
                continue
            g = shapely.geometry.shape(polygon)
            ids.append(int(label))
            geometry.append(g)

        # each area ID may have multiple discontiguous patches in labeled array: combine their polygons to get one polygon per area
        ids_ = []
        geometry_ = []
        for id_ in set(ids):
            geometry_.append(
                shapely.union_all([g for g, i in zip(geometry, ids) if i == id_])
            )
            ids_.append(id_)

        if combine_child_patches:
            # for each user-specified region acronym, group all its children into a single polygon
            combined_geometry = []
            combined_ids = []
            for row in user_df.iter_rows(named=True):
                row["acronym"]
                v = row["value"]
                value = v[0] if is_left_img else v[1]
                if np.isnan(value):
                    continue
                deepest_child_ids = row["deepest_child_ids"]
                if len(deepest_child_ids) <= 1 or not set(deepest_child_ids) & set(
                    ids_
                ):
                    continue
                combined_geometry.append(
                    shapely.union_all(
                        [g for g, i in zip(geometry_, ids_) if i in deepest_child_ids]
                    )
                )
                combined_ids.append(row["id"])
                for i in deepest_child_ids:
                    if i in ids_:
                        idx = ids_.index(i)
                        ids_.pop(idx)
                        geometry_.pop(idx)
            ids_ += combined_ids
            geometry_ += combined_geometry
        if is_left_img:
            ids_left, geometry_left = ids_, geometry_
        else:
            ids_right, geometry_right = ids_, geometry_

    gdf_left = gpd.GeoDataFrame({"id": ids_left, "geometry": geometry_left})
    gdf_right = gpd.GeoDataFrame({"id": ids_right, "geometry": geometry_right})
    for idx, gdf in enumerate((gdf_left, gdf_right)):
        gdf["position"] = position
        gdf["projection"] = projection
        gdf["value"] = np.nan
        gdf["hemisphere"] = "left" if gdf is gdf_left else "right"
        for row in user_df.iter_rows(named=True):
            gdf.loc[gdf["id"] == row["id"], "value"] = row["value"][idx]
    gdf = pd.concat((gdf_left, gdf_right)).merge(
        right=ccf_utils.get_ccf_structure_tree_df().to_pandas(),
        left_on="id",
        right_on="id",
        how="inner",  # keep rasterized regions that have ids in ccf structure tree
    )
    logger.info(
        f"Created GeoDataFrame with {len(gdf)} polygons in {time.time() - t0:.2f}s"
    )
    return gdf


def plot_brain_heatmap(
    regions: Iterable[str] | npt.ArrayLike,
    values: Iterable[float] | npt.ArrayLike,
    sagittal_planes: float | Iterable[float] | None = None,
    top_layer_agg_func: str | None = None,
    cmap: str = "viridis",
    clevels: tuple[float, float] | None = None,
    remove_redundant_parents: bool = True,
    combine_child_patches: bool = True,
    horizontal_upright: bool = False,
    labels: bool = False,
    labels_on_areas: bool = False,
    interactive: bool = False,
    patch_params: Mapping[str, Any] = {},
    missing_params: Mapping[str, Any] = {},
    plane_line_params: Mapping[str, Any] = {},
    annotation_params: Mapping[str, Any] = {},
) -> tuple[matplotlib.figure.Figure, tuple[pd.DataFrame]]:
    fig = plt.figure()
    gdfs = []
    if sagittal_planes is None:
        sagittal_planes = []
    elif not isinstance(sagittal_planes, Iterable):
        sagittal_planes = (sagittal_planes,)
    else:
        sagittal_planes = tuple(sagittal_planes)  # type: ignore
    sagittal_planes = [i + ccf_utils.get_midline_ccf_ml()  for i in sagittal_planes]
    if clevels is not None:
        clevels = tuple(clevels)  # type: ignore
        if len(clevels) != 2:
            raise ValueError("clevels must be a sequence of length 2")
    else:
        clevels = (np.nanmin(np.array(values)), np.nanmax(np.array(values)))
    norm = matplotlib.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
    axes = []

    vol = ccf_utils.get_ccf_volume(True, True)
    max_ap = vol.shape[2] * ccf_utils.RESOLUTION_UM
    max_dv = vol.shape[1] * ccf_utils.RESOLUTION_UM
    max_ml = vol.shape[0] * ccf_utils.RESOLUTION_UM
    height_top = max_ap if horizontal_upright else max_ml
    height_sagittal = max_dv
    gs = matplotlib.gridspec.GridSpec(
        len(sagittal_planes) + 2,
        1,
        figure=fig,
        height_ratios=[height_top / height_sagittal]
        + [1] * len(sagittal_planes)
        + [0.1],
    )
    axes.append(ax_top := fig.add_subplot(gs[0, 0]))
    gdf = get_heatmap_gdf(
        regions=regions,
        values=values,
        projection="top",
        horizontal_upright=horizontal_upright,
        remove_redundant_parents=remove_redundant_parents,
        combine_child_patches=combine_child_patches,
        top_layer_agg_func=top_layer_agg_func,
    )
    gdfs.append(gdf)
    missing_kwds = (
        {"color": "lightgrey"}
        | {k: v for k, v in patch_params.items() if k in ("edgecolor", "linewidth")}
        | missing_params
    )
    patch_kwds = {"edgecolor": "darkgrey", "linewidth": 0.1} | patch_params
    gdf.plot(
        column="value",
        cmap=cmap,
        missing_kwds=missing_kwds,
        ax=ax_top,
        norm=norm,
        **patch_kwds,
    )

    if labels:
        if labels_on_areas:
            # add a label for each area in the top view with a value (in right hemisphere only)
            for _idx, area in enumerate(
                gdf.dropna(subset=["value"])["acronym"].unique()
            ):
                rows = gdf[(gdf["acronym"] == area)]
                rows_with_values = rows.dropna(subset=["value"])
                if len(rows_with_values) == 0:
                    continue
                if len(rows_with_values) == 2:
                    # remove left row if right has value
                    rows_with_values = rows_with_values[
                        rows_with_values["hemisphere"] == "right"
                    ]
                assert len(rows_with_values) == 1
                row = rows.iloc[0]
                center_x, center_y = row.geometry.centroid.x, row.geometry.centroid.y
                if row["hemisphere"] == "left":
                    center_y = (
                        2 * (max_ap if horizontal_upright else max_ml) / 2 - center_y
                    )
                if labels_on_areas:
                    ax_top.text(
                        center_x,
                        center_y,
                        row["acronym"],
                        **{"fontsize": 1.5, "ha": "center", "va": "center"}
                        | annotation_params,
                    )
        else:
            # make annotations with lines pointing to the center of the area, spaced evenly around
            # the top-half of axes in an arc
            ap_center_of_mass = 0.5 * max_ap  # shift the center slightly posterior
            if horizontal_upright:
                brain_center_angle = 0.0
                brain_center_x, brain_center_y = max_ml / 2, ap_center_of_mass
            else:
                brain_center_angle = 0
                brain_center_x, brain_center_y = ap_center_of_mass, max_ml / 2
            arc_radius = 0.55 * max_ap
            angular_extent = np.pi
            label_gdf = gdf
            angular_spacing = angular_extent / len(
                label_gdf.dropna(subset=["value"])["acronym"].unique()
            )
            # add columns for distance and angle of centroid from brain center
            label_gdf["distance_from_center"] = np.sqrt(
                (label_gdf.geometry.centroid.x - brain_center_x) ** 2
                + (label_gdf.geometry.centroid.y - brain_center_y) ** 2
            )
            label_gdf["angle_from_horizontal"] = np.arctan2(
                label_gdf.geometry.centroid.y - brain_center_y,
                label_gdf.geometry.centroid.x - brain_center_x,
            )
            label_gdf = label_gdf.sort_values("angle_from_horizontal").dropna(
                subset=["value"]
            )
            for idx, area in enumerate(label_gdf["acronym"].unique()):
                rows = label_gdf[(label_gdf["acronym"] == area)]
                rows_with_values = rows.dropna(subset=["value"])
                if len(rows_with_values) == 0:
                    continue
                if len(rows_with_values) == 2:
                    # remove left row if right has value
                    rows_with_values = rows_with_values[
                        rows_with_values["hemisphere"] == "right"
                    ]
                assert len(rows_with_values) == 1
                row = rows_with_values.iloc[0]
                if set(row["parent_ids"]) & set(label_gdf["id"]):
                    parent_ids = row["parent_ids"]
                    # skip if any parents are due to be plot
                    if (
                        not label_gdf[label_gdf["id"].isin(parent_ids)]
                        .dropna(subset=["value"])
                        .empty
                    ):
                        continue
                center_x, center_y = row.geometry.centroid.x, row.geometry.centroid.y
                annotation_angle = brain_center_angle - np.pi - idx * angular_spacing
                if row["hemisphere"] == "left":
                    center_y = 2 * brain_center_y - center_y
                    annotation_angle = 2 * np.pi - annotation_angle
                length = arc_radius - row["distance_from_center"]
                # jitter length
                length += np.random.rand() * 0.3 * length
                x = center_x + length * np.cos(annotation_angle)
                y = center_y - length * np.sin(annotation_angle)
                ax_top.annotate(
                    row["acronym"],
                    xy=(center_x, center_y),
                    xytext=(x, y),
                    **{
                        "arrowprops": {"lw": 0.1, "arrowstyle": "-", "color": "black"},
                        "fontsize": 2,
                        "font": "arial",
                    }
                    | annotation_params,
                )
    for i, coord in enumerate(sorted(sagittal_planes, reverse=True)):
        axes.append(ax := fig.add_subplot(gs[i + 1, 0]))
        gdf = get_heatmap_gdf(
            regions=regions,
            values=values,
            projection="sagittal",
            remove_redundant_parents=remove_redundant_parents,
            combine_child_patches=combine_child_patches,
            position=coord,
        )
        gdfs.append(gdf)
        gdf.plot(
            column="value",
            cmap=cmap,
            missing_kwds=missing_kwds,
            ax=ax,
            norm=norm,
            **patch_params,
        )
        ax.set_xlim(0, max_ap)
        ax.set_ylim(0, max_dv)
        ax.invert_yaxis()
        if horizontal_upright:
            axlinefunc = ax_top.axvline
        else:
            axlinefunc = ax_top.axhline
        axlinefunc(
            coord, **{"color": "k", "linestyle": "--", "lw": 0.1} | plane_line_params
        )

    if horizontal_upright:
        ax_top.set_xlim(0, max_ml)
        ax_top.set_ylim(max_ap, 0)
    else:
        ax_top.set_xlim(0, max_ap)
        ax_top.set_ylim(0, max_ml)

    axes.append(ax_cbar := fig.add_subplot(gs[len(sagittal_planes) + 1, 0]))
    fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(*clevels),
            cmap=cmap,
        ),
        ax=ax_cbar,
        fraction=0.5,
        orientation="horizontal",
        location="bottom",
    )
    for ax in axes:
        ax.set_aspect(1)
        ax.set_axis_off()
        ax.set_clip_on(False)

    if interactive:
        chart = plot_gdf_alt(gdfs, ccf_colors=False, cmap=cmap, clevels=clevels)
        return chart, tuple(gdfs)
    else:
        return fig, tuple(gdfs)


def plot_gdf_alt(
    gdfs: gpd.GeoDataFrame | Iterable[gpd.GeoDataFrame],
    ccf_colors: bool = False,
    cmap: str = "viridis",
    value_name: str = "value",
    clevels: tuple[float, float] | None = None,
) -> alt.Chart:
    if isinstance(gdfs, gpd.GeoDataFrame):
        gdfs = (gdfs,)
    else:
        gdfs = tuple(gdfs)
    vol = ccf_utils.get_ccf_volume(True, True)
    max_ap = vol.shape[2] * ccf_utils.RESOLUTION_UM
    max_dv = vol.shape[1] * ccf_utils.RESOLUTION_UM
    max_ml = vol.shape[0] * ccf_utils.RESOLUTION_UM
    charts = []

    @functools.cache
    def get_background_gdf(projection: str, position: float | None):
        return gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.union_all(
                        list(
                            get_heatmap_gdf(
                                regions=[],
                                values=[],
                                projection=projection,
                                position=position,
                            )["geometry"].values
                        )
                    )
                ],
            }
        )

    def get_fit(projection: str, is_top_upright):
        xmin, ymin = 0, 0
        if projection in ("top", "horizontal"):
            if is_top_upright:
                xmax, ymax = max_ml, max_ap
            else:
                xmax, ymax = max_ap, max_ml
        elif projection == "sagittal":
            xmax, ymax = max_ap, max_dv
        elif projection == "coronal":
            xmax, ymax = max_ml, max_dv
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                    ]
                ],
            },
            "properties": {},
        }

    for gdf in gdfs:
        projection = gdf["projection"].iloc[0]
        if projection in ("top", "horizontal"):
            is_upright = bool(
                gdf.geometry.centroid.y.max() > gdf.geometry.centroid.x.max()
            )
            background_position = None if projection == "top" else max_dv / 2
        elif projection == "sagittal":
            background_position = (max_ml / 2) + 250
        elif projection == "coronal":
            background_position = max_ap / 2
        else:
            raise ValueError(f"Invalid projection {projection}")

        tooltip = ["id:Q", "acronym:N", "name:N", "hemisphere:N"]
        if ccf_colors:
            color = alt.Color("acronym").scale(
                domain=ccf_utils.get_ccf_structure_tree_df()["acronym"].to_list(),
                range=ccf_utils.get_ccf_structure_tree_df()["color_hex_str"].to_list(),
            )
        else:
            tooltip.append("value:Q")
            color = alt.Color(
                "value:Q",
                title=value_name,
                scale=alt.Scale(scheme=cmap.lower(), domainMax=clevels[1], domainMin=clevels[0]),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
                # condition=condition,
            )
        chart = (
            alt.Chart(gdf)
            .mark_geoshape(
                strokeWidth=0.05,
                stroke="darkgrey",
            )
            .encode(
                tooltip=tooltip,
                color=color,
            )
            .project(
                type="identity",
                reflectY=projection != "sagittal",
                fit=get_fit(
                    projection,
                    is_upright if projection in ("top", "horizontal") else None,
                ),
            )
        )
        null_slice = (
            alt.Chart(gdf[gdf["value"].isna() | gdf["value"].isnull()])
            .mark_geoshape(
                strokeWidth=0.05,
                stroke="white",
            )
            .encode(
                tooltip=tooltip,
                color=alt.value("#eee"),
            )
            .project(
                type="identity",
                reflectY=projection != "sagittal",
                fit=get_fit(
                    projection,
                    is_upright if projection in ("top", "horizontal") else None,
                ),
            )
        )
        chart = alt.layer(null_slice, chart)

        with_background = True
        if with_background:
            background = (
                alt.Chart(get_background_gdf(projection, background_position))
                .mark_geoshape(strokeWidth=0.2, stroke="darkgrey")
                .encode(
                    color=alt.value("#fff"),
                )
                .project(
                    type="identity",
                    reflectY=projection != "sagittal",
                    fit=get_fit(
                        projection,
                        is_upright if projection in ("top", "horizontal") else None,
                    ),
                )
            )
            chart = alt.layer(background, chart)

        # add lines (positions aren't correct):
        """
        if projection in ("top", "horizontal"):
            other_gdfs = [
                gdf
                for gdf in gdfs
                if gdf["projection"].iloc[0] not in ("top", "horizontal")
            ]
            positions = [gdf["position"].iloc[0] for gdf in other_gdfs]
            projections = [gdf["projection"].iloc[0] for gdf in other_gdfs]
            for pos, proj in zip(positions, projections):
                if proj == "sagittal":
                    chart += (
                        alt.Chart(pl.DataFrame({"y": [pos]}))
                        .mark_rule(strokeDash=[2, 2])
                        .encode(y="y")
                    )
                elif proj == "coronal":
                    chart += alt.Chart({"x": [pos]}).mark_rule().encode(x="x")
        """
        charts.append(chart)
    return alt.hconcat(*charts).configure_legend(disable=True)
