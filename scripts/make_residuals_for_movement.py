"""Save trialwise baseline residuals from the movement encoding model.

Run from the repository environment, for example::

    python scripts/make_residuals_for_movement.py --session-id 713655_2024-08-05

The design matrix settings must match the run that produced the saved weights.
"""

from __future__ import annotations

import argparse

import dr_datacube
import numpy as np
import pandas as pd
import polars as pl

from dynamic_routing_analysis import encoding_utils, io_utils
from dynamic_routing_analysis.design_matrix_utils import build_design_matrix

SCORES_PATH = "s3://aind-scratch-data/dynamic-routing/encoding/scores"
RESULTS_PATH = "s3://aind-scratch-data/dynamic-routing/encoding/results"
RESIDUALS_PATH = "s3://aind-scratch-data/dynamic-routing/glm_residuals"
UNITS_PATH = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb_components/"
    "v0.0.289/consolidated/units.parquet"
)
READ_OPTIONS = {"skip_signature": "true"}


def get_units_to_process(
    result_prefix: str, run_id: str, session_id: str | None = None
) -> pl.DataFrame:
    """Select units passing the requested GLM score and unit quality filters."""
    scores = (
        pl.scan_parquet(
            f"{SCORES_PATH}/{result_prefix}_{run_id}.parquet",
            storage_options=READ_OPTIONS,
        )
        .filter(pl.col("project").cast(pl.Utf8).str.to_lowercase() != "templeton")
    )
    if session_id is not None:
        scores = scores.filter(pl.col("session_id") == session_id)

    unit_metrics = pl.scan_parquet(UNITS_PATH, storage_options=READ_OPTIONS).select(
        "unit_id", "activity_drift", "isi_violations_ratio",
        "presence_ratio", "amplitude_cutoff", "decoder_label",
    )
    return (
        scores.join(unit_metrics, on="unit_id", how="inner")
        .filter(
            (pl.col("activity_drift") <= 0.2)
            & (pl.col("isi_violations_ratio") <= 0.5)
            & (pl.col("amplitude_cutoff") <= 0.1)
            & (pl.col("presence_ratio") >= 0.7)
            & (pl.col("decoder_label") != "noise")
            & (pl.col("cv_test_fullmodel") >= 0.005)
        )
        .select("session_id", "unit_id")
        .unique()
        .collect()
    )


def get_fullmodel_weights(
    unit_ids: list[str], result_prefix: str, run_id: str
) -> pl.DataFrame:
    return (
        pl.scan_parquet(
            f"{RESULTS_PATH}/{result_prefix}_{run_id}/",
            storage_options=READ_OPTIONS,
        )
        .filter(pl.col("unit_id").is_in(unit_ids) & (pl.col("model_label") == "fullmodel"))
        .select("unit_id", "weights")
        .collect()
    )


def boxcar_rate(
    spike_counts: np.ndarray,
    epoch_trace: np.ndarray,
    spike_bin_width: float,
    window: int = 5,
) -> np.ndarray:
    """Center a boxcar within each epoch, then convert counts to spikes/s."""
    spike_counts = np.asarray(spike_counts, dtype=float)
    epoch_trace = np.asarray(epoch_trace)
    if len(spike_counts) != len(epoch_trace):
        raise ValueError("Spike counts and epoch trace must have the same length")
    if window < 1:
        raise ValueError("Boxcar window must be positive")

    rate = np.empty(len(spike_counts), dtype=float)
    for epoch in pd.unique(epoch_trace):
        mask = epoch_trace == epoch
        rate[mask] = (
            pd.Series(spike_counts[mask])
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy()
            / spike_bin_width
        )
    return rate


def calculate_trialwise_residuals(
    design_matrix,
    weights: pl.DataFrame,
    units: pl.DataFrame,
    trials: pd.DataFrame,
    *,
    fit: dict,
    pre_window: float,
) -> pd.DataFrame:
    """Return one row per trial and one normalized residual column per unit."""
    if pre_window <= 0:
        raise ValueError("pre_window must be positive")
    if "stim_start_time" not in trials or "rewarded_modality" not in trials:
        raise ValueError("Trials must contain stim_start_time and rewarded_modality")

    bin_centers = np.asarray(fit["bin_centers"], dtype=float)
    feature_mat = np.asarray(design_matrix.data, dtype=float)
    if feature_mat.shape[0] != len(bin_centers):
        raise ValueError("Design matrix rows do not match its timestamps")
    np.testing.assert_allclose(design_matrix.timestamps.values, bin_centers)
    timebins = fit["timebins"]
    spike_bin_width = fit["spike_bin_width"]

    stim_times = trials["stim_start_time"].to_numpy(dtype=float)
    baseline_bins = [
        (bin_centers >= stim_time - pre_window) & (bin_centers < stim_time)
        for stim_time in stim_times
    ]
    residual_means = pd.DataFrame(index=pd.RangeIndex(len(trials), name="index"))
    residual_means["context"] = pd.NA
    for trial_no, mask in enumerate(baseline_bins):
        if not mask.any():
            print(f"skipping trial {trial_no} with no residuals")
            continue
        residual_means.loc[trial_no, "context"] = trials.iloc[trial_no][
            "rewarded_modality"
        ]
    residual_columns: dict[str, np.ndarray] = {}

    unit_spikes = {}
    for row in units.iter_rows(named=True):
        unit_id = str(row["unit_id"])
        if unit_id in unit_spikes:
            raise ValueError(f"Duplicate spike times for unit {unit_id}")
        if row["spike_times"] is None:
            raise ValueError(f"Spike times are missing for unit {unit_id}")
        unit_spikes[unit_id] = np.sort(np.asarray(row["spike_times"], dtype=float))

    seen_weights = set()
    for row in weights.iter_rows(named=True):
        unit_id = str(row["unit_id"])
        if unit_id in seen_weights:
            raise ValueError(f"Duplicate fullmodel weights for unit {unit_id}")
        seen_weights.add(unit_id)
        if unit_id not in unit_spikes:
            raise ValueError(f"Spike times are missing for unit {unit_id}")
        coefficients = np.asarray(row["weights"], dtype=float)
        if coefficients.size != feature_mat.shape[1]:
            raise ValueError(
                f"{unit_id}: {coefficients.size} weights for "
                f"{feature_mat.shape[1]} design-matrix columns"
            )

        # Predictions are counts/bin, like the training target.
        prediction = feature_mat @ coefficients
        fullmodel_rate = prediction / spike_bin_width
        spike_counts = io_utils.get_spike_counts(unit_spikes[unit_id], timebins)
        spike_rate = spike_counts / spike_bin_width
        spike_rate_std = np.std(spike_rate)
        if spike_rate_std == 0 or not np.isfinite(spike_rate_std):
            print(f"Unit {unit_id} has zero or invalid spike-rate variance; saving NaN residuals")
            residual_columns[unit_id] = np.full(len(trials), np.nan)
            continue
        residuals = (spike_rate - fullmodel_rate) / spike_rate_std
        residual_columns[unit_id] = np.array(
            [
                np.mean(residuals[mask]) if mask.any() else np.nan
                for mask in baseline_bins
            ]
        )

    if residual_columns:
        residual_means = pd.concat(
            [
                residual_means,
                pd.DataFrame(residual_columns, index=residual_means.index),
            ],
            axis=1,
            copy=False,
        )

    return residual_means


def process_session(
    session_id: str,
    unit_ids: list[str],
    *,
    result_prefix: str,
    run_id: str,
    pre_window: float,
) -> str:
    weights = get_fullmodel_weights(unit_ids, result_prefix, run_id)
    missing_weights = set(unit_ids) - set(weights["unit_id"].to_list())
    if missing_weights:
        raise ValueError(f"{session_id}: missing fullmodel weights for {sorted(missing_weights)}")

    # Match the movement model's feature settings and the notebook's data source.
    design_matrix = build_design_matrix(
        session_id,
        time_of_interest="full_trial",
        input_variables=["movements"],
        orthogonalize_against=None,
        spike_bin_width=0.1,
    )
    with dr_datacube.config.override(anon=True, use_cache=True):
        lazy_units, behavior_info = io_utils.get_session_data(session_id, lazy=True)
        units = (
            lazy_units.filter(pl.col("unit_id").is_in(unit_ids))
            .select("unit_id", "spike_times")
            .collect()
        )
    params = encoding_utils.Params.model_construct(
        result_prefix=result_prefix,
        run_id=run_id,
        time_of_interest="full_trial",
        spike_bin_width=0.1,
    )
    fit = io_utils.establish_timebins(params.model_dump(), {}, behavior_info)

    residual_means = calculate_trialwise_residuals(
        design_matrix,
        weights,
        units,
        behavior_info["trials"],
        fit=fit,
        pre_window=pre_window,
    )
    residual_means_reset = residual_means.reset_index()
    residual_means_reset.columns = [str(col) for col in residual_means_reset.columns]
    if not residual_means_reset.columns.is_unique:
        duplicate_cols = residual_means_reset.columns[
            residual_means_reset.columns.duplicated()
        ].tolist()
        print(
            f"Duplicate residual columns found for session {session_id}: "
            f"{duplicate_cols}. Keeping first occurrence."
        )
        residual_means_reset = residual_means_reset.loc[
            :, ~residual_means_reset.columns.duplicated()
        ]
    save_path = f"{RESIDUALS_PATH}/{session_id}_{result_prefix}_{run_id}.parquet"
    pl.from_pandas(residual_means_reset).write_parquet(save_path)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-prefix", default="v289_movementfeatures_fulltrial")
    parser.add_argument("--run-id", default="1")
    parser.add_argument("--session-id", help="Process one session; default is all eligible sessions")
    parser.add_argument("--pre-window", type=float, default=0.5, help="Baseline seconds before stimulus")
    args = parser.parse_args()
    if args.pre_window <= 0:
        parser.error("--pre-window must be positive")

    selected = get_units_to_process(args.result_prefix, args.run_id, args.session_id)
    if selected.is_empty():
        raise ValueError("No units match the requested run and session filters")

    for session_id in sorted(selected["session_id"].unique().to_list()):
        unit_ids = selected.filter(pl.col("session_id") == session_id)[
            "unit_id"
        ].unique().to_list()
        print(f"Processing {session_id}: {len(unit_ids)} units")
        save_path = process_session(
            session_id,
            unit_ids,
            result_prefix=args.result_prefix,
            run_id=args.run_id,
            pre_window=args.pre_window,
        )
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
