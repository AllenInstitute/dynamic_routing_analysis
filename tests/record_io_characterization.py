"""Record intentional baselines for the real-data characterization tests.

Run this script only after reviewing an intentional behavior change:

    uv run python tests/record_io_characterization.py --record
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import upath

from dynamic_routing_analysis import io_utils

REFERENCE_DIR = Path(__file__).parent / "fixtures" / "io_characterization"
PRIMARY_SESSION_ID = "626791_2022-08-15"
SECONDARY_SESSION_ID = "620263_2022-07-26"
SESSION_IDS = (PRIMARY_SESSION_ID, SECONDARY_SESSION_ID)
TRANSPORT_COLUMNS = {"_nwb_path", "_table_path", "_table_index"}
NWB_VERSION = "0.0.289"


def _pinned_nwb_path(session_id: str) -> upath.UPath:
    return upath.UPath(
        "s3://aind-scratch-data/dynamic-routing/cache/"
        f"nwb/v{NWB_VERSION}/{session_id}.nwb",
        anon=True,
    )


def _as_pandas(frame: object) -> pd.DataFrame:
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    to_pandas = getattr(frame, "to_pandas", None)
    if callable(to_pandas):
        result = to_pandas()
        if isinstance(result, pd.DataFrame):
            return result
    return pd.DataFrame(frame)


def _write_frame(frame: object, path: Path) -> None:
    normalized = _as_pandas(frame).drop(
        columns=list(TRANSPORT_COLUMNS), errors="ignore"
    )
    normalized.reset_index(drop=True).to_parquet(path, index=False)


def _build_event_design(
    behavior_info: dict, fit: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_names = ("vis1", "hit", "context", "session_time")
    kernels = {}
    for name in kernel_names:
        config = copy.deepcopy(io_utils.master_kernels_list[name])
        config.update(length=0, offset=0, orthogonalize=None, num_weights=None)
        kernels[name] = config

    design = io_utils.DesignMatrix(fit)
    design, result_fit = io_utils.add_kernels(
        design,
        {"kernels": kernels, "use_context_belief": False},
        PRIMARY_SESSION_ID,
        fit,
        behavior_info,
    )
    if result_fit["failed_kernels"]:
        raise RuntimeError(
            f"failed to record kernels: {result_fit['kernel_error_dict']}"
        )
    matrix = design.get_X()
    return (
        matrix.values,
        matrix.coords["timestamps"].values,
        matrix.coords["weights"].values.astype(str),
    )


def _make_fit(behavior_info: dict) -> dict:
    run_params = {
        "time_of_interest": "trial",
        "spike_bin_width": 0.1,
        "trial_start_time": -2.0,
        "trial_stop_time": 3.0,
        "quiescent_start_time": -1.5,
        "quiescent_stop_time": 0.0,
        "input_offsets": False,
        "leave_blocks_out": False,
    }
    fit = io_utils.establish_timebins(run_params, {}, behavior_info)
    return fit


def record() -> None:
    original_get_nwb_paths = io_utils.datacube_utils.get_nwb_paths
    original_is_available = io_utils.datacube_utils.is_datacube_available
    io_utils.datacube_utils.get_nwb_paths = _pinned_nwb_path
    io_utils.datacube_utils.is_datacube_available = lambda: True
    try:
        session_values = {
            session_id: io_utils.get_session_data_from_datacube(session_id)
            for session_id in SESSION_IDS
        }
        _, behavior_info = session_values[PRIMARY_SESSION_ID]
        fit = _make_fit(behavior_info)
        kernels = {
            "running": io_utils.running(
                "running", PRIMARY_SESSION_ID, fit, behavior_info
            ),
            "pupil": io_utils.pupil("pupil", PRIMARY_SESSION_ID, fit, behavior_info),
            "nose": io_utils.facial_features(
                "nose", PRIMARY_SESSION_ID, fit, behavior_info
            ),
        }
        matrix, timestamps, weights = _build_event_design(behavior_info, fit)
        kernels.update(
            event_design_matrix=matrix,
            event_design_timestamps=timestamps,
            event_design_weights=weights,
        )
    finally:
        io_utils.datacube_utils.get_nwb_paths = original_get_nwb_paths
        io_utils.datacube_utils.is_datacube_available = original_is_available

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for session_id, (units, session_behavior_info) in session_values.items():
        _write_frame(units, REFERENCE_DIR / f"{session_id}_units.parquet")
        _write_frame(
            session_behavior_info["trials"],
            REFERENCE_DIR / f"{session_id}_trials.parquet",
        )
        _write_frame(
            session_behavior_info["epoch_info"],
            REFERENCE_DIR / f"{session_id}_epochs.parquet",
        )
        np.save(
            REFERENCE_DIR / f"{session_id}_dprime.npy",
            np.asarray(session_behavior_info["dprime"]),
            allow_pickle=False,
        )
    np.savez_compressed(REFERENCE_DIR / f"{PRIMARY_SESSION_ID}_kernels.npz", **kernels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="confirm that existing reference files may be replaced",
    )
    args = parser.parse_args()
    if not args.record:
        parser.error("pass --record to replace characterization references")
    record()
