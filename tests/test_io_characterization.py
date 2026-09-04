"""Characterization tests for session data and GLM kernel values.

These tests intentionally read pinned real sessions. Enable them with
``RUN_IO_CHARACTERIZATION=1 uv run --group testing pytest
tests/test_io_characterization.py``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import upath

from dynamic_routing_analysis import io_utils

RUN_CHARACTERIZATION = os.environ.get("RUN_IO_CHARACTERIZATION") == "1"
REFERENCE_DIR = Path(__file__).parent / "fixtures" / "io_characterization"
PRIMARY_SESSION_ID = "626791_2022-08-15"
SECONDARY_SESSION_ID = "620263_2022-07-26"
SESSION_IDS = (PRIMARY_SESSION_ID, SECONDARY_SESSION_ID)
TRANSPORT_COLUMNS = {"_nwb_path", "_table_path", "_table_index"}
NWB_VERSION = "0.0.289"

pytestmark = [
    pytest.mark.io_characterization,
    pytest.mark.skipif(
        not RUN_CHARACTERIZATION,
        reason="set RUN_IO_CHARACTERIZATION=1 to read pinned real-session data",
    ),
]


def _pinned_nwb_path(session_id: str) -> upath.UPath:
    return upath.UPath(
        "s3://aind-scratch-data/dynamic-routing/cache/"
        f"nwb/v{NWB_VERSION}/{session_id}.nwb",
        anon=True,
    )


@pytest.fixture(scope="module", autouse=True)
def pinned_anonymous_nwb_source() -> None:
    """Pin source data and avoid environment-specific AWS credentials."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(io_utils.datacube_utils, "get_nwb_paths", _pinned_nwb_path)
    yield
    monkeypatch.undo()


@pytest.fixture(scope="module")
def loaded_sessions() -> dict[str, tuple[object, dict]]:
    return {
        session_id: io_utils.get_session_data_from_datacube(session_id)
        for session_id in SESSION_IDS
    }


@pytest.fixture(scope="module")
def primary_session(
    loaded_sessions: dict[str, tuple[object, dict]],
) -> tuple[dict, dict]:
    _, behavior_info = loaded_sessions[PRIMARY_SESSION_ID]
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
    return behavior_info, fit


def _as_pandas(frame: object) -> pd.DataFrame:
    """Normalize supported dataframe containers for value comparison only."""
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


def _assert_frame_preserves_reference(actual: object, reference_path: Path) -> None:
    expected = pd.read_parquet(reference_path)
    actual_df = _as_pandas(actual).drop(
        columns=list(TRANSPORT_COLUMNS), errors="ignore"
    )
    missing_columns = expected.columns.difference(actual_df.columns)
    assert missing_columns.empty, f"missing columns: {missing_columns.tolist()}"

    pd.testing.assert_frame_equal(
        actual_df.loc[:, expected.columns].reset_index(drop=True),
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-7,
        atol=1e-7,
    )


def _build_event_design(
    behavior_info: dict, fit: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_names = ("vis1", "hit", "context", "session_time")
    kernels = {}
    for name in kernel_names:
        config = copy.deepcopy(io_utils.master_kernels_list[name])
        config.update(length=0, offset=0, orthogonalize=None, num_weights=None)
        kernels[name] = config

    run_params = {
        "kernels": kernels,
        "use_context_belief": False,
    }
    design = io_utils.DesignMatrix(fit)
    design, result_fit = io_utils.add_kernels(
        design,
        run_params,
        PRIMARY_SESSION_ID,
        fit,
        behavior_info,
    )
    assert result_fit["failed_kernels"] == set()
    matrix = design.get_X()
    return (
        matrix.values,
        matrix.coords["timestamps"].values,
        matrix.coords["weights"].values.astype(str),
    )


@pytest.mark.parametrize("session_id", SESSION_IDS)
def test_loaded_session_values_are_unchanged(
    session_id: str, loaded_sessions: dict[str, tuple[object, dict]]
) -> None:
    units, behavior_info = loaded_sessions[session_id]
    assert behavior_info["session_id"] == session_id

    _assert_frame_preserves_reference(
        units, REFERENCE_DIR / f"{session_id}_units.parquet"
    )
    _assert_frame_preserves_reference(
        behavior_info["trials"],
        REFERENCE_DIR / f"{session_id}_trials.parquet",
    )
    _assert_frame_preserves_reference(
        behavior_info["epoch_info"],
        REFERENCE_DIR / f"{session_id}_epochs.parquet",
    )
    expected_dprime = np.load(REFERENCE_DIR / f"{session_id}_dprime.npy")
    np.testing.assert_allclose(
        np.asarray(behavior_info["dprime"]),
        expected_dprime,
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )


def test_running_kernel_values_are_unchanged(
    primary_session: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior_info, fit = primary_session
    monkeypatch.setattr(io_utils.datacube_utils, "is_datacube_available", lambda: True)

    actual = io_utils.running("running", PRIMARY_SESSION_ID, fit, behavior_info)
    with np.load(REFERENCE_DIR / "626791_2022-08-15_kernels.npz") as reference:
        expected = reference["running"]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )


def test_pupil_kernel_values_are_unchanged(
    primary_session: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior_info, fit = primary_session
    monkeypatch.setattr(io_utils.datacube_utils, "is_datacube_available", lambda: True)

    actual = io_utils.pupil("pupil", PRIMARY_SESSION_ID, fit, behavior_info)
    with np.load(REFERENCE_DIR / "626791_2022-08-15_kernels.npz") as reference:
        expected = reference["pupil"]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )


def test_facial_feature_kernel_values_are_unchanged(
    primary_session: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior_info, fit = primary_session
    monkeypatch.setattr(io_utils.datacube_utils, "is_datacube_available", lambda: True)

    actual = io_utils.facial_features("nose", PRIMARY_SESSION_ID, fit, behavior_info)
    with np.load(REFERENCE_DIR / "626791_2022-08-15_kernels.npz") as reference:
        expected = reference["nose"]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )


def test_event_design_matrix_values_are_unchanged(
    primary_session: tuple[dict, dict],
) -> None:
    behavior_info, fit = primary_session
    actual_matrix, actual_timestamps, actual_weights = _build_event_design(
        behavior_info, fit
    )

    with np.load(REFERENCE_DIR / "626791_2022-08-15_kernels.npz") as expected:
        np.testing.assert_allclose(
            actual_matrix,
            expected["event_design_matrix"],
            rtol=1e-7,
            atol=1e-7,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual_timestamps,
            expected["event_design_timestamps"],
            rtol=1e-7,
            atol=1e-7,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            actual_weights,
            expected["event_design_weights"],
        )
