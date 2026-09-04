# stdlib imports --------------------------------------------------- #
from __future__ import annotations

import contextlib
import functools
import logging
import logging.handlers
import pathlib
import typing
from collections.abc import Iterable
from typing import Literal

# 3rd-party imports necessary for processing ----------------------- #
import dr_datacube
import polars as pl
import upath

import dynamic_routing_analysis.codeocean_utils as codeocean_utils

logger = logging.getLogger(__name__)

LATE_AUTOREWARDS_SESSION_FILTER: pl.Expr = (
    pl.col("keywords").list.contains("late_autorewards").eq(True)
)
EARLY_AUTOREWARDS_SESSION_FILTER = (
    pl.col("keywords").list.contains("early_autorewards").eq(True)
)

def get_datacube_version() -> str:
    return dr_datacube.config.version


def _normalize_datacube_version(version: str) -> str:
    if version == "any":
        return dr_datacube.config.version
    return f"v{version.removeprefix('v')}"


def is_datacube_available() -> bool:
    with contextlib.suppress(FileNotFoundError):
        return dr_datacube.config.asset_dir.exists()
    return False


# data access ------------------------------------------------------- #
@functools.cache
def get_session_table() -> pl.DataFrame:
    if codeocean_utils.on_code_ocean():
        return pl.read_parquet(
            (codeocean_utils.get_datacube_dir() / "session_table.parquet").as_posix()
        )
    return pl.read_parquet(
        's3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet',
        storage_options={"skip_signature": "true"},
    )


@typing.overload
def get_df(
    component: str,
    lazy: Literal[False] = False,
    nwb: bool = False,
    session_id: str | None = None,
    version: str | None = None,
) -> pl.DataFrame:
    ...


@typing.overload
def get_df(
    component: str,
    lazy: Literal[True] = True,
    nwb: bool = False,
    session_id: str | None = None,
    version: str | None = None,
) -> pl.LazyFrame:
    ...


def get_df(
    component: str,
    lazy: bool = False,
    nwb: bool = False,
    session_id: str | None = None,
    version: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    if version is None:
        lf = dr_datacube.get_lf(
            component, session_id=session_id, nwb=nwb
        )
    else:
        with dr_datacube.config.override(
            version=_normalize_datacube_version(version), use_cache=True
        ):
            lf = dr_datacube.get_lf(
                component, session_id=session_id, nwb=nwb
            )
    if lazy:
        return lf
    return lf.collect()

@typing.overload
def get_nwb_paths(session_id: str) -> pathlib.Path: ...


@typing.overload
def get_nwb_paths(session_id: Literal[None] = None) -> tuple[pathlib.Path, ...]: ...


@functools.cache
def get_nwb_paths(
    session_id: str | None = None,
) -> pathlib.Path | tuple[pathlib.Path, ...]:
    """Returns a single path if a session ID is provided"""
    paths = [upath.UPath(p) for p in dr_datacube.list_nwb_sources()]
    if session_id:
        try:
            return next(p for p in paths if p.stem == session_id)
        except StopIteration:
            raise FileNotFoundError(
                f"Cannot find NWB file for {session_id!r} in {dr_datacube.config.nwb_dir}"
            ) from None
    else:
        return tuple(p for p in paths if p.is_file())


def _parse_nwb_path_from_input(
    session_id_or_path: str | pathlib.Path,
    raise_on_missing: bool = True,
) -> upath.UPath | pathlib.Path | None:
    if isinstance(session_id_or_path, (pathlib.Path, upath.UPath)):
        return session_id_or_path
    if not isinstance(session_id_or_path, str):
        raise TypeError(
            f"Input should be a session ID (str) or path to an NWB file (str/Path), got: {session_id_or_path!r}"
        )
    if upath.UPath(session_id_or_path).exists():
        return upath.UPath(session_id_or_path)
    elif session_id_or_path.endswith(".nwb") and any(
        p.name == session_id_or_path for p in get_nwb_paths()
    ):
        return next(p for p in get_nwb_paths() if p.name == session_id_or_path)
    else:
        try:
            return next(p for p in get_nwb_paths() if p.stem == session_id_or_path)
        except StopIteration:
            msg = f"Could not find NWB file for {session_id_or_path!r}"
            if not raise_on_missing:
                logger.error(msg)
                return None
            else:
                raise FileNotFoundError(
                    f"{msg}. Available files: {[p.name for p in get_nwb_paths()]}"
                ) from None


def get_pynwb(
    session_id_or_path: str | pathlib.Path,
    raise_on_missing: bool = True,
    raise_on_bad_file: bool = True,
) -> "pynwb.NWBFile" | None:  # noqa
    import pynwb

    nwb_path = _parse_nwb_path_from_input(
        session_id_or_path, raise_on_missing=raise_on_missing
    )
    if nwb_path is None:
        return None
    logger.info(f"Reading {nwb_path}")
    try:
        nwb = pynwb.NWBHDF5IO(nwb_path).read()
    except RecursionError:
        msg = f"{nwb_path.name} cannot be read due to RecursionError (hdf5 may still be accessible)"
        if not raise_on_bad_file:
            logger.error(msg)
            return None
        else:
            raise RecursionError(msg)
    else:
        return nwb


def get_lazynwb(
    session_id_or_path: str | pathlib.Path,
    raise_on_missing: bool = True,
    raise_on_bad_file: bool = True,
) -> "dr_datacube.lazynwb.LazyNWB" | None:  # noqa

    nwb_path = _parse_nwb_path_from_input(
        session_id_or_path, raise_on_missing=raise_on_missing
    )
    if nwb_path is None:
        return None
    logger.info(f"Reading {nwb_path}")
    try:
        nwb = dr_datacube.lazynwb.LazyNWB(nwb_path)
    except RecursionError:
        msg = f"{nwb_path.name} cannot be read due to RecursionError (hdf5 may still be accessible)"
        if not raise_on_bad_file:
            logger.error(msg)
            return None
        else:
            raise RecursionError(msg)
    else:
        return nwb


def unit_id_to_session_id(unit_id: str) -> str:
    return unit_id.rpartition("_")[0]


def combine_exprs(exprs: Iterable[pl.Expr]) -> pl.Expr:
    return pl.Expr.and_(*exprs)

def get_passing_blocks_performance_filter(
    cross_modality_dprime: float | None = 1.0,
    min_trials: int | None = 10,
    min_contingent_rewards: int | None = 10,
) -> pl.Expr:
    cross_modal_dprime_filter: pl.Expr = (
        pl.col("cross_modality_dprime") >= cross_modality_dprime
        if cross_modality_dprime is not None
        else pl.lit(True)
    )
    min_n_trials_filter: pl.Expr = (
        pl.col("n_trials") >= min_trials if min_trials is not None else pl.lit(True)
    )
    min_n_responses_filter: pl.Expr = (
        pl.col("n_responses") >= min_contingent_rewards
        if min_contingent_rewards is not None
        else pl.lit(True)
    )
    return combine_exprs([cross_modal_dprime_filter, min_n_trials_filter, min_n_responses_filter])


def get_prod_trials(
    cross_modal_dprime_threshold: float = 1.0,
    late_autorewards: bool | None = None,
    by_session: bool = True,
    include_templeton: bool = False,
) -> pl.DataFrame:
    """
    late_autorewards: If False/True, include sessions with early/late autorewards, respectively.
    If None, include both.

    by_session: If True (default), all blocks within the session are returned if the session as a
    whole passed the performance criteria. If False, passing blocks for all sessions will be returned, even if the
    session as a whole does not meet performance criteria for good behavior.
    """
    late_autorewards_expr = {
        True: LATE_AUTOREWARDS_SESSION_FILTER,
        False: EARLY_AUTOREWARDS_SESSION_FILTER,
        None: pl.lit(True),
    }[late_autorewards]

    # session_ids to use based on project, experiment-type, training history etc.:
    session_table = (
        dr_datacube.get_session_table(with_behavior_filter=True)
        .filter(late_autorewards_expr)
    )
    session_types = ["brainwide"]
    if include_templeton:
        session_types.append("templeton")
    session_table = session_table.filter(pl.col("session_type").is_in(session_types))

    if by_session: 
        # keep all blocks from sessions that pass the performance criteria
        trials = (
            get_df("trials")
            .filter(
                pl.col("session_id").is_in(session_table["session_id"].implode())
            )
        )
    else:
        if include_templeton:
            templeton_filter = pl.col("session_id").is_in(session_table.filter(pl.col("session_type") == "templeton")["session_id"].implode())
        else:
            templeton_filter = pl.lit(False)
        passing_blocks: pl.DataFrame = (
            get_df("performance")
            .filter(
                get_passing_blocks_performance_filter(cross_modality_dprime=cross_modal_dprime_threshold) 
                | templeton_filter
            )
        )
        trials = (
            get_df("trials")
            .join(
                passing_blocks,
                on=["session_id", 'block_index'],
                how="semi", # filter trials to only those in passing blocks
            )
        )
    return (
        trials
        # add a column that indicates if the first block in a session is aud context:
        .with_columns(
            (pl.col("rewarded_modality").first() == "aud")
            .over("session_id")
            .alias("is_first_block_aud"),
        )
        .sort('session_id', 'block_index', 'trial_index')
    )
