# stdlib imports --------------------------------------------------- #
from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import logging.handlers
import pathlib
import re
from collections.abc import Iterable
import typing

# 3rd-party imports necessary for processing ----------------------- #
from matplotlib.pylab import f
import npc_lims
import polars as pl
import upath

import dynamic_routing_analysis.codeocean_utils as codeocean_utils

logger = logging.getLogger(__name__)

PROD_EPHYS_SESSION_FILTER = pl.Expr.and_(
    *[
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
    ]
)
"""does not include Templeton sessions"""

LATE_AUTOREWARDS_SESSION_FILTER: pl.Expr = (
    pl.col("keywords").list.contains("late_autorewards").eq(True)
)
EARLY_AUTOREWARDS_SESSION_FILTER = (
    pl.col("keywords").list.contains("early_autorewards").eq(True)
)

DEFAULT_UNIT_QC = pl.Expr.and_(
    *[
        pl.col("activity_drift") <= 0.2,
        pl.col("isi_violations_ratio") <= 0.5,
        pl.col("amplitude_cutoff") <= 0.1,
        pl.col("presence_ratio") >= 0.7,
        pl.col("decoder_label") != "noise",
    ]
)

@dataclasses.dataclass
class DatacubeConfig:
    use_scratch_dir: bool = False
    datacube_version: str | None = None

    @property
    def nwb_dir(self) -> upath.UPath | pathlib.Path:
        if not self.datacube_version:
            self.datacube_version = npc_lims.get_current_cache_version()
        if self.use_scratch_dir:
            return npc_lims.get_nwb_path(
                "366122_2023-12-31", version=self.datacube_version
            ).parent
        if codeocean_utils.is_capsule() or codeocean_utils.is_pipeline():
            d = next(codeocean_utils.get_datacube_dir().rglob("*.nwb"), None)
            if d is None:
                raise FileNotFoundError(
                    f"Cannot find NWB files in {codeocean_utils.get_datacube_dir()}"
                )
            datacube_version = parse_version(d.as_posix())
            if datacube_version != self.datacube_version:
                logger.warning(
                    f"Requested datacube version {self.datacube_version} in config does not match discovered asset {d.parent=}"
                )
            return d.parent
        raise ValueError(
            f"Cannot determine NWB dir: {self.datacube_version=} {self.use_scratch_dir=}"
        )

    @property
    def consolidated_parquet_dir(self) -> upath.UPath | pathlib.Path:
        if self.use_scratch_dir:
            return (
                self.nwb_dir.parent.parent
                / f"nwb_components/{self.datacube_version}/consolidated"
            )
        if codeocean_utils.is_capsule() or codeocean_utils.is_pipeline():
            datacube_version = parse_version(
                codeocean_utils.get_datacube_dir().as_posix()
            )
            if datacube_version != self.datacube_version:
                logger.warning(
                    f"Requested datacube version {self.datacube_version} in config does not match discovered asset {codeocean_utils.get_datacube_dir()=}"
                )
            return codeocean_utils.get_datacube_dir() / "consolidated"
        raise ValueError(
            f"Cannot determine consolidated parquet dir: {self.datacube_version=} {self.use_scratch_dir=}"
        )


datacube_config = DatacubeConfig(
    use_scratch_dir=not (codeocean_utils.is_capsule() or codeocean_utils.is_pipeline())
)


def configure(
    use_scratch_dir: bool = False,
    datacube_version: str | None = None,
    nwb_dir: str | upath.UPath | None = None,
    consolidated_parquet_dir: str | upath.UPath | None = None,
) -> None:
    datacube_config.use_scratch_dir = use_scratch_dir
    if datacube_version:
        datacube_config.datacube_version = datacube_version
    if consolidated_parquet_dir:
        datacube_config.consolidated_parquet_dir = upath.UPath(  # type:ignore[misc]
            consolidated_parquet_dir
        )
    if nwb_dir:
        datacube_config.nwb_dir = upath.UPath(nwb_dir)  # type:ignore[misc]


def parse_version(path: str) -> str:
    """
    >>> parse_version("s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.210/366122_2023-12-31.nwb.zarr")
    'v0.0.210'
    """
    result = re.search(r"v\d+.\d+.\d+", path)
    if not result:
        raise ValueError(f"Could not parse version from path: {path!r}")
    return result.group(0)


@functools.cache
def get_datacube_version() -> str:
    return parse_version(codeocean_utils.get_datacube_dir().as_posix())


@functools.cache
def is_datacube_available() -> bool:
    with contextlib.suppress(FileNotFoundError):
        _ = codeocean_utils.get_datacube_dir()
        return True
    return False


# data access ------------------------------------------------------- #
@functools.cache
def get_session_table() -> pl.DataFrame:
    if codeocean_utils.is_pipeline() or codeocean_utils.is_capsule():
        return pl.read_parquet(
            (codeocean_utils.get_datacube_dir() / "session_table.parquet").as_posix()
        )
    return pl.read_parquet(
        "https://github.com/AllenInstitute/dynamic_routing_analysis/raw/refs/heads/main/bin/session_table.parquet"
    )


def get_df(component: str) -> pl.DataFrame:
    path = datacube_config.consolidated_parquet_dir / f"{component}.parquet"
    return pl.read_parquet(path.as_posix()).with_columns(
        # remove optional session_id prefix in case it's present:
        pl.col("session_id").str.split("_").list.slice(0, 2).list.join("_")
    )


@typing.overload
def get_nwb_paths(session_id: str) -> pathlib.Path: ...


@typing.overload
def get_nwb_paths(session_id: None) -> tuple[pathlib.Path, ...]: ...


@functools.cache
def get_nwb_paths(
    session_id: str | None = None,
) -> pathlib.Path | tuple[pathlib.Path, ...]:
    paths = datacube_config.nwb_dir.rglob("*.nwb")
    if session_id:
        try:
            return next(p for p in paths if p.stem == session_id)
        except StopIteration:
            raise FileNotFoundError(
                f"Cannot find NWB file for {session_id!r} in {datacube_config.nwb_dir}"
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
) -> "lazynwb.LazyNWB" | None:  # noqa
    import lazynwb

    nwb_path = _parse_nwb_path_from_input(
        session_id_or_path, raise_on_missing=raise_on_missing
    )
    if nwb_path is None:
        return None
    logger.info(f"Reading {nwb_path}")
    try:
        nwb = lazynwb.LazyNWB(nwb_path)
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


def combine_exprs(exprs: Iterable[pl.expr]) -> pl.expr:
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


def get_passing_session_ids(
    cross_modality_dprime: float | None = None,
    min_trials: int | None = None,
    min_contingent_rewards: int | None = None,
    of_each_rewarded_modality: bool = True,
    min_n_blocks: int = 2,
    include_templeton: bool = False,
) -> pl.Series:
    passing_block_filter = get_passing_blocks_performance_filter(
        cross_modality_dprime=cross_modality_dprime,
        min_trials=min_trials,
        min_contingent_rewards=min_contingent_rewards,
    )
    n_blocks_each_context: pl.Expr = (
        pl.col("rewarded_modality")
        .filter(passing_block_filter)
        .unique_counts()
        .over("session_id", mapping_strategy="join")
    )
    if include_templeton:
        templeton_session_ids = get_session_table().filter("is_templeton")["session_id"]
        templeton_filter = pl.col("session_id").is_in(templeton_session_ids)
    else:
        templeton_filter = pl.lit(False)
    if of_each_rewarded_modality:
        passing_session_filter = (
            # at least n_blocks of each context, and two contexts:
            (
                n_blocks_each_context.list.eval(pl.element() >= min_n_blocks).list.all()
                & (n_blocks_each_context.list.len() == 2)
            ) | templeton_filter
        )
    else:
        passing_session_filter = (
            # at least n_blocks of any context:
            (
                n_blocks_each_context.list.sum()
                >= min_n_blocks
            ) | templeton_filter
        )
    return get_df("performance").filter(passing_session_filter.explode())["session_id"].unique().sort()


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
    session_ids_by_type: pl.Series = get_df("session").filter(
        PROD_EPHYS_SESSION_FILTER, late_autorewards_expr
    )["session_id"]
    if include_templeton:
        templeton_session_ids = get_session_table().filter("is_templeton")["session_id"]
        session_ids_by_type = session_ids_by_type.append(templeton_session_ids)

    # session_ids to use based on performance:
    if by_session: 
        # keep only sessions that pass the performance criteria
        session_ids_by_performance = get_passing_session_ids(
            cross_modality_dprime=cross_modal_dprime_threshold,
            include_templeton=include_templeton,
        )
        trials = (
            get_df("trials")
            .filter(
                pl.col("session_id").is_in(
                    set(session_ids_by_type).intersection(session_ids_by_performance)
                ),
            )
        )
    else:
        if include_templeton:
            templeton_filter = pl.col("session_id").is_in(templeton_session_ids)
        else:
            templeton_filter = pl.lit(False)
        passing_blocks: pl.DataFrame = get_df("performance").filter(
            get_passing_blocks_performance_filter(cross_modality_dprime=cross_modal_dprime_threshold) | templeton_filter
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
