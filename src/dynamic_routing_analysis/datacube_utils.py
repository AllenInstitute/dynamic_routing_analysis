# stdlib imports --------------------------------------------------- #
from __future__ import annotations

import dataclasses
import functools
import logging
import logging.handlers
import pathlib
import re
from collections.abc import Iterable

# 3rd-party imports necessary for processing ----------------------- #
import npc_lims
import polars as pl
import pynwb
import upath

import dynamic_routing_analysis.codeocean_utils as codeocean_utils

logger = logging.getLogger(__name__)


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
                self.nwb_dir.parent
                / f"nwb_components/{self.datacube_version}/consolidated"
            )
        if codeocean_utils.is_capsule() or codeocean_utils.is_pipeline():
            datacube_version = parse_version(codeocean_utils.get_datacube_dir().as_posix())
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


# data access ------------------------------------------------------- #
@functools.cache
def get_session_table() -> pl.DataFrame:
    if codeocean_utils.is_pipeline() or codeocean_utils.is_capsule():
        return pl.read_parquet(
            codeocean_utils.get_datacube_dir() / "session_table.parquet"
        )
    return pl.read_parquet(
        "https://github.com/AllenInstitute/dynamic_routing_analysis/raw/refs/heads/main/bin/session_table.parquet"
    )


def get_df(component: str) -> pl.DataFrame:
    path = datacube_config.consolidated_parquet_dir / f"{component}.parquet"
    return pl.read_parquet(path).with_columns(
        pl.col("session_id").str.split("_").list.slice(0, 2).list.join("_")
    )


@functools.cache
def get_nwb_paths() -> tuple[pathlib.Path, ...]:
    return tuple(datacube_config.nwb_dir.rglob("*.nwb"))


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
) -> pynwb.NWBFile | None:
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
) -> lazynwb.LazyNWB | None:
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
