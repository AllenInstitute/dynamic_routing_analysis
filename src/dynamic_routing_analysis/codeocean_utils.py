# stdlib imports --------------------------------------------------- #
from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import pathlib
import uuid
from collections.abc import Iterable

# 3rd-party imports necessary for processing ----------------------- #

logger = logging.getLogger(__name__)

CO_COMPUTATION_ID = os.environ.get("CO_COMPUTATION_ID")
AWS_BATCH_JOB_ID = os.environ.get("AWS_BATCH_JOB_ID")


def is_capsule() -> bool:
    return bool(CO_COMPUTATION_ID)


def is_pipeline() -> bool:
    return bool(AWS_BATCH_JOB_ID)


@functools.cache
def get_data_root() -> pathlib.Path:
    expected_paths = (
        "/data",
        "/tmp/data",
    )
    for p in expected_paths:
        if (data_root := pathlib.Path(p)).exists():
            logger.debug(f"Using {data_root=}")
        return data_root
    else:
        raise FileNotFoundError(f"data dir not present at any of {expected_paths=}")


@functools.cache
def get_code_root() -> pathlib.Path:
    expected_paths = (
        "/code",
        "/tmp/code",
    )
    for p in expected_paths:
        if (code_root := pathlib.Path(p)).exists():
            logger.debug(f"Using {code_root=}")
        return code_root
    else:
        raise FileNotFoundError(f"code dir not present at any of {expected_paths=}")


# paths ----------------------------------------------------------- #
@functools.cache
def get_datacube_dir() -> pathlib.Path:
    for p in sorted(
        get_data_root().iterdir(), reverse=True
    ):  # in case we have multiple assets attached, the latest will be used
        if p.is_dir() and p.name.startswith("dynamicrouting_datacube"):
            path = p
            break
    else:
        for p in get_data_root().iterdir():
            if any(
                pattern in p.name
                for pattern in (
                    "session_table",
                    "nwb",
                    "consolidated",
                )
            ):
                path = get_data_root()
                break
        else:
            raise FileNotFoundError(
                f"Cannot determine datacube dir: {list(get_data_root().iterdir())=}"
            )
    logger.debug(f"Using files in {path}")
    return path


def ensure_nonempty_results_dirs(dirs: str | Iterable[str] = "/results") -> None:
    """A pipeline run can crash if a results folder is expected and not found or is empty
    - ensure that a non-empty folder exists by creating a unique file"""
    if not is_pipeline():
        return
    if isinstance(dirs, str):
        dirs = (dirs,)
    for d in dirs:
        results_dir = pathlib.Path(d)
        results_dir.mkdir(exist_ok=True)
        if not list(results_dir.iterdir()):
            path = results_dir / uuid.uuid4().hex
            logger.info(f"Creating {path} to ensure results folder is not empty")
            path.touch()


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(
            doctest.NORMALIZE_WHITESPACE
            | doctest.ELLIPSIS
            | doctest.IGNORE_EXCEPTION_DETAIL
        ),
    )
