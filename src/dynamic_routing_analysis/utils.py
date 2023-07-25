import concurrent.futures
import pathlib
from typing import Any, Callable, Generator

import nwbwidgets
import pandas as pd
import pynwb

process_pool = concurrent.futures.ProcessPoolExecutor()

all_nwb_paths = pathlib.Path('/data/nwb/nwb').glob('*.nwb')

PATHS: tuple[pathlib.Path, ...] = tuple(path for path in all_nwb_paths if path.stat().st_size > 1024 ** 2)
SESSIONS: tuple[str, ...] = tuple(path.stem for path in PATHS)
SESSION_TO_PATH: dict[str, pathlib.Path] = dict(zip(SESSIONS, PATHS))

def get_nwb(session: str) -> pynwb.NWBHDF5IO:
    return pynwb.NWBHDF5IO(SESSION_TO_PATH[session], mode='r', load_namespaces=True).read()

def nwbs() -> Generator[pynwb.NWBHDF5IO, None, None]:
    for session in SESSIONS:
        yield get_nwb(session)

def apply_to_all(func: Callable[[pynwb.NWBHDF5IO, Any], Any], *args, **kwargs) -> tuple[Any, ...]:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return tuple(executor.map(func, nwbs(), *args, **kwargs))
        
        

        
