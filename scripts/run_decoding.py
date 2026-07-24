from __future__ import annotations

import os
os.environ['RUST_BACKTRACE'] = '1'
#os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'


# stdlib imports --------------------------------------------------- #
import json
import logging
import pathlib
import time
from typing import Annotated, Literal

# 3rd-party imports necessary for processing ----------------------- #
import lazynwb
import matplotlib
import polars as pl
import upath

lazynwb.config.anon = True

# local modules ---------------------------------------------------- #
from dynamic_routing_analysis import (
    codeocean_utils,
    datacube_utils,
    decoding_utils,
    utils,
)
from dynamic_routing_analysis.decoding_utils import Params


# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem if __name__.endswith("_main__") else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux

        
# processing function ---------------------------------------------- #

if not codeocean_utils.on_code_ocean():
    # datacube_utils.configure(datacube_version='v0.0.272', use_scratch_dir=True)
    datacube_utils.configure(datacube_version='v0.0.288', use_scratch_dir=True)

def main():
    t0 = time.time()

    if codeocean_utils.on_code_ocean():
        utils.setup_logging()
    else:
        logging.basicConfig(level=logging.INFO)
    params = Params() # reads from CLI args
    logger.setLevel(params.logging_level)
  
    if params.override_params_json:
        logger.info(f"Overriding parameters with {params.override_params_json}")
        params = Params(**json.loads(params.override_params_json))
        
    if params.test:
        params = Params(
            result_prefix=f"test/{params.result_prefix}",
            min_n_units=20,
            n_repeats=1,
            trials_filter=pl.col('is_aud_stim'),
        )
        logger.info("Test mode: using modified set of parameters")
        
    
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:
    #TODO session table is not versioned - current content represents naive datacube
    session_table = datacube_utils.get_session_table().to_pandas()
    session_table['issues']=session_table['issues'].astype(str)
    session_ids: list[str] = session_table.query(params.session_table_query)['session_id'].values.tolist()
    logger.info(f"Found {len(session_ids)} session_ids after filtering session table")
    
    if params.session_id is not None:
        if params.session_id not in session_ids:
            logger.warning(f"{params.session_id!r} not in filtered session_ids")
        logger.info(f"Using single session_id {params.session_id} provided via command line argument")
        session_ids = [params.session_id]
    elif utils.is_pipeline(): 
        # only one nwb will be available 
        session_ids = set(session_ids) & set(p.stem for p in utils.get_nwb_paths())
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids (matching filter and available for use)")

    if codeocean_utils.on_code_ocean():
        upath.UPath('/results/params.json').write_text(params.model_dump_json(indent=4))
    if params.json_path.exists():
        existing_params = json.loads(params.json_path.read_text())
        if existing_params != params.model_dump():
            raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}")
    else:            
        logger.info(f'Writing params file: {params.json_path}')
        params.json_path.parent.mkdir(parents=True, exist_ok=True)
        params.json_path.write_text(params.model_dump_json(indent=4))
    
    logger.info(f'starting decode_context_with_linear_shift with {params!r}')
    decoding_utils.decode_context_with_linear_shift(session_ids=session_ids, params=params)
    
    utils.ensure_nonempty_results_dir()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
