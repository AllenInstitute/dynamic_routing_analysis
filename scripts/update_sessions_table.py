# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dotenv",
#     "polars",
# ]
# ///
import pathlib

import polars as pl
import dotenv

dotenv.load_dotenv()

(
    pl.read_parquet(
        's3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet'
    )
    .write_parquet(
        pathlib.Path(__file__).parent.parent / 'bin/sessions_table.parquet'
    )
)
