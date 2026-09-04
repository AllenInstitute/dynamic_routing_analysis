# Session I/O characterization references

These files capture observable values produced from NWB cache version `0.0.289`.
They are regression references for the session-data access refactor, not claims
that every legacy behavior is scientifically correct.

The sessions were selected to cover different shapes of data:

- `626791_2022-08-15`: DynamicRouting task session with six performance rows,
  video, eye tracking, running, and lick events.
- `620263_2022-07-26`: Templeton session with a single performance row and a
  different behavioral structure.

For both sessions, the references contain every currently returned row and
column from units, trials, epochs, and cross-modality d-prime. Transport-only
columns (`_nwb_path`, `_table_path`, and `_table_index`) are intentionally
excluded. Adding new columns is allowed, but removing or changing an existing
column is not.

The primary session also contains full-length references for:

- running-speed kernel input;
- pupil kernel input;
- nose-motion kernel input (representative of facial-feature processing); and
- an end-to-end design matrix containing `vis1`, `hit`, `context`, and
  `session_time`, including timestamps and weight labels.

Floating-point values use `rtol=1e-7` and `atol=1e-7`. Integers, booleans,
strings, ordering, shapes, labels, and null placement are exact.

The legacy lick kernel is not recorded because the pinned NWB stores lick
events as timestamps without a `data` array, while the current kernel requires
`timeseries.data`. The legacy `input_offsets=True` path is also omitted because
`establish_timebins` currently fails its mask-length assertion for the primary
session. These are known gaps, not approved baseline behavior.

The tests use anonymous, read-only access to the pinned public S3 objects and
are opt-in:

```shell
RUN_IO_CHARACTERIZATION=1 uv run --group testing pytest tests/test_io_characterization.py
```

Normal `pytest` runs skip them and do not access the network.

Only replace the references after reviewing an intentional output change:

```shell
uv run python tests/record_io_characterization.py --record
```
