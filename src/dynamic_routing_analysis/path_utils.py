import upath

ANALYSIS_ROOT_PATH = upath.UPath("s3://aind-scratch-data/dynamic-routing/ethan")
DECODING_ROOTH_PATH: upath.UPath = ANALYSIS_ROOT_PATH / "decoding-results"
SINGLE_UNIT_METRICS_PATH: upath.UPath = ANALYSIS_ROOT_PATH / "single-unit-metrics"