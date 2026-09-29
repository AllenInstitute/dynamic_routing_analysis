# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "dynamic_routing_analysis"
# ]
#
# [tool.uv.sources]
# dynamic_routing_analysis = { git = "https://github.com/AllenInstitute/dynamic_routing_analysis" }
# ///

import dr_datacube
from dynamic_routing_analysis import encoding_utils, io_utils

# Build the design matrix for the selected session using the encoding pipeline settings.
# Params.__init__ parses Jupyter kernel arguments; model_construct uses class defaults instead.
params = encoding_utils.Params.model_construct(
    result_prefix=result_prefix,
    run_id=run_id,
    time_of_interest='quiescent',
    spike_bin_width=0.1,
    orthogonalize_against_context=['facial_features'],
)

project = session_table.filter(pl.col('session_id') == session_to_plot).select('project').item()
run_params = params.model_dump()
run_params.update(fullmodel_fitted=False, model_label='fullmodel', project=project)
run_params = io_utils.define_kernels(run_params)

# Use the public scratch cache; the default Code Ocean asset denies anonymous listing.
with dr_datacube.config.override(anon=True, use_cache=True):
    _, behavior_info = io_utils.get_session_data(session_to_plot, lazy=True)
    fit = io_utils.establish_timebins(run_params, {}, behavior_info)
    design = io_utils.DesignMatrix(fit)
    design, fit = io_utils.add_kernels(
        design=design,
        run_params=run_params,
        session=session_to_plot,
        fit=fit,
        behavior_info=behavior_info,
    )
    if fit['failed_kernels']:
        raise RuntimeError(f"Failed to build kernels: {sorted(fit['failed_kernels'])}")

feature_mat = design.get_X()  # dimensions: timestamps x weights
feature_mat
