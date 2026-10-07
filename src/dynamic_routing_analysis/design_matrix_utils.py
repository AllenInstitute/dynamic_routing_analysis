"""Build encoding-model design matrices for individual sessions."""

from __future__ import annotations

from collections.abc import Sequence

import dr_datacube
import xarray as xr

from dynamic_routing_analysis import encoding_utils, io_utils

def _as_list(values: Sequence[str] | str | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return list(values)


def build_design_matrix(
    session_id: str,
    *,
    time_of_interest: str = "quiescent",
    input_variables: Sequence[str] | str | None = None,
    orthogonalize_against: Sequence[str] | str | None = ("facial_features",),
    spike_bin_width: float = 0.1,
    project: str = "DynamicRouting",
    use_cache: bool = True,
    anonymous: bool = True,
) -> xr.DataArray:
    """Return a session's design matrix with dimensions ``timestamps`` × ``weights``.

    ``input_variables=None`` uses the defaults for ``time_of_interest`` from
    :func:`io_utils.define_kernels`. Group names such as ``facial_features``
    expand to their component kernels. ``orthogonalize_against`` names the
    kernels to orthogonalize against context; pass ``None`` to disable it.

    The function reads session behavior and kernel inputs but does not load or
    process unit spikes. By default it uses the anonymous datacube cache.
    """
    session_id = io_utils.extract_session_id(session_id)
    if time_of_interest not in {
        "full", "full_trial", "trial", "quiescent", "spontaneous"
    }:
        raise ValueError(f"Unsupported time_of_interest: {time_of_interest!r}")
    if spike_bin_width <= 0:
        raise ValueError("spike_bin_width must be positive")

    # Params.__init__ parses command-line arguments, including Jupyter's -f.
    # model_construct supplies the class defaults without that CLI parsing.
    requested_inputs = _as_list(input_variables)
    requested_orthogonalization = _as_list(orthogonalize_against) or []
    params = encoding_utils.Params.model_construct(
        result_prefix="design_matrix",
        time_of_interest=time_of_interest,
        input_variables=requested_inputs,
        orthogonalize_against_context=requested_orthogonalization,
        spike_bin_width=spike_bin_width,
    )
    run_params = params.model_dump()
    if requested_inputs is not None:
        # Params excludes input_variables from model_dump().
        run_params["input_variables"] = requested_inputs
    run_params.update(fullmodel_fitted=False, model_label="fullmodel", project=project)
    run_params = io_utils.define_kernels(run_params)

    with dr_datacube.config.override(anon=anonymous, use_cache=use_cache):
        _, behavior_info = io_utils.get_session_data(session_id, lazy=True)
        fit = io_utils.establish_timebins(run_params, {}, behavior_info)
        design = io_utils.DesignMatrix(fit)
        design, fit = io_utils.add_kernels(
            design=design,
            run_params=run_params,
            session=session_id,
            fit=fit,
            behavior_info=behavior_info,
        )

    if fit["failed_kernels"]:
        details = {name: fit["kernel_error_dict"][name] for name in fit["failed_kernels"]}
        raise RuntimeError(f"Could not build kernels for {session_id}: {details}")

    matrix = design.get_X()
    matrix.attrs.update(
        session_id=session_id,
        time_of_interest=time_of_interest,
        input_variables=run_params["input_variables"],
        spike_bin_width=spike_bin_width,
        orthogonalize_against=requested_orthogonalization,
    )
    return matrix
