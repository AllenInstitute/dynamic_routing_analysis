from __future__ import annotations

import _thread
import concurrent.futures as cf
import contextlib
import copy
import datetime
import functools
import gc
import logging
import multiprocessing
import os
import pathlib
import pickle
import tempfile
from collections.abc import Iterable
from typing import Annotated, Any, Literal

import dr_datacube
import lazynwb
import polars as pl
import pydantic
import pydantic.functional_serializers
import pydantic_settings
import tqdm
import upath

from dynamic_routing_analysis import datacube_utils, glm_utils, io_utils

logger = logging.getLogger(__name__)

# Required for serializing polars expressions
Expr = Annotated[
    pl.Expr,
    pydantic.functional_serializers.PlainSerializer(
        lambda expr: expr.meta.serialize(format="json"), return_type=str
    ),
]

test_units = [
    "664851_2023-11-15_D-218",
    "664851_2023-11-15_B-239",
    "686176_2023-12-07_B-508",
    "686176_2023-12-07_A-10",
    "686176_2023-12-07_C-195",
    "686176_2023-12-07_C-578",
    "686176_2023-12-07_C-549",
    "686176_2023-12-07_C-277",
    "686176_2023-12-07_A-252",
    "686176_2023-12-07_E-110",
    "703880_2024-04-15_E-39",
    "726088_2024-06-21_F-0",
    "726088_2024-06-21_A-56",
    "726088_2024-06-21_E-148",
    "726088_2024-06-21_F-41",
    "726088_2024-06-21_F-36",
    "726088_2024-06-21_F-55",
    "726088_2024-06-21_F-77",
    "726088_2024-06-21_F-186",
    "726088_2024-06-21_F-83",
    "726088_2024-06-21_A-8",
    "726088_2024-06-21_A-5",
    "726088_2024-06-21_A-124",
    "726088_2024-06-21_A-117",
    "726088_2024-06-21_E-39",
    "726088_2024-06-21_E-252",
    "726088_2024-06-21_E-102",
    "726088_2024-06-21_E-80",
    "726088_2024-06-21_E-99",
    "726088_2024-06-21_E-273",
    "742903_2024-10-22_E-1260",
    "742903_2024-10-22_F-193",
    "742903_2024-10-22_C-205",
]


class Params(pydantic_settings.BaseSettings, extra="allow"):
    # ----------------------------------------------------------------------------------
    # Required parameters
    result_prefix: str
    "An identifier for the decoding run, used to name the output files (can have duplicates with different run_id)"
    # ----------------------------------------------------------------------------------

    # Capsule-specific parameters -------------------------------------- #
    unit_ids_to_use: list[str] = pydantic.Field(
        default_factory=list, exclude=True, repr=True
    )
    single_session_id_to_use: str | None = pydantic.Field(None, exclude=True, repr=True)
    """If provided, only process this session_id. Otherwise, process all sessions that match the filtering criteria"""
    session_table_query: str = pydantic.Field(
        'is_ephys & is_task & is_annotated & is_production & issues=="[]"', exclude=True
    )
    run_id: str = pydantic.Field(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )  # created once at runtime: same for all Params instances
    """A unique string that should be attached to all decoding runs in the same batch"""
    skip_existing: bool = pydantic.Field(True, exclude=True)
    test: bool = pydantic.Field(False, exclude=True)
    logging_level: str | int = pydantic.Field("INFO", exclude=True)
    update_packages_from_source: bool = pydantic.Field(False, exclude=True)
    override_params_json: str | None = pydantic.Field("{}", exclude=True)
    use_process_pool: bool = pydantic.Field(True, exclude=True, repr=True)
    max_workers: int | None = pydantic.Field(os.environ.get('CO_CPUS'), exclude=True, repr=True)
    """For process pool"""
    pickle_temp_dir: pathlib.Path | None = pydantic.Field(
        None, exclude=True, repr=False
    )
    """Run-scoped directory for temporary pickle files."""

    # Run parameters that define a unique run (ie will be checked for 'skip_existing')
    datacube_version: str = dr_datacube.config.version
    time_of_interest: str = "full_trial"
    input_offsets: bool = True
    input_window_lengths: dict[str, float] = pydantic.Field(default_factory=dict)

    use_context_belief: bool = False

    # unit inclusion parameters ---------------------------------------- #
    presence_ratio: float | None = 0.7
    decoder_labels_to_exclude: list[str] = pydantic.Field(
        default_factory=lambda: ["noise"]
    )

    spike_bin_width: float = 0.1
    """in seconds"""
    areas_to_include: list[str] = pydantic.Field(default_factory=list)
    areas_to_exclude: list[str] = pydantic.Field(default_factory=list)
    orthogonalize_against_context: list[str] = pydantic.Field(
        default_factory=lambda: ["facial_features"]
    )
    trial_start_time: float = -2
    trial_stop_time: float = 3
    quiescent_start_time: float = -1.5
    quiescent_stop_time: float = 0
    intercept: bool = True
    """Whether to include an intercept in the design matrix"""
    method: Literal[
        "ridge_regression",
        "lasso_regression",
        "reduced_rank_regression",
        "elastic_net_regression",
    ] = "ridge_regression"
    no_nested_CV: bool = False
    optimize_on: float = 0.3
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    leave_blocks_out: bool = False
    optimize_penalty_by_cell: bool = False
    optimize_penalty_by_area: bool = False
    optimize_penalty_by_firing_rate: bool = False
    use_fixed_penalty: bool = False
    num_rate_clusters: int = 5

    # RIDGE + ELASTIC NET
    L2_grid_type: Literal["log", "linear"] = "log"
    L2_grid_range: list[float] = pydantic.Field(default_factory=lambda: [1, 2**12])
    L2_grid_num: int = 13
    L2_fixed_lambda: float | None = None

    # LASSO
    L1_grid_type: Literal["log", "linear"] = "log"
    L1_grid_range: list[float] = pydantic.Field(
        default_factory=lambda: [10**-6, 10**-2]
    )
    L1_grid_num: int = 13
    L1_fixed_lambda: float | None = None

    cell_regularization: float | None = None
    cell_regularization_nested: float | None = None

    # ELASTIC NET
    L1_ratio_grid_type: Literal["log", "linear"] = "log"
    L1_ratio_grid_range: list[float] = pydantic.Field(
        default_factory=lambda: [10**-6, 10**-1]
    )
    L1_ratio_grid_num: int = 9
    L1_ratio_fixed: float | None = None
    cell_L1_ratio: float | None = None
    cell_L1_ratio_nested: float | None = None

    # RRR
    rank_grid_num: int = 10
    rank_fixed: int | None = None
    cell_rank: int | None = None
    cell_rank_nested: int | None = None

    reuse_regularization_coefficients: bool = True
    """Whether to re-use regularization coefficients"""

    linear_shift_by: float = 1

    smooth_spikes_half_gaussian: bool = False
    half_gaussian_std_dev: float = 0.05
    run_dropout: bool = True
    run_linear_shift: bool = True
    features_to_drop: list[str] | None = pydantic.Field(default=None, exclude=True)

    """For modifying via app panel, but needs populating otherwise"""

    linear_shift_variables: list[str] = pydantic.Field(
        default_factory=lambda: ["context"]
    )
    """Will not work with certain feature groups (ie those with offsets)"""

    # Params that will be updated many times during processing (ie for each model) -------------- #
    # project: str | None = pydantic.Field(default=None, exclude=True)
    # drop_variables: list[str] | None = pydantic.Field(default=None, exclude=True)
    input_variables: list[str] | None = pydantic.Field(default=None, exclude=True)
    # fullmodel_fitted: bool | None = pydantic.Field(default=None, exclude=True)
    # model_label: str | None  = pydantic.Field(default=None, exclude=True)

    @property
    def results_dir(self) -> upath.UPath:
        """Path to general encoding results dir on S3"""
        return upath.UPath("s3://aind-scratch-data/dynamic-routing/encoding/results")

    @property
    def results_folder_name(self) -> str:
        """Name of the results folder on S3, including the run_id"""
        return f"{self.result_prefix}_{self.run_id}"

    @property
    def pkl_data_dir(self) -> upath.UPath:
        """Path to pkl data subfolder within results dir on S3"""
        return self.results_dir.parent / f"fullmodel_pkl_files/{self.results_folder_name}"

    @property
    def json_path(self) -> upath.UPath:
        """Path to params json on S3"""
        return self.results_dir / f"{self.results_folder_name}.json"

    @pydantic.computed_field(repr=False)
    @property
    def unit_inclusion_criteria(self) -> Expr:
        exprs = []
        exprs.append(pl.col("decoder_label") != "noise")
        if self.presence_ratio is not None:
            exprs.append(pl.col("presence_ratio") >= self.presence_ratio)
        if self.areas_to_include:
            exprs.append(pl.col("structure").is_in(self.areas_to_include))
        if self.areas_to_exclude:
            exprs.append(pl.col("structure").is_in(self.areas_to_exclude).not_())
        if self.unit_ids_to_use:
            exprs.append(pl.col("unit_id").is_in(self.unit_ids_to_use))
        return pl.Expr.and_(*exprs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources below is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(
                settings_cls, json_file="parameters.json"
            ),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )


# ------------------------------------------------------------------ #


def regularization_coefficients_path(session_id: str, params: Params) -> pathlib.Path:
    if params.pickle_temp_dir is None:
        raise RuntimeError("A temporary pickle directory has not been configured")
    return params.pickle_temp_dir / f"{session_id}_regularization_coefficients.pkl"


def get_regularization_coefficients(session_id: str, params: Params) -> dict[str, Any]:
    if regularization_coefficients_path(session_id, params).exists():
        return pickle.loads(
            regularization_coefficients_path(session_id, params).read_bytes()
        )

    elif get_s3_fullmodel_result_pickle_path(session_id, params).exists():
        fit = pickle.loads(
            get_s3_fullmodel_result_pickle_path(session_id, params).read_bytes()
        )["fit"]
        return get_regularization_coef_dict_from_fit(fit)
    else:
        raise FileNotFoundError(
            f"Could not find saved files (local or on S3) containing regularization coefficients for session {session_id}"
        )


def get_regularization_coef_dict_from_fit(fit):
    regularization_coef_dict = {}
    for prefix in ["cell_regularization", "cell_L1_ratio", "cell_rank"]:
        for suffix in ["", "_nested"]:
            key = f"{prefix}{suffix}"
            regularization_coef_dict[key] = fit[key]
    return regularization_coef_dict


def local_fullmodel_data_path(session_id: str, params: Params) -> pathlib.Path:
    if params.pickle_temp_dir is None:
        raise RuntimeError("A temporary pickle directory has not been configured")
    return params.pickle_temp_dir / f"{session_id}.pkl"


def get_local_fullmodel_data(
    session_id: str, params: Params, lock: _thread.LockType | None = None
) -> dict[str, dict]:
    """Equivalent to reading the 'inputs.npz' file in the pipeline.
    If it doesn't exist, it will be created.
    """
    if local_fullmodel_data_path(session_id, params).exists():
        data = pickle.loads(local_fullmodel_data_path(session_id, params).read_bytes())
        return data
    else:
        # doesn't exist, so create it
        data = generate_fullmodel_data(session_id, params, lock=lock)

        # and save it
        local_fullmodel_data_path(session_id, params).write_bytes(pickle.dumps(data))
        if dr_datacube.on_codeocean() and params.test:
            pathlib.Path(
                local_fullmodel_data_path(session_id)
                .as_posix()
                .replace("scratch", "results")
            ).write_bytes(pickle.dumps(data))
        return data


def generate_fullmodel_data(session_id, params, lock: _thread.LockType | None = None):
    print(f"{session_id} | getting units and behavior info via DRA")
    if lock is None:
        lock = contextlib.nullcontext()  # type: ignore[assignment]
    assert lock is not None

    print(f"{session_id} | acquiring lock to get units and behavior info")
    with lock:
        lazy = True
        lazy_units, behavior_info = io_utils.get_session_data(
            session_id, lazy=lazy,
            scan_nwb_kwargs=dict(low_memory=True),
            get_df_kwargs=dict(parallel=not params.use_process_pool),
        )
        print("getting units table with lazynwb")
        if lazy:
            units_table = (
                lazy_units
                .filter(params.unit_inclusion_criteria)
                .select("unit_id", "spike_times", "obs_intervals")
                .collect()
            )
        else:
            units_table = (
                lazy_units
                .filter(params.unit_inclusion_criteria)
                .pipe(lazynwb.merge_array_column, 'spike_times')
                .pipe(lazynwb.merge_array_column, 'obs_intervals')
                .select("unit_id", "spike_times", "obs_intervals")
            )
        print(units_table.columns)
        print(f"{session_id} | converting units_table to pandas")
        units_table = units_table.to_pandas()
    if len(units_table) == 0:
        raise ValueError("No units meet the inclusion criteria — units_table is empty.")
    run_params = params.model_dump()
    if params.input_variables:
        run_params |= {
          "input_variables": params.input_variables
        }
    run_params |= {
        "fullmodel_fitted": False,
        "model_label": "fullmodel",
        "project": get_project(session_id),
    }
    run_params = io_utils.define_kernels(run_params)

    fit: dict[str, Any] = {}
    fit = io_utils.establish_timebins(
        run_params=run_params, fit=fit, behavior_info=behavior_info
    )
    fit = io_utils.process_spikes(
        units_table=units_table, run_params=run_params, fit=fit
    )
    del units_table
    gc.collect()

    design: io_utils.DesignMatrix = io_utils.DesignMatrix(fit)
    design, fit = io_utils.add_kernels(
        design=design,
        run_params=run_params,
        session=session_id,
        fit=fit,
        behavior_info=behavior_info,
    )

    design_matrix = design.get_X()
    return {"fit": fit, "design_matrix": design_matrix, "run_params": run_params}


def get_s3_fullmodel_result_pickle_path(session_id: str, params: Params):
    return params.pkl_data_dir / f"{session_id}.pkl"


def get_project(session_id: str) -> str:
    return datacube_utils.get_session_table().filter(pl.col("session_id") == session_id)[
        "project"
    ][0]


def helper_fullmodel(
    session_id: str, params: Params, lock: _thread.LockType | None = None
) -> None:
    model_label = "fullmodel"
    if (
        params.skip_existing
        and get_parquet_path(
            session_id=session_id, params=params, model_label=model_label
        ).exists()
    ):
        print(f"{session_id} | skipping fullmodel helper")
        return
    print(f"{session_id} | running fullmodel helper")
    data = get_local_fullmodel_data(session_id=session_id, params=params, lock=lock)
    run_params = data["run_params"]
    run_params |= {
        "fullmodel_fitted": False,
        "model_label": model_label,
    }
    fit = glm_utils.optimize_model(
        fit=data["fit"], design_mat=data["design_matrix"], run_params=run_params
    )
    fit = glm_utils.evaluate_model(
        fit=fit, design_mat=data["design_matrix"], run_params=run_params
    )
    regularization_coefficients_path(session_id, params).write_bytes(
        pickle.dumps(get_regularization_coef_dict_from_fit(fit))
    )
    save_results(session_id=session_id, fit=fit, run_params=run_params, params=params)


def get_features_to_drop(session_id: str, params: Params) -> list[str]:
    if params.features_to_drop:
        return params.features_to_drop
    run_params = get_fullmodel_pkl_local_or_s3(session_id, params)["run_params"]
    features_to_drop = list(run_params["input_variables"]) + [
        run_params["kernels"][key]["function_call"]
        for key in run_params["input_variables"]
    ]
    if "context_templeton" in features_to_drop:
        features_to_drop.remove("context_templeton")
    return sorted(set(features_to_drop))


def helper_dropout(
    session_id: str,
    params: Params,
    feature_to_drop: str,
    lock: _thread.LockType | None = None,
) -> None:
    model_label = f"drop_{feature_to_drop}"
    if (
        params.skip_existing
        and get_parquet_path(
            session_id=session_id, params=params, model_label=model_label
        ).exists()
    ):
        print(f"{session_id} | skipping dropout helper {feature_to_drop}")
        return
    data = get_local_fullmodel_data(session_id=session_id, params=params, lock=lock)
    run_params = data["run_params"]
    run_params |= {
        "drop_variables": [feature_to_drop],
        "fullmodel_fitted": params.reuse_regularization_coefficients,
        "model_label": model_label,
    }
    run_params = io_utils.define_kernels(run_params)
    fit = glm_utils.dropout(
        fit=data["fit"] | get_regularization_coefficients(session_id, params),
        design_mat=data["design_matrix"],
        run_params=run_params,
    )
    save_results(session_id=session_id, fit=fit, run_params=run_params, params=params)


def helper_linear_shift(
    session_id: str,
    params: Params,
    shift: int,
    blocks: Iterable[int],
    shift_columns: list[int],
    lock: _thread.LockType | None = None,
) -> None:
    model_label = f"shift_{shift}"
    if (
        params.skip_existing
        and get_parquet_path(
            session_id=session_id, params=params, model_label=model_label
        ).exists()
    ):
        print(f"{session_id} | skipping shift helper {shift}")
        return
    data = get_local_fullmodel_data(session_id=session_id, params=params, lock=lock)
    run_params = data["run_params"]
    run_params |= {
        "fullmodel_fitted": params.reuse_regularization_coefficients,
        "model_label": model_label,
    }
    fit = glm_utils.apply_shift_to_design_matrix(
        fit=data["fit"] | get_regularization_coefficients(session_id, params),
        design_mat=data["design_matrix"],
        run_params=run_params,
        blocks=blocks,
        shift_columns=shift_columns,
        shift=shift,
    )
    save_results(session_id=session_id, fit=fit, run_params=run_params, params=params)


def get_fullmodel_pkl_local_or_s3(session_id: str, params: Params) -> dict[str, dict]:
    """Watch out: returned dict does not design matrix if from S3, and fit does not contain results
    if from local"""
    if local_fullmodel_data_path(session_id, params).exists():  # fit does not contain results
        data = get_local_fullmodel_data(session_id=session_id, params=params)
    elif get_s3_fullmodel_result_pickle_path(
        session_id, params
    ).exists():  # data does not contain design matrix
        data = pickle.loads(
            get_s3_fullmodel_result_pickle_path(session_id, params).read_bytes()
        )
    else:
        raise FileNotFoundError(
            f"Could not find saved files (local or on S3) containing fit/run_params for session {session_id}"
        )
    return data


def get_shift_columns(session_id: str, params: Params) -> list[int]:

    data = get_fullmodel_pkl_local_or_s3(session_id=session_id, params=params)
    if "design_matrix" in data:
        weight_labels = data["design_matrix"].coords["weights"].data  # type: ignore
    else:
        weight_labels = data["fit"]["fullmodel"]["weight_label"]
    return [
        i
        for i, label in enumerate(weight_labels)  # type: ignore
        if any([key in label for key in params.linear_shift_variables])
    ]


def get_linear_shifts(
    session_id: str, params: Params
) -> tuple[Iterable[int], Iterable[int]]:
    data = get_fullmodel_pkl_local_or_s3(session_id=session_id, params=params)
    if "design_matrix" in data:
        context = data["design_matrix"].sel(weights="context_0").data  # type: ignore
    else:
        # make a spoofed design matrix to get the context regressor (but without binned spike counts)
        run_params_for_context = copy.deepcopy(data["run_params"])
        run_params_for_context["input_variables"] = ["context"]
        run_params_for_context["intercept"] = False
        run_params_for_context = io_utils.define_kernels(run_params_for_context)

        design_matrix_for_context = io_utils.DesignMatrix(data["fit"])

        design_matrix_for_context, _ = io_utils.add_kernels(
            design=design_matrix_for_context,
            run_params=run_params_for_context,
            session=session_id,
            fit=data["fit"],
            behavior_info=io_utils.get_session_data(session_id)[1],
        )
        context = design_matrix_for_context.get_X().data.flatten()
    return glm_utils.get_shift_bins(
        run_params=params.model_dump(),  # type: ignore
        fit=data["fit"],
        context=context,
    )


def get_parquet_path(session_id: str, params: Params, model_label: str) -> upath.UPath:
    return (
        params.results_dir
        / params.results_folder_name
        / f"{session_id}_{model_label}.parquet"
    )


def save_results(
    session_id: str, fit: dict[str, Any], params: Params, run_params: dict[str, Any]
) -> None:
    fit["spike_count_arr"].pop("spike_counts", None)
    if run_params["model_label"] == "fullmodel":
        pkl_path = get_s3_fullmodel_result_pickle_path(session_id, params)
        logger.info(f"Writing fullmodel data to {pkl_path}")
        pkl_path.write_bytes(pickle.dumps({"fit": fit, "run_params": run_params}))

    model_label = run_params["model_label"]
    if model_label == "fullmodel":
        dropped_variable = None
        shift_index = None

    elif model_label.startswith("drop_"):
        dropped_variable = "_".join(model_label.split("_")[1:])
        shift_index = None

    elif model_label.startswith("shift_"):
        dropped_variable = None
        shift_index = int(model_label.split("_")[-1])
    else:
        raise ValueError(f"Unrecognized model label: {model_label}")

    # save some contents of fit as parquet on S3
    parquet_path = get_parquet_path(
        session_id=session_id,
        params=params,
        model_label=run_params["model_label"],
    )
    logger.info(f"Writing results to {parquet_path}")
    (
        pl.DataFrame(
            {
                "session_id": session_id,
                "unit_id": fit["spike_count_arr"]["unit_id"].tolist(),
                "project": get_project(session_id),
                "cv_test": fit[run_params["model_label"]]["cv_var_test"],
                "cv_train": fit[run_params["model_label"]]["cv_var_train"],
                "weights": (
                    None
                    if run_params["model_label"].startswith("shift")
                    else fit[run_params["model_label"]]["weights"].T.tolist()
                ),
                "dropped_variable": dropped_variable,
                "shift_index": shift_index,
                "model_label": model_label.split("_")[0],
            },
            schema_overrides={
                "shift_index": pl.Int32,
                "dropped_variable": pl.String,
                "weights": pl.List(pl.Float64),
                "model_label": pl.Enum(['fullmodel', 'drop', 'shift']),
                "project": pl.Enum(['DynamicRouting', 'Templeton']),
                "dropped_variable": pl.Enum(['running', 'miss', 'sound1', 'context', 'vis2', 'licks', 'session_time', 'correct_reject', 'nose', 'stimulus', 'jaw', 'false_alarm', 'ears', 'choice', 'whisker_pad', 'sound2', 'facial_features', 'pupil', 'vis1', 'hit']),
            }
        ).write_parquet(
            parquet_path.as_posix(),
            compression_level=18,
            statistics="full",
        )
    )


def run_after_full_model(
    session_id: str, params: Params, lock: _thread.LockType | None = None
) -> None:
    if params.run_dropout:
        print(f"{session_id} | running drop features after full model")
        for feature_to_drop in get_features_to_drop(session_id=session_id, params=params):
            print(f"{session_id} | {feature_to_drop=}")
            helper_dropout(
                session_id=session_id,
                params=params,
                feature_to_drop=feature_to_drop,
                lock=lock,
            )
            if params.test:
                logger.info("Test mode: exiting after first feature dropout")
                break
    else:
        print(f"{session_id} | skipping dropout")

    if params.run_linear_shift:
        print(f"{session_id} | running linear shift")

        good_behavior_sessions = dr_datacube.get_session_ids_from_github(
            session_type=["brainwide", "templeton"],
            with_behavior_filter=True,
        )

        if session_id not in good_behavior_sessions:
            return

        shifts, blocks = get_linear_shifts(session_id=session_id, params=params)
        for shift in shifts:
            print(f"{session_id} | {shift=}")
            helper_linear_shift(
                session_id=session_id,
                params=params,
                shift=shift,
                blocks=blocks,
                shift_columns=get_shift_columns(session_id=session_id, params=params),
                lock=lock,
            )
            if params.test:
                logger.info("Test mode: exiting after first shift")
                break
    else:
        print(f"{session_id} | skipping linear shift")

def schedule_more_jobs(
    future: cf.Future,
    session_id: str,
    params: Params,
    executor: cf.ProcessPoolExecutor,
    lock: _thread.LockType | None = None,
) -> None:
    #! don't do closure on session_id/params
    _ = future  # future must be first arg, but isn't needed
    print(f'{session_id} | submitting more jobs')
    f = executor.submit(
        run_after_full_model,
        session_id=session_id,
        params=params,
        #lock=lock,
    )

def _run_encoding(
    session_ids: str | Iterable[str],
    params: Params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    if not params.use_process_pool:
        for session_id in tqdm.tqdm(
            session_ids,
            total=len(tuple(session_ids)),
            unit="session",
            desc="Encoding",
        ):
            logger.info(f"Processing session {session_id}")
            helper_fullmodel(session_id=session_id, params=params)
            run_after_full_model(session_id=session_id, params=params)

    else:
        future_to_session: dict[cf.Future, str] = {}
        lock =  None # multiprocessing.Manager().Lock()  # or None
        with cf.ProcessPoolExecutor(
            max_workers=params.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for session_id in session_ids:
                future = executor.submit(
                    helper_fullmodel,
                    session_id=session_id,
                    params=params,
                    #lock=lock,
                )


                future.add_done_callback(
                    functools.partial(
                        schedule_more_jobs,
                        session_id=session_id,
                        params=params,
                        executor=executor,
                    )
                )

                future_to_session[future] = session_id
                logger.debug(
                    f"Submitted encoding to process pool for session {session_id}"
                )
                if params.test:
                    logger.info("Test mode: exiting after first session")
                    break
            for future in tqdm.tqdm(
                cf.as_completed(future_to_session),
                total=len(future_to_session),
                unit="session",
                desc="Encoding",
            ):
                session_id = future_to_session[future]
                try:
                    _ = future.result()
                except Exception:
                    logger.exception(f"{session_id} | Failed:")


def run_encoding(
    session_ids: str | Iterable[str],
    params: Params,
) -> None:
    with tempfile.TemporaryDirectory(prefix="dynamic-routing-encoding-") as temp_dir:
        params = params.model_copy(
            update={"pickle_temp_dir": pathlib.Path(temp_dir)}
        )
        _run_encoding(session_ids=session_ids, params=params)
