from __future__ import annotations

import functools
import logging
import tempfile
import time
from typing import Iterable, Literal, TypeVar

import nrrd
import numba
import numpy as np
import numpy.typing as npt
import polars as pl
import upath

logger = logging.getLogger(__name__)

RESOLUTION_UM = 25
AXIS_TO_DIM = {"ml": 0, "dv": 1, "ap": 2}
PROJECTION_TO_AXIS = {"sagittal": "ml", "coronal": "ap", "horizontal": "dv"}
PROJECTION_YX = {
    "sagittal": ("dv", "ap"),
    "coronal": ("dv", "ml"),
    "horizontal": ("ap", "ml"),
}


@functools.cache
def get_ccf_volume(left_hemisphere=True, right_hemisphere=False) -> npt.NDArray:
    """
    array[ap, ml, dv]

    >>> volume = get_ccf_volume()
    >>> assert volume.any()
    """

    local_path = upath.UPath(
        f"//allen/programs/mindscope/workgroups/np-behavior/annotation_{RESOLUTION_UM}.nrrd"
    )
    cloud_path = upath.UPath(
        f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_{RESOLUTION_UM}.nrrd"
    )

    path = local_path if local_path.exists() else cloud_path
    if path.suffix not in (supported := (".nrrd",)):
        raise ValueError(
            f"{path.suffix} files not supported, must be one of {supported}"
        )
    if path.protocol:  # cloud path - download it
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = upath.UPath(tempdir) / path.name
            logger.warning(f"Downloading CCF volume to temporary file {temp_path.as_posix()}")
            temp_path.write_bytes(path.read_bytes())
            path = temp_path
            volume, _ = nrrd.read(path, index_order="C")  # ml, dv, ap
    else:
        logger.info(f"Using CCF volume from {path.as_posix()}")
        volume, _ = nrrd.read(path, index_order="C")  # ml, dv, ap
    ml_dim = AXIS_TO_DIM["ml"]
    dims = [
        (slice(0, volume.shape[ml_dim] // 2) if dim == ml_dim else slice(None))
        for dim in range(3)
    ]

    if left_hemisphere and not right_hemisphere:
        return volume[dims[0], dims[1], dims[2]]
    if right_hemisphere and not left_hemisphere:
        r_dims = [
            (
                slice(volume.shape[ml_dim] // 2, volume.shape[ml_dim])
                if dim == ml_dim
                else slice(None)
            )
            for dim in range(3)
        ]
        return volume[dims[0], dims[1], dims[2]]
    return volume


@functools.cache
def get_midline_ccf_ml() -> float:
    return (
        RESOLUTION_UM
        * 0.5
        * get_ccf_volume(
            left_hemisphere=True,
            right_hemisphere=True,
        ).shape[AXIS_TO_DIM["ml"]]
    )


def ccf_to_volume_index(coord: float) -> int:
    return round(coord / RESOLUTION_UM)


@functools.cache
def get_ccf_structure_tree_df() -> pl.DataFrame:
    t0 = time.time()
    path = "https://raw.githubusercontent.com/cortex-lab/allenCCF/master/structure_tree_safe_2017.csv"
    logging.info(f"Using CCF structure tree from {path}")
    df = pl.read_csv(path)
    len_0 = len(df)
    df = (
        df.with_columns(
            color_hex_int=pl.col("color_hex_triplet").str.to_integer(base=16),
            color_hex_str=pl.lit("#") + pl.col("color_hex_triplet"),
        )
        .with_columns(
            r=pl.col("color_hex_triplet")
            .str.slice(0, 2)
            .str.to_integer(base=16)
            .mul(1 / 255),
            g=pl.col("color_hex_triplet")
            .str.slice(2, 2)
            .str.to_integer(base=16)
            .mul(1 / 255),
            b=pl.col("color_hex_triplet")
            .str.slice(4, 2)
            .str.to_integer(base=16)
            .mul(1 / 255),
        )
        .with_columns(
            color_rgb=pl.concat_list("r", "g", "b"),
        )
        .drop("r", "g", "b")
        .with_columns(
            parent_ids=pl.col("structure_id_path")
            .str.split("/")
            .cast(pl.List(int), strict=False)
            .list.drop_nulls()
            .list.slice(offset=0, length=pl.col("depth")),
        )
    )
    df = df.join(
        other=(
            df.explode(pl.col("parent_ids"))
            .group_by(pl.col("parent_ids").alias("id"), maintain_order=True)
            .agg(pl.col("id").alias("child_ids"))
        ),
        on="id",
        how="left",
    ).with_columns(
        pl.col("child_ids").fill_null([]),
        is_deepest=~pl.col("id").is_in(df["parent_structure_id"]),
    )
    assert not any(df.filter(pl.col("is_deepest"))["child_ids"].to_list())
    # add list of deepest children for each area
    df = df.join(
        other=(
            df.explode("child_ids")
            .filter(pl.col("child_ids").is_in(df.filter(pl.col("is_deepest"))["id"]))
            .group_by("id", maintain_order=True)
            .agg(
                pl.all().exclude("child_ids").first(),
                pl.col("child_ids").alias("deepest_child_ids"),
            )
        ),
        on="id",
        how="left",
    ).with_columns(
        pl.col("deepest_child_ids").fill_null([]),
    )
    assert len(df) == len_0
    logger.info(f"CCF structure tree loaded in {time.time() - t0:.2f}s")
    return df


def get_ccf_structure_info(ccf_acronym_or_id: str | int) -> dict:
    """
    >>> get_ccf_structure_info('MOs')
    {'id': 993, 'atlas_id': 831, 'name': 'Secondary motor area', 'acronym': 'MOs', 'st_level': None, 'ontology_id': 1, 'hemisphere_id': 3, 'weight': 8690, 'parent_structure_id': 500, 'depth': 7, 'graph_id': 1, 'graph_order': 24, 'structure_id_path': '/997/8/567/688/695/315/500/993/', 'color_hex_triplet': '1F9D5A', 'neuro_name_structure_id': None, 'neuro_name_structure_id_path': None, 'failed': 'f', 'sphinx_id': 25, 'structure_name_facet': 1043755260, 'failed_facet': 734881840, 'safe_name': 'Secondary motor area', 'color_hex_int': 2071898, 'color_hex_str': '#1F9D5A', 'color_rgb': [0.12156862745098039, 0.615686274509804, 0.3529411764705882]}
    """
    if not isinstance(ccf_acronym_or_id, int):
        ccf_id: int = convert_ccf_acronyms_or_ids(ccf_acronym_or_id)
    else:
        ccf_id = ccf_acronym_or_id
    results = get_ccf_structure_tree_df().filter(pl.col("id") == ccf_id)
    if len(results) == 0:
        raise ValueError(
            f"No area found with acronym {convert_ccf_acronyms_or_ids(ccf_id)}"
        )
    if len(results) > 1:
        logger.warning(
            f"Multiple areas found: {results['acronym'].to_list()}. Using the first one"
        )
    return results[0].limit(1).to_dicts()[0]


def get_all_parents(ccf_acronym_or_id: str | int) -> list[str]:
    """
    >>> get_all_parents('MOs2/3')
    ['root', 'grey', 'CH', 'CTX', 'CTXpl', 'Isocortex', 'MO', 'MOs']
    """
    info = get_ccf_structure_info(ccf_acronym_or_id)
    parent_ids = [int(id_) for id_ in info["structure_id_path"].split("/")[1:-2]]
    parent_acronyms = (
        get_ccf_structure_tree_df()
        .filter(
            pl.col("id").is_in(parent_ids),
        )["acronym"]
        .to_list()
    )
    assert info["id"] not in parent_acronyms
    return parent_acronyms


def get_all_children(ccf_acronym_or_id: str | int) -> list[str]:
    """
    >>> get_all_children('MOs')
    ['MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b']
    """
    if not isinstance(ccf_acronym_or_id, int):
        ccf_id: int = convert_ccf_acronyms_or_ids(ccf_acronym_or_id)
    else:
        ccf_id = ccf_acronym_or_id
    children = (
        get_ccf_structure_tree_df()
        .filter(
            pl.col("structure_id_path").str.contains(f"/{ccf_id}/"),
            ~pl.col("structure_id_path").str.ends_with(f"/{ccf_id}/"),
        )["acronym"]
        .to_list()
    )
    assert str(ccf_acronym_or_id) not in children
    return children


def get_deepest_children(ccf_acronym_or_id: str | int) -> list[str]:
    """
    >>> get_deepest_children('MOs')
    ['MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b']
    >>> get_deepest_children('MOs1')
    ['MOs1']
    >>> assert 'VISpor' not in get_deepest_children('VIS')
    """
    if not isinstance(ccf_acronym_or_id, int):
        try:
            ccf_id: int = convert_ccf_acronyms_or_ids(ccf_acronym_or_id)
        except KeyError:
            return []
    else:
        ccf_id = ccf_acronym_or_id
    df = get_ccf_structure_tree_df().filter(
        pl.col("structure_id_path").str.contains(f"/{ccf_id}/")
    )
    if "root" in df["acronym"]:
        return []
    if len(df) == 0:
        return []
    return df.filter(~pl.col("id").is_in(pl.col("parent_structure_id")))[
        "acronym"
    ].to_list()


def group_child_labels_in_slice(
    slice_array: npt.NDArray[np.uint32],
    acronyms_or_ids: Iterable[str | int],
) -> npt.NDArray[np.uint32]:
    """
    For a given slice and CCF areas (acronyms or IDs), return a new slice with the labels grouped,
    so that all areas in the same group have the same label. For example, passing ["MOS"] would
    change the label index of all child areas in the MOs tree to have the MOs value.

    >>> mos_id = get_ccf_structure_info('MOs')['id']
    >>> slice_array = get_ccf_volume()[:, :, 100]
    >>> assert mos_id not in slice_array
    >>> new_slice = group_child_labels_in_slice(slice_array, ['MOs'])
    >>> assert mos_id in new_slice
    """
    slice_array = slice_array.copy()
    for ccf_acronym_or_id in acronyms_or_ids:
        if not isinstance(ccf_acronym_or_id, int):
            ccf_id: int = convert_ccf_acronyms_or_ids(ccf_acronym_or_id)
        else:
            ccf_id = ccf_acronym_or_id
        children = get_ccf_immediate_children_ids(ccf_id)
        if ccf_id in children:
            children.remove(ccf_id)
        logger.debug(
            f"Grouping {children} under {convert_ccf_acronyms_or_ids(ccf_acronym_or_id)}"
        )
        for child in children:
            slice_array[slice_array == child] = ccf_id
    return slice_array


def get_ccf_immediate_children_ids(ccf_acronym_or_id: str | int) -> set[int]:
    """
    >>> ids = get_ccf_immediate_children_ids('MOs')
    >>> sorted(convert_ccf_acronyms_or_ids(ids))
    ['MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b']
    """
    if not isinstance(ccf_acronym_or_id, int):
        ccf_id: int = convert_ccf_acronyms_or_ids(ccf_acronym_or_id)
    else:
        ccf_id = ccf_acronym_or_id
    return set(
        get_ccf_structure_tree_df()
        .filter(pl.col("parent_structure_id") == get_ccf_structure_info(ccf_id)["id"])
        .get_column("id")
    )


def get_ccf_children_ids_in_volume(ccf_acronym_or_id: str | int | None) -> set[int]:
    """
    >>> ids = get_ccf_children_ids_in_volume('MOs')
    >>> sorted(convert_ccf_acronyms_or_ids(ids))
    ['MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b']
    """
    if ccf_acronym_or_id is None or ccf_acronym_or_id == "":
        logger.info("No acronym provided, returning IDs for all non-zero areas")
        return get_ids_in_volume()
    if not isinstance(ccf_acronym_or_id, int):
        ccf_id: int = convert_ccf_acronyms_or_ids(ccf_acronym_or_id)
    else:
        ccf_id = ccf_acronym_or_id
    if ccf_id in get_ids_in_volume():
        return {ccf_id}
    children = get_ccf_immediate_children_ids(ccf_id)
    while not children.issubset(get_ids_in_volume()):
        children_not_in_volume = children - get_ids_in_volume()
        while children_not_in_volume:
            parent = children_not_in_volume.pop()
            children.remove(parent)
            children.update(get_ccf_immediate_children_ids(parent))
    logger.info(
        f"Found {len(children)} children for {convert_ccf_acronyms_or_ids(ccf_id)}"
    )
    return children


@functools.cache
def _ccf_acronym_to_id() -> dict[str, int]:
    """
    Use convert_ccf_acronyms_or_ids()

    >>> _ccf_acronym_to_id()['MOs']
    993
    >>> _ccf_acronym_to_id()['VISp']
    385
    """
    return dict(
        zip(*[get_ccf_structure_tree_df().get_column(col) for col in ("acronym", "id")])
    )


@functools.cache
def _ccf_id_to_acronym() -> dict[int, str]:
    """
    Use convert_ccf_acronyms_or_ids()

    >>> _ccf_id_to_acronym()[993]
    'MOs'
    >>> _ccf_id_to_acronym()[385]
    'VISp'
    """
    return dict(
        zip(*[get_ccf_structure_tree_df().get_column(col) for col in ("id", "acronym")])
    )


T = TypeVar("T", int, str, contravariant=True)


def convert_ccf_acronyms_or_ids(ccf_acronym_or_id: T | Iterable[T]) -> T | tuple[T]:
    """
    >>> convert_ccf_acronyms_or_ids('MOs')
    993
    >>> convert_ccf_acronyms_or_ids(993)
    'MOs'
    >>> convert_ccf_acronyms_or_ids(['MOs', 'VISp'])
    (993, 385)
    >>> convert_ccf_acronyms_or_ids([993, 385])
    ('MOs', 'VISp')
    """
    if isinstance(ccf_acronym_or_id, str):
        result = _ccf_acronym_to_id()[ccf_acronym_or_id]
    elif isinstance(ccf_acronym_or_id, int):
        result = _ccf_id_to_acronym()[ccf_acronym_or_id]
    else:
        result = tuple(convert_ccf_acronyms_or_ids(a) for a in ccf_acronym_or_id)
    logging.debug(f"Converted {ccf_acronym_or_id} to {result}")
    return result


@numba.jit(parallel=False)
def isin_numba(volume: npt.NDArray[np.uint32], ids: set[int]) -> npt.NDArray[np.bool_]:
    """
    May be faster than np.isin for large arrays - about the same as isin with
    'sort' for the 25 um volume
    From:
    https://stackoverflow.com/questions/62007409/is-there-method-faster-than-np-isin-for-large-array
    """
    shape_a: tuple[int, ...] = volume.shape
    volume: npt.NDArray[np.uint32] = volume.ravel()
    n: int = len(volume)
    result: npt.NDArray[np.bool_] = np.full(n, False)
    for i in numba.prange(n):
        if volume[i] in ids:
            result[i] = True
    return result.reshape(shape_a)


def get_ccf_volume_binary_mask(
    ccf_acronym_or_id: str | int | None = None,
    include_right_hemisphere: bool = False,
) -> npt.NDArray:
    """
    >>> volume = get_ccf_volume_binary_mask('MOs')
    >>> assert volume.any()
    >>> volume = get_ccf_volume_binary_mask()
    >>> assert volume.any()
    >>> volume = get_ccf_volume_binary_mask('CP')
    >>> assert volume.any()
    """
    if ccf_acronym_or_id is None or ccf_acronym_or_id == "":
        logger.info("No acronym provided, returning mask for the whole volume")
        for kind in ("gt",):
            t0 = time.time()
            if kind == "nonzero":
                masked_volume = get_ccf_volume(
                    right_hemisphere=include_right_hemisphere
                ).nonzero()
            else:
                masked_volume = (
                    get_ccf_volume(right_hemisphere=include_right_hemisphere) > 0
                )
            logger.info(
                f"Masked volume with no ccf area fetched with {kind=!r} in {time.time() - t0:.2f}s"
            )
        return masked_volume
    for kind in ("sort",):  # ('table', 'sort', 'numba')
        ids = get_ccf_children_ids_in_volume(ccf_acronym_or_id)
        if kind == "numba":
            s = ids
        elif kind in ("table", "sort"):
            s = np.fromiter(ids, dtype=int)
        else:
            raise ValueError(f"Invalid kind {kind}")
        v = get_ccf_volume(right_hemisphere=include_right_hemisphere)
        t0 = time.time()
        if kind == "numba":
            masked_volume = isin_numba(v, s)
        else:
            masked_volume = np.isin(
                v, s, kind=kind
            )  # set must be converted to np.array for np.isin to work
        logger.info(
            f"Masked volume found for {ccf_acronym_or_id} with {kind=!r} in {time.time() - t0:.2f}s"
        )
    return masked_volume


@functools.cache
def get_ids_in_volume() -> set[int]:
    return set(np.unique(get_ccf_volume()))


@functools.cache
def get_acronyms_in_volume() -> set[str]:
    """Reverse lookup on integers in ccf volume to get their corresponding acronyms

    >>> areas = get_acronyms_in_volume()
    >>> assert areas
    """
    return set(
        get_ccf_structure_tree_df().filter(pl.col("id").is_in(get_ids_in_volume()))[
            "acronym"
        ]
    )


@functools.cache
def get_ccf_projection(
    ccf_acronym_or_id: str | int | None = None,
    volume: npt.NDArray | None = None,
    include_right_hemisphere: bool = False,
    projection: Literal["sagittal", "coronal", "horizontal"] = "horizontal",
    slice_center_index: int | None = None,
    thickness_indices: int | None = None,
    with_color: bool = True,
    with_opacity: bool = True,
    normalize_rgb: bool = True,
) -> npt.NDArray:
    """
    >>> projection_img = get_ccf_projection() # no acronym returns all non-zero areas
    >>> assert projection_img.any()
    """
    if volume is None:
        volume = get_ccf_volume_binary_mask(
            ccf_acronym_or_id=ccf_acronym_or_id,
            include_right_hemisphere=include_right_hemisphere,
        )
    if not len(volume.shape) == 3:
        raise ValueError(f"Volume must be 3D, got {volume.shape=}")

    depth_dim = AXIS_TO_DIM[PROJECTION_TO_AXIS[projection]]

    if slice_center_index is None:
        slice_center_index = volume.shape[depth_dim] // 2
    if thickness_indices is None:
        thickness_indices = volume.shape[depth_dim]
    slice_ = slice(
        max(round(slice_center_index - 0.5 * thickness_indices), 0),
        min(
            round(slice_center_index + 0.5 * thickness_indices) + 1,
            volume.shape[depth_dim],
        ),
        1,
    )
    dims = [slice_ if dim == depth_dim else slice(None) for dim in range(volume.ndim)]
    subvolume = volume[dims[0], dims[1], dims[2]]
    if with_opacity:
        density_projection = (p := subvolume.sum(axis=depth_dim)) / p.max()
    else:
        density_projection = (subvolume.sum(axis=depth_dim) > 0) * 1

    # flip image axes to match projection_xy
    if (
        tuple(k for k in AXIS_TO_DIM if AXIS_TO_DIM[k] != depth_dim)
        != PROJECTION_YX[projection]
    ):
        density_projection = np.transpose(density_projection, (1, 0))

    if not with_color:
        logger.info(f"Returning binary_image.shape={density_projection.shape}")
        return density_projection

    if ccf_acronym_or_id is None or ccf_acronym_or_id == "":
        rgb = [0.6] * 3
    else:
        rgb: list[float] = get_ccf_structure_info(ccf_acronym_or_id)["color_rgb"]
    rgb_image = rgb * np.stack([np.ones_like(density_projection)] * 3, axis=-1)
    if not normalize_rgb:
        rgb_image *= 255
    rgba_image = np.concatenate(
        (rgb_image, density_projection[:, :, np.newaxis]), axis=-1
    )
    logger.info(f"Returning {rgba_image.shape=}")
    return rgba_image


def get_scatter_image(
    ccf_locations_df: pl.DataFrame | pl.LazyFrame,
    projection: Literal["sagittal", "coronal", "horizontal"] = "horizontal",
    image_shape: tuple[int, int] | None = None,
    opacity_range: tuple[float, float] = (0.8, 1.0),
    include_right_hemisphere: bool = False,
) -> npt.NDArray[np.floating]:
    """Alternative to using a scatter plot: create an image array (rgba) with the
    scatter points at full opacity white, and the rest of the image zeros, full
    opacity.

    - expects a DataFrame with columns 'ccf_ml', 'ccf_ap', 'ccf_dv'

    >>> ccf_df = pl.DataFrame({'ccf_ml': [1, 2, 3], 'ccf_ap': [1, 2, 3], 'ccf_dv': [1, 2, 3]})
    >>> scatter_image = get_scatter_image(ccf_df)
    >>> assert scatter_image.any()
    """
    t = time.time()
    if not image_shape:
        image_shape: tuple[int, int] = get_ccf_projection(projection=projection, include_right_hemisphere=include_right_hemisphere).shape[:2]  # type: ignore
    columns = tuple(f"ccf_{axis}" for axis in PROJECTION_YX[projection])
    yx_locations = (
        ccf_locations_df.lazy().select(columns).collect().to_numpy().T / RESOLUTION_UM
    )
    y_edges = np.linspace(0, image_shape[0], image_shape[0] + 1)
    x_edges = np.linspace(0, image_shape[1], image_shape[1] + 1)
    yx_hist_counts, _, _ = np.histogram2d(
        *yx_locations, bins=(y_edges, x_edges), density=True
    )
    logging.info(
        f"Creating opacity image from hist counts {yx_hist_counts.shape=}, {yx_hist_counts.max()=}"
    )
    # scale:
    if np.diff(opacity_range) == 0:
        yx_opacity = (yx_hist_counts > 0) * 1
    else:
        assert np.diff(opacity_range) > 0, f"{opacity_range=}"
        yx_opacity: npt.NDArray = opacity_range[0] + np.diff(opacity_range)[0] * (
            yx_hist_counts / yx_hist_counts.max()
        )
        # make background zero opacity:
        yx_opacity[yx_opacity <= opacity_range[0]] = 0
    yx_opacity[yx_opacity > 1] = 1
    yx_opacity[yx_opacity < 0] = 0
    logging.info(
        f"Opacity image created with range ({yx_opacity.min()}, {yx_opacity.max()})"
    )
    assert yx_opacity.any()
    assert yx_opacity.any()
    # yx_opacity = xy_density.T # flip xy back to image axes yx
    image = np.stack((*[(yx_opacity > 0) * 1] * 3, yx_opacity), axis=2)
    assert image.shape[2] == 4, f"{image.shape=}"
    assert len(image.shape) == 3, f"{image.shape=}, {len(image.shape)=}"
    assert image.any()
    logger.info(
        f"Returning scatter {image.shape=} ({columns}) in {time.time() - t:.2f}s"
    )
    return image


def project_first_nonzero_labels(
    volume: npt.NDArray,
    axis: int = AXIS_TO_DIM["dv"],
) -> npt.NDArray:
    """
    Project the first non-zero label encountered from one side of the 3D volume.

    Parameters:
    volume (np.ndarray): 3D array containing non-zero area labels.
    axis (int): Axis along which to project (0, 1, or 2).

    Returns:
    np.ndarray: 2D array with the projected labels.
    """
    if volume.ndim != 3:
        raise ValueError(f"Volume must be 3D: {volume.shape=}")
    dims = tuple(range(volume.ndim))
    if axis not in dims:
        raise ValueError("Axis must be 0, 1, or 2.")
    plane_dims = [d for d in dims if d != axis]
    mask = volume > 0
    idx_along_projection_axis = np.argmax(mask, axis=axis)
    idx_in_plane_axes = [np.arange(volume.shape[d]) for d in plane_dims]
    if axis == 0:
        projection = volume[
            idx_along_projection_axis,
            idx_in_plane_axes[0][:, None],
            idx_in_plane_axes[1],
        ]
    elif axis == 1:
        projection = volume[
            idx_in_plane_axes[0][:, None],
            idx_along_projection_axis,
            idx_in_plane_axes[1],
        ]
    elif axis == 2:
        projection = volume[
            idx_in_plane_axes[0][:, None],
            idx_in_plane_axes[1],
            idx_along_projection_axis,
        ]
    projection = projection.astype(float)
    projection[projection == 0] = np.nan
    return projection


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    import doctest

    doctest.testmod()
