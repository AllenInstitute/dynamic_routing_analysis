from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import altair as alt
import geopandas as gpd
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import rasterio.features
import shapely
import shapely.geometry


from dynamic_routing_analysis import ccf_utils

logger = logging.getLogger(__name__)


def _aggregate_top_layer(
    regions: npt.ArrayLike,
    values: npt.ArrayLike,
    agg_func_name: str,
) -> tuple[npt.NDArray[str], npt.NDArray[float]]:
    """
    For the 'top' view of cortex, layer 1 is in view, but typically not informative.
    Find all the layers for an area, aggregate and update layer 1's value

    >>> regions, values = _aggregate_top_layer(np.array(["VISp2/3", "VISp4"]), np.array([[2, np.nan], [np.nan, 3]]), "max")
    >>> list(regions)
    [np.str_('VISp1')]
    >>> list(values)
    [array([2., 3.])]
    """
    top_values = []
    top_regions = []

    def get_agg_layer_df(regions, values):
        return (
            # create a mapping of layer acronym to corresponding top-level layer and an aggregate value across all layers
            ccf_utils.get_ccf_structure_tree_df()
            # get values for any matching areas, but operate on all areas until very end:
            .join(
                pl.DataFrame({"acronym": regions, "value": values}),
                on="acronym",
                how="left",
            )  # preserve all rows
            # exclude nan values from calculations:
            .with_columns(
                pl.col("value").fill_nan(None),
            )
            .filter(
                pl.col("name").str.to_lowercase().str.contains("layer"),
            )
            .group_by("parent_structure_id")
            .agg("acronym", "value")
            .filter(
                pl.col("acronym").list.join("").str.contains("1"),
                pl.col("acronym").list.len()
                > 1,  # if only one acronym in group, it has a 1 in the name but no layers (e.g. CA1)
            )
            .with_columns(
                pl.col("acronym").list.first().alias("top_layer"),
                # if all values are null, agg = 0.0, which is incorrect
                # - we also don't want to drop nulls completely, so apply selectively:
                pl.when(pl.col("value").list.drop_nulls().list.len() != 0)
                .then(
                    pl.col("value")
                    .list.drop_nulls()
                    .list.eval(getattr(pl.element(), agg_func_name)())
                    .list.get(0)
                )
                .otherwise(pl.lit(np.nan)),
            )
            .explode("acronym")
            # filter rows with matches in actual data
            .join(
                pl.DataFrame({"acronym": regions, "value": values}),
                on="acronym",
                how="semi",
            )
        )

    df_left = get_agg_layer_df(regions, values[:, 0])
    df_right = get_agg_layer_df(regions, values[:, 1])
    for idx, r in enumerate(regions):
        if r not in df_left["acronym"] and r not in df_right["acronym"]:
            top_regions.append(r)
            top_values.append(values[idx, :])
        else:
            top_layer = df_left.filter(pl.col("acronym") == r)["top_layer"][
                0
            ]  # doesn't matter which df we use here
            if top_layer in top_regions:
                continue  # already added
            lr_values = [
                df.filter(pl.col("acronym") == r)["value"][0]
                for df in (df_left, df_right)
            ]
            top_regions.append(top_layer)
            top_values.append(lr_values)
    assert len(top_values) == len(top_regions)
    return np.array(top_regions), np.array(top_values)


def get_heatmap_gdf(
    regions: Iterable[str] | npt.ArrayLike,
    values: Iterable[float] | npt.ArrayLike,
    projection: str | Literal["sagittal", "coronal", "horizontal", "top"] = "top",
    position: float | None = None,
    remove_redundant_parents: bool = True,
    combine_child_patches: bool = True,
    top_layer_agg_func: str | None = None,
    horizontal_upright: bool = False,
) -> gpd.GeoDataFrame:
    t0 = time.time()
    # clean up inputs
    if position is None and projection != "top":
        raise ValueError("position must be specified for non-top projections")
    if projection == "top" and position is not None:
        logger.warning("position is ignored for top view")
    # check for duplicate regions
    if len(set(regions)) != len(regions):
        raise ValueError(
            f"Provide only one value per area acronym: {Counter(regions).most_common(3)=}"
        )
    regions = np.array(regions)
    values = np.array(values)
    if values.ndim == 1:
        values = np.array([values, np.full(len(values), np.nan)]).T
    if values.ndim > 1 and values.shape[0] == 2 and values.shape[1] != 2:
        values = values.T
    if values.shape[0] != regions.shape[0]:
        raise ValueError(f"{values.shape[0]=} does not match {regions.shape[0]=}")

    if top_layer_agg_func is not None:
        if not isinstance(top_layer_agg_func, str):
            raise ValueError(
                f"Layer aggregation function should be specified as a string, e.g. 'max', 'mean', not {top_layer_agg_func!r}"
            )
        regions, values = _aggregate_top_layer(regions, values, top_layer_agg_func)
    assert values.shape[1] == 2
    assert values.shape[0] == regions.shape[0]

    if not regions.size:
        regions = ccf_utils.get_ccf_structure_tree_df()["acronym"].to_numpy()
        values = np.full((len(regions), 2), np.nan)
    user_df = pl.DataFrame({"acronym": regions, "value": values}).join(
        ccf_utils.get_ccf_structure_tree_df(),
        on="acronym",
        how="left",
    )

    missing_ccf = set(regions) - set(user_df["acronym"])
    if missing_ccf:
        logger.warning(
            f"{len(missing_ccf)} acronyms specified in 'regions' have no match in CCF tree: {missing_ccf}"
        )

    expr = pl.col("id").is_in(user_df["parent_ids"].explode())
    redundant_parents = user_df.filter(expr)
    if len(redundant_parents):
        if remove_redundant_parents:
            logger.debug(
                f"Removing {len(redundant_parents)} regions as they are parents of other regions ({remove_redundant_parents=!r}): {redundant_parents['acronym'].to_numpy()}"
            )
            user_df = user_df.filter(~expr)
        else:
            raise ValueError(
                f"Found {len(redundant_parents)} regions that are parents of other regions: {redundant_parents['acronym'].to_numpy()} (try setting `remove_redundant_parents=True`)"
            )

    if not combine_child_patches:
        # only the deepest children in the tree are labelled in the volume:
        # replace any parents in user-specified 'regions' with their children
        #  - this is a requirement for plotting unless child patch polygons are combined below
        #! note: this will apply to both left and right hemispheres
        logger.info(
            f"Converting each of {len(regions)} regions to deepest children in CCF tree for plotting purposes ({combine_child_patches=!r})"
        )
        values_ = []
        child_ids = []
        for row in user_df.iter_rows(named=True):
            row_child_ids = row["deepest_child_ids"]
            if set(row_child_ids) & set(child_ids):
                raise ValueError(
                    f"In trying to add children for {row['acronym']!r}, {set(row_child_ids) & set(child_ids)} already have values: not sure how to continue"
                )
            child_ids.extend(row_child_ids)
            values_.extend([row["value"]] * len(row_child_ids))
        user_df = pl.DataFrame({"id": child_ids, "value": values_}).join(
            other=ccf_utils.get_ccf_structure_tree_df(),
            on="id",
            how="left",
        )

    # get slice/projection img:
    vol = ccf_utils.get_ccf_volume(left_hemisphere=True, right_hemisphere=True)
    if projection == "top":
        img = ccf_utils.project_first_nonzero_labels(vol)
        img[np.isnan(img)] = 0
    else:
        assert position is not None
        p = ccf_utils.ccf_to_volume_index(position)
        if projection == "sagittal":
            img = vol[p, :, :]
        elif projection == "horizontal":
            img = vol[:, p, :]
        elif projection == "coronal":
            img = vol[:, :, p].T

    mirror_lr = True
    dtype = np.int32
    volume_ml_midline = round(vol.shape[0] / 2)

    def _split_img_on_midline():
        img_l = img.copy().astype(dtype)
        img_r = img.copy().astype(dtype)
        assert len(np.unique(img_l)) == len(np.unique(img))
        if projection in ("top", "horizontal"):
            img_l[volume_ml_midline:, :] = 0
            if mirror_lr:
                img_r = img_l[::-1, :]
            else:
                img_r[:volume_ml_midline, :] = 0
        elif projection == "sagittal":
            assert position is not None
            if ccf_utils.ccf_to_volume_index(position) <= volume_ml_midline:
                img_r = np.zeros_like(img).astype(dtype)
            else:
                img_l = np.zeros_like(img).astype(dtype)
        elif projection == "coronal":
            img_l[:, volume_ml_midline:] = 0
            if mirror_lr:
                img_r = img_l[:, ::-1]
            else:
                img_r[:, :volume_ml_midline] = 0
        return img_l, img_r

    img_left, img_right = _split_img_on_midline()
    # get shapely polygons from connected labelled regions:
    transform = rasterio.Affine(
        ccf_utils.RESOLUTION_UM, 0, 0, 0, ccf_utils.RESOLUTION_UM, 0
    )

    for img in (img_left, img_right):
        is_left_img = img is img_left
        ids = []
        geometry = []

        if projection in ("top", "horizontal") and horizontal_upright:
            img = img.T

        # find connected regions in the image:
        assert img.dtype in (np.int32, np.float32)
        for polygon, label in rasterio.features.shapes(
            img, connectivity=4, transform=transform
        ):
            if label == 0 or np.isnan(label):
                continue
            g = shapely.geometry.shape(polygon)
            ids.append(int(label))
            geometry.append(g)

        # each area ID may have multiple discontiguous patches in labeled array: combine their polygons to get one polygon per area
        ids_ = []
        geometry_ = []
        for id_ in set(ids):
            geometry_.append(
                shapely.union_all([g for g, i in zip(geometry, ids) if i == id_])
            )
            ids_.append(id_)

        if combine_child_patches:
            # for each user-specified region acronym, group all its children into a single polygon
            combined_geometry = []
            combined_ids = []
            for row in user_df.iter_rows(named=True):
                row["acronym"]
                v = row["value"]
                value = v[0] if is_left_img else v[1]
                if np.isnan(value):
                    continue
                deepest_child_ids = row["deepest_child_ids"]
                if len(deepest_child_ids) <= 1 or not set(deepest_child_ids) & set(
                    ids_
                ):
                    continue
                combined_geometry.append(
                    shapely.union_all(
                        [g for g, i in zip(geometry_, ids_) if i in deepest_child_ids]
                    )
                )
                combined_ids.append(row["id"])
                for i in deepest_child_ids:
                    if i in ids_:
                        idx = ids_.index(i)
                        ids_.pop(idx)
                        geometry_.pop(idx)
            ids_ += combined_ids
            geometry_ += combined_geometry
        if is_left_img:
            ids_left, geometry_left = ids_, geometry_
        else:
            ids_right, geometry_right = ids_, geometry_

    gdf_left = gpd.GeoDataFrame({"id": ids_left, "geometry": geometry_left})
    gdf_right = gpd.GeoDataFrame({"id": ids_right, "geometry": geometry_right})
    for idx, gdf in enumerate((gdf_left, gdf_right)):
        gdf["position"] = position
        gdf["projection"] = projection
        gdf["value"] = np.nan
        gdf["hemisphere"] = "left" if gdf is gdf_left else "right"
        for row in user_df.iter_rows(named=True):
            gdf.loc[gdf["id"] == row["id"], "value"] = row["value"][idx]
    gdf = pd.concat((gdf_left, gdf_right)).merge(
        right=ccf_utils.get_ccf_structure_tree_df().to_pandas(),
        left_on="id",
        right_on="id",
        how="inner",  # keep rasterized regions that have ids in ccf structure tree
    )
    logger.info(
        f"Created GeoDataFrame with {len(gdf)} polygons in {time.time() - t0:.2f}s"
    )
    return gdf


def plot_brain_heatmap(
    regions: Iterable[str] | npt.ArrayLike,
    values: Iterable[float] | npt.ArrayLike,
    sagittal_planes: float | Iterable[float] | None = None,
    top_layer_agg_func: str | None = None,
    cmap: str = "viridis",
    clevels: tuple[float, float] | None = None,
    remove_redundant_parents: bool = True,
    combine_child_patches: bool = True,
    horizontal_upright: bool = False,
    labels: bool = False,
    labels_on_areas: bool = False,
    interactive: bool = False,
    patch_params: Mapping[str, Any] = {},
    missing_params: Mapping[str, Any] = {},
    plane_line_params: Mapping[str, Any] = {},
    annotation_params: Mapping[str, Any] = {},
) -> tuple[matplotlib.figure.Figure, tuple[pd.DataFrame]]:
    fig = plt.figure()
    gdfs = []
    if sagittal_planes is None:
        sagittal_planes = []
    elif not isinstance(sagittal_planes, Iterable):
        sagittal_planes = (sagittal_planes,)
    else:
        sagittal_planes = tuple(sagittal_planes)  # type: ignore

    if clevels is not None:
        clevels = tuple(clevels)  # type: ignore
        if len(clevels) != 2:
            raise ValueError("clevels must be a sequence of length 2")
    else:
        clevels = (np.nanmin(np.array(values)), np.nanmax(np.array(values)))
    norm = matplotlib.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
    axes = []

    vol = ccf_utils.get_ccf_volume(True, True)
    max_ap = vol.shape[2] * ccf_utils.RESOLUTION_UM
    max_dv = vol.shape[1] * ccf_utils.RESOLUTION_UM
    max_ml = vol.shape[0] * ccf_utils.RESOLUTION_UM
    height_top = max_ap if horizontal_upright else max_ml
    height_sagittal = max_dv
    gs = matplotlib.gridspec.GridSpec(
        len(sagittal_planes) + 2,
        1,
        figure=fig,
        height_ratios=[height_top / height_sagittal]
        + [1] * len(sagittal_planes)
        + [0.1],
    )
    axes.append(ax_top := fig.add_subplot(gs[0, 0]))
    gdf = get_heatmap_gdf(
        regions=regions,
        values=values,
        projection="top",
        horizontal_upright=horizontal_upright,
        remove_redundant_parents=remove_redundant_parents,
        combine_child_patches=combine_child_patches,
        top_layer_agg_func=top_layer_agg_func,
    )
    gdfs.append(gdf)
    missing_kwds = (
        {"color": "lightgrey"}
        | {k: v for k, v in patch_params.items() if k in ("edgecolor", "linewidth")}
        | missing_params
    )
    patch_kwds = {"edgecolor": "darkgrey", "linewidth": 0.1} | patch_params
    gdf.plot(
        column="value",
        cmap=cmap,
        missing_kwds=missing_kwds,
        ax=ax_top,
        norm=norm,
        **patch_kwds,
    )

    if labels:
        if labels_on_areas:
            # add a label for each area in the top view with a value (in right hemisphere only)
            for _idx, area in enumerate(
                gdf.dropna(subset=["value"])["acronym"].unique()
            ):
                rows = gdf[(gdf["acronym"] == area)]
                rows_with_values = rows.dropna(subset=["value"])
                if len(rows_with_values) == 0:
                    continue
                if len(rows_with_values) == 2:
                    # remove left row if right has value
                    rows_with_values = rows_with_values[
                        rows_with_values["hemisphere"] == "right"
                    ]
                assert len(rows_with_values) == 1
                row = rows.iloc[0]
                center_x, center_y = row.geometry.centroid.x, row.geometry.centroid.y
                if row["hemisphere"] == "left":
                    center_y = (
                        2 * (max_ap if horizontal_upright else max_ml) / 2 - center_y
                    )
                if labels_on_areas:
                    ax_top.text(
                        center_x,
                        center_y,
                        row["acronym"],
                        **{"fontsize": 1.5, "ha": "center", "va": "center"}
                        | annotation_params,
                    )
        else:
            # make annotations with lines pointing to the center of the area, spaced evenly around
            # the top-half of axes in an arc
            ap_center_of_mass = 0.5 * max_ap  # shift the center slightly posterior
            if horizontal_upright:
                brain_center_angle = 0.0
                brain_center_x, brain_center_y = max_ml / 2, ap_center_of_mass
            else:
                brain_center_angle = 0
                brain_center_x, brain_center_y = ap_center_of_mass, max_ml / 2
            arc_radius = 0.55 * max_ap
            angular_extent = np.pi
            label_gdf = gdf
            angular_spacing = angular_extent / len(
                label_gdf.dropna(subset=["value"])["acronym"].unique()
            )
            # add columns for distance and angle of centroid from brain center
            label_gdf["distance_from_center"] = np.sqrt(
                (label_gdf.geometry.centroid.x - brain_center_x) ** 2
                + (label_gdf.geometry.centroid.y - brain_center_y) ** 2
            )
            label_gdf["angle_from_horizontal"] = np.arctan2(
                label_gdf.geometry.centroid.y - brain_center_y,
                label_gdf.geometry.centroid.x - brain_center_x,
            )
            label_gdf = label_gdf.sort_values("angle_from_horizontal").dropna(
                subset=["value"]
            )
            for idx, area in enumerate(label_gdf["acronym"].unique()):
                rows = label_gdf[(label_gdf["acronym"] == area)]
                rows_with_values = rows.dropna(subset=["value"])
                if len(rows_with_values) == 0:
                    continue
                if len(rows_with_values) == 2:
                    # remove left row if right has value
                    rows_with_values = rows_with_values[
                        rows_with_values["hemisphere"] == "right"
                    ]
                assert len(rows_with_values) == 1
                row = rows_with_values.iloc[0]
                if set(row["parent_ids"]) & set(label_gdf["id"]):
                    parent_ids = row["parent_ids"]
                    # skip if any parents are due to be plot
                    if (
                        not label_gdf[label_gdf["id"].isin(parent_ids)]
                        .dropna(subset=["value"])
                        .empty
                    ):
                        continue
                center_x, center_y = row.geometry.centroid.x, row.geometry.centroid.y
                annotation_angle = brain_center_angle - np.pi - idx * angular_spacing
                if row["hemisphere"] == "left":
                    center_y = 2 * brain_center_y - center_y
                    annotation_angle = 2 * np.pi - annotation_angle
                length = arc_radius - row["distance_from_center"]
                # jitter length
                length += np.random.rand() * 0.3 * length
                x = center_x + length * np.cos(annotation_angle)
                y = center_y - length * np.sin(annotation_angle)
                ax_top.annotate(
                    row["acronym"],
                    xy=(center_x, center_y),
                    xytext=(x, y),
                    **{
                        "arrowprops": {"lw": 0.1, "arrowstyle": "-", "color": "black"},
                        "fontsize": 2,
                        "font": "arial",
                    }
                    | annotation_params,
                )
    for i, coord in enumerate(sorted(sagittal_planes, reverse=True)):
        axes.append(ax := fig.add_subplot(gs[i + 1, 0]))
        gdf = get_heatmap_gdf(
            regions=regions,
            values=values,
            projection="sagittal",
            remove_redundant_parents=remove_redundant_parents,
            combine_child_patches=combine_child_patches,
            position=coord,
        )
        gdfs.append(gdf)
        gdf.plot(
            column="value",
            cmap=cmap,
            missing_kwds=missing_kwds,
            ax=ax,
            norm=norm,
            **patch_params,
        )
        ax.set_xlim(0, max_ap)
        ax.set_ylim(0, max_dv)
        ax.invert_yaxis()
        if horizontal_upright:
            axlinefunc = ax_top.axvline
        else:
            axlinefunc = ax_top.axhline
        axlinefunc(
            coord, **{"color": "k", "linestyle": "--", "lw": 0.1} | plane_line_params
        )

    if horizontal_upright:
        ax_top.set_xlim(0, max_ml)
        ax_top.set_ylim(max_ap, 0)
    else:
        ax_top.set_xlim(0, max_ap)
        ax_top.set_ylim(0, max_ml)

    axes.append(ax_cbar := fig.add_subplot(gs[len(sagittal_planes) + 1, 0]))
    fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(*clevels),
            cmap=cmap,
        ),
        ax=ax_cbar,
        fraction=0.5,
        orientation="horizontal",
        location="bottom",
    )
    for ax in axes:
        ax.set_aspect(1)
        ax.set_axis_off()
        ax.set_clip_on(False)

    if interactive:
        chart = plot_gdf_alt(gdfs, ccf_colors=False)
        fig.clf()
        return chart, tuple(gdfs)
    else:
        return fig, tuple(gdfs)


def plot_gdf_alt(
    gdfs: gpd.GeoDataFrame | Iterable[gpd.GeoDataFrame],
    ccf_colors: bool = False,
    value_name: str = "value",
) -> alt.Chart:
    if isinstance(gdfs, gpd.GeoDataFrame):
        gdfs = (gdfs,)
    else:
        gdfs = tuple(gdfs)
    vol = ccf_utils.get_ccf_volume(True, True)
    max_ap = vol.shape[2] * ccf_utils.RESOLUTION_UM
    max_dv = vol.shape[1] * ccf_utils.RESOLUTION_UM
    max_ml = vol.shape[0] * ccf_utils.RESOLUTION_UM
    charts = []

    @functools.cache
    def get_background_gdf(projection: str, position: float | None):
        return gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.union_all(
                        list(
                            get_heatmap_gdf(
                                regions=[],
                                values=[],
                                projection=projection,
                                position=position,
                            )["geometry"].values
                        )
                    )
                ],
            }
        )

    def get_fit(projection: str, is_top_upright):
        xmin, ymin = 0, 0
        if projection in ("top", "horizontal"):
            if is_top_upright:
                xmax, ymax = max_ml, max_ap
            else:
                xmax, ymax = max_ap, max_ml
        elif projection == "sagittal":
            xmax, ymax = max_ap, max_dv
        elif projection == "coronal":
            xmax, ymax = max_ml, max_dv
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [xmax, ymax],
                        [xmax, ymin],
                        [xmin, ymin],
                        [xmin, ymax],
                        [xmax, ymax],
                    ]
                ],
            },
            "properties": {},
        }

    for gdf in gdfs:
        projection = gdf["projection"].iloc[0]
        if projection in ("top", "horizontal"):
            is_upright = bool(
                gdf.geometry.centroid.y.max() > gdf.geometry.centroid.x.max()
            )
            background_position = None if projection == "top" else max_dv / 2
        elif projection == "sagittal":
            background_position = (max_ml / 2) + 250
        elif projection == "coronal":
            background_position = max_ap / 2
        else:
            raise ValueError(f"Invalid projection {projection}")

        tooltip = ["id:Q", "acronym:N", "name:N", "hemisphere:N"]
        if ccf_colors:
            color = alt.Color("acronym").scale(
                domain=ccf_utils.get_ccf_structure_tree_df()["acronym"].to_list(),
                range=ccf_utils.get_ccf_structure_tree_df()["color_hex_str"].to_list(),
            )
        else:
            tooltip.append("value:Q")
            color = alt.Color(
                "value:Q",
                title=value_name,
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
                # condition=condition,
            )
        chart = (
            alt.Chart(gdf)
            .mark_geoshape(
                strokeWidth=0.05,
                stroke="darkgrey",
            )
            .encode(
                tooltip=tooltip,
                color=color,
            )
            .project(
                type="identity",
                reflectY=projection != "sagittal",
                fit=get_fit(
                    projection,
                    is_upright if projection in ("top", "horizontal") else None,
                ),
            )
        )
        null_slice = (
            alt.Chart(gdf[gdf["value"].isna() | gdf["value"].isnull()])
            .mark_geoshape(
                strokeWidth=0.05,
                stroke="white",
            )
            .encode(
                tooltip=tooltip,
                color=alt.value("#eee"),
            )
            .project(
                type="identity",
                reflectY=projection != "sagittal",
                fit=get_fit(
                    projection,
                    is_upright if projection in ("top", "horizontal") else None,
                ),
            )
        )
        chart = alt.layer(null_slice, chart)

        with_background = True
        if with_background:
            background = (
                alt.Chart(get_background_gdf(projection, background_position))
                .mark_geoshape(strokeWidth=0.2, stroke="darkgrey")
                .encode(
                    color=alt.value("#fff"),
                )
                .project(
                    type="identity",
                    reflectY=projection != "sagittal",
                    fit=get_fit(
                        projection,
                        is_upright if projection in ("top", "horizontal") else None,
                    ),
                )
            )
            chart = alt.layer(background, chart)

        # add lines (positions aren't correct):
        """
        if projection in ("top", "horizontal"):
            other_gdfs = [
                gdf
                for gdf in gdfs
                if gdf["projection"].iloc[0] not in ("top", "horizontal")
            ]
            positions = [gdf["position"].iloc[0] for gdf in other_gdfs]
            projections = [gdf["projection"].iloc[0] for gdf in other_gdfs]
            for pos, proj in zip(positions, projections):
                if proj == "sagittal":
                    chart += (
                        alt.Chart(pl.DataFrame({"y": [pos]}))
                        .mark_rule(strokeDash=[2, 2])
                        .encode(y="y")
                    )
                elif proj == "coronal":
                    chart += alt.Chart({"x": [pos]}).mark_rule().encode(x="x")
        """
        charts.append(chart)
    return alt.hconcat(*charts).configure_legend(disable=True)
