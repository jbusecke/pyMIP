import warnings

import numpy as np
import pkg_resources
import xarray as xr
import yaml

from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds


path = "specs/staggered_grid_config.yaml"  # always use slash
grid_spec = pkg_resources.resource_filename(__name__, path)


def _parse_bounds_vertex(da, dim="bnds", position=[0, 1]):
    """Convenience function to extract positions from bounds/vertices"""
    return tuple([da.isel({dim: i}).load().data for i in position])


def _interp_vertex_to_bounds(da, orientation):
    """
    Convenience function to average 4 vertex points into two bound points.
    Helpful to recreate e.g. the latitude at the `lon_bounds` points.
    """
    if orientation == "x":
        datasets = [
            da.isel(vertex=[0, 1]).mean("vertex"),
            da.isel(vertex=[3, 2]).mean("vertex"),
        ]
    elif orientation == "y":
        datasets = [
            da.isel(vertex=[0, 3]).mean("vertex"),
            da.isel(vertex=[1, 2]).mean("vertex"),
        ]

    return xr.concat(datasets, dim="bnds")


def distance_deg(lon0, lat0, lon1, lat1):
    """Calculate the distance in degress longitude and latitude between two points

    Parameters
    ----------
    lon0 : np.array
        Longitude of first point
    lat0 : np.array
        Latitude of first point
    lon1 : np.array
        Longitude of second point
    lat1 : np.array
        Latitude of second point
    """
    delta_lon = lon1 - lon0
    delta_lat = lat1 - lat0
    # very small differences can end up negative, so zero them out based on a simple
    # criterion
    # this should work for CMIP6 (no 1/1 deg models) but should be based on actual grid
    # info in the future
    small_crit = 1 / 10
    delta_lon = np.where(
        abs(delta_lon) < small_crit, 0.0, delta_lon
    )  # , np.nan, delta_lon)
    delta_lat = np.where(
        abs(delta_lat) < small_crit, 0.0, delta_lat
    )  # , np.nan, delta_lat)

    #     # some bounds are wrapped aroud the lon discontinuty.
    delta_lon = np.where(delta_lon < (-small_crit * 2), 360 + delta_lon, delta_lon)  #
    delta_lon = np.where(
        delta_lon > (360 + small_crit * 2), -360 + delta_lon, delta_lon
    )

    return delta_lon, delta_lat


def distance(lon0, lat0, lon1, lat1):
    """Calculate the distance in m between two points on a spherical globe

    Parameters
    ----------
    lon0 : np.array
        Longitude of first point
    lat0 : np.array
        Latitude of first point
    lon1 : np.array
        Longitude of second point
    lat1 : np.array
        Latitude of second point
    """
    Re = 6.378e6
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    dy = Re * (np.pi * delta_lat / 180)
    dx = Re * (np.pi * delta_lon / 180) * np.cos(np.pi * lat0 / 180)
    return np.sqrt(dx**2 + dy**2)


def recreate_metrics(ds, grid):
    """Recreate a full set of horizontal distance metrics.

    Calculates distances between points in lon/lat coordinates


    The naming of the metrics is as follows:
    [metric_axis]_t : metric centered at tracer point
    [metric_axis]_gx : metric at the cell face on the x-axis.
        For instance `dx_gx` is the x distance centered on the eastern cell face if the shift is `right`
    [metric_axis]_gy : As above but along the y-axis
    [metric_axis]_gxgy : The metric located at the corner point.
        For example `dy_dxdy` is the y distance on the south-west corner if both axes as shifted left.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    grid : xgcm.Grid
        xgcm Grid object matching `ds`

    Returns
    -------
    xr.Dataset, dict
        Dataset with added metrics as coordinates and dictionary that can be passed to xgcm.Grid to recognize new metrics
    """
    ds = ds.copy()

    # Since this puts out numpy arrays, the arrays need to be transposed correctly
    transpose_dims = ["y", "x"]
    dims = [di for di in ds.dims if di not in transpose_dims]

    ds = ds.transpose(*tuple(transpose_dims + dims))

    # is the vel point on left or right?
    axis_vel_pos = {
        axis: list(set(grid.axes[axis].coords.keys()) - set(["center"]))[0]
        for axis in ["X", "Y"]
    }
    # determine the appropriate vertex position for the north/south and east/west edge,
    # based on the grid config
    if axis_vel_pos["Y"] in ["left"]:
        ns_vertex_idx = [0, 3]
        ns_bound_idx = [0]
    elif axis_vel_pos["Y"] in ["right"]:
        ns_vertex_idx = [1, 2]
        ns_bound_idx = [1]

    if axis_vel_pos["X"] in ["left"]:
        ew_vertex_idx = [0, 1]
        ew_bound_idx = [0]
    elif axis_vel_pos["X"] in ["right"]:
        ew_vertex_idx = [3, 2]
        ew_bound_idx = [1]

    # infer dx at tracer points
    if "lon_bounds" in ds.coords and "lat_vertices" in ds.coords:
        lon0, lon1 = _parse_bounds_vertex(ds["lon_bounds"])
        lat0, lat1 = _parse_bounds_vertex(
            _interp_vertex_to_bounds(ds["lat_vertices"], "x")
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dx_t"] = xr.DataArray(dist, coords=ds.lon.coords)

    # infer dy at tracer points
    if "lat_bounds" in ds.coords and "lon_vertices" in ds.coords:
        lat0, lat1 = _parse_bounds_vertex(ds["lat_bounds"])
        lon0, lon1 = _parse_bounds_vertex(
            _interp_vertex_to_bounds(ds["lon_vertices"], "y")
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dy_t"] = xr.DataArray(dist, coords=ds.lon.coords)

    if "lon_vertices" in ds.coords and "lat_vertices" in ds.coords:
        # infer dx at the north/south face
        lon0, lon1 = _parse_bounds_vertex(
            ds["lon_vertices"], dim="vertex", position=ns_vertex_idx
        )
        lat0, lat1 = _parse_bounds_vertex(
            ds["lat_vertices"], dim="vertex", position=ns_vertex_idx
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dx_gy"] = xr.DataArray(
            dist, coords=grid.interp(ds.lon, "Y", boundary="extrapolate").coords
        )

        # infer dy at the east/west face
        lon0, lon1 = _parse_bounds_vertex(
            ds["lon_vertices"], dim="vertex", position=ew_vertex_idx
        )
        lat0, lat1 = _parse_bounds_vertex(
            ds["lat_vertices"], dim="vertex", position=ew_vertex_idx
        )
        dist = distance(lon0, lat0, lon1, lat1)
        ds.coords["dy_gx"] = xr.DataArray(
            dist, coords=grid.interp(ds.lon, "X", boundary="extrapolate").coords
        )

    # for the distances that dont line up with the cell boundaries we need some different logic
    boundary = "extend"
    # TODO: This should be removed once we have the default boundary merged in xgcm

    # infer dx at eastern/western bound from tracer points
    lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
        ds.lon.load(), axis_vel_pos["X"]
    )
    lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
        ds.lat.load(), axis_vel_pos["X"]
    )
    dx = distance(lon0, lat0, lon1, lat1)
    ds.coords["dx_gx"] = xr.DataArray(
        dx, coords=grid.interp(ds.lon, "X", boundary=boundary).coords
    )

    # infer dy at northern bound from tracer points
    lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
        ds.lat.load(), axis_vel_pos["Y"], boundary=boundary
    )
    lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
        ds.lon.load(), axis_vel_pos["Y"], boundary=boundary
    )
    dy = distance(lon0, lat0, lon1, lat1)
    ds.coords["dy_gy"] = xr.DataArray(
        dy, coords=grid.interp(ds.lat, "Y", boundary=boundary).coords
    )

    # infer dx at the corner point
    lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
        _interp_vertex_to_bounds(ds.lon_vertices.load(), "y")
        .isel(bnds=ns_bound_idx)
        .squeeze(),
        axis_vel_pos["X"],
    )
    lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
        ds.lat_bounds.isel(bnds=ns_bound_idx).squeeze().load(), axis_vel_pos["X"]
    )
    dx = distance(lon0, lat0, lon1, lat1)
    ds.coords["dx_gxgy"] = xr.DataArray(
        dx,
        coords=grid.interp(
            grid.interp(ds.lon, "X", boundary=boundary), "Y", boundary=boundary
        ).coords,
    )

    # infer dy at the corner point
    lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
        _interp_vertex_to_bounds(ds.lat_vertices.load(), "x")
        .isel(bnds=ew_bound_idx)
        .squeeze(),
        axis_vel_pos["Y"],
    )
    lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
        ds.lon_bounds.isel(bnds=ew_bound_idx).squeeze().load(), axis_vel_pos["Y"]
    )
    dy = distance(lon0, lat0, lon1, lat1)
    ds.coords["dy_gxgy"] = xr.DataArray(
        dy,
        coords=grid.interp(
            grid.interp(ds.lon, "X", boundary=boundary), "Y", boundary=boundary
        ).coords,
    )

    # infer dz at tracer point
    if "lev_bounds" in ds.coords:
        ds = ds.assign_coords(
            dz_t=("lev", ds["lev_bounds"].diff("bnds").squeeze(drop=True).data)
        )

    metrics_dict = {
        "X": [co for co in ["dx_t", "dx_gy", "dx_gx"] if co in ds.coords],
        "Y": [co for co in ["dy_t", "dy_gy", "dy_gx"] if co in ds.coords],
        "Z": [co for co in ["dz_t"] if co in ds.coords],
    }
    # # only put out axes that have entries
    metrics_dict = {k: v for k, v in metrics_dict.items() if len(v) > 0}

    return ds, metrics_dict


def detect_shift(ds_base, ds, axis):
    """Detects the shift of `ds` relative to `ds` on logical grid axes, using
    lon and lat positions.

    Parameters
    ----------
    ds_base : xr.Dataset
        Reference ('base') dataset to compare to. Assumed that this is located at the 'center' coordinate.
    ds : xr.Dataset
        Comparison dataset. The resulting shift will be computed as this dataset relative to `ds_base`
    axis : str
        xgcm logical axis on which to detect the shift

    Returns
    -------
    str
        Shift string output, in xgcm conventions.
    """
    ds_base = ds_base.copy()
    ds = ds.copy()
    axis = axis.lower()
    axis_coords = {"x": "lon", "y": "lat"}

    # check the shift only for one point, somewhat in the center to avoid the
    # distorted polar regions
    check_point = {"x": len(ds_base.x) // 2, "y": len(ds_base.y) // 2}
    check_point_diff = {k: [v, v + 1] for k, v in check_point.items()}

    shift = (
        ds.isel(**check_point)[axis_coords[axis]].load().data
        - ds_base.isel(**check_point)[axis_coords[axis]].load().data
    )
    diff = ds[axis].isel({axis: check_point_diff[axis]}).diff(axis).data.tolist()[0]
    threshold = 0.1
    # the fraction of full cell distance, that a point has to be shifted in order to
    # be recognized.
    # This avoids detection of shifts for very small differences that sometimes happen
    # if the coordinates were written e.g. by different modulel of a model

    axis_shift = "center"

    if shift > (diff * threshold):
        axis_shift = "right"
    elif shift < -(diff * threshold):
        axis_shift = "left"
    return axis_shift


def create_full_grid(base_ds, grid_dict=None):
    """Generate a full xgcm-compatible dataset from a reference datasets `base_ds`.
    This dataset should be representing a tracer fields, e.g. the cell center.

    Parameters
    ----------
    base_ds : xr.Dataset
        The reference ('base') datasets, assumed to be at the tracer position/cell center
    grid_dict : dict, optional
        Dictionary with info about the grid staggering.
        Must be encoded using the base_ds attrs (e.g. {'model_name':{'axis_shift':{'X':'left',...}}}).
        If deactivated (default), will load from the internal database for CMIP6 models, by default None

    Returns
    -------
    xr.Dataset
        xgcm compatible dataset
    """

    # load dict with grid shift info for each axis
    if grid_dict is None:
        ff = open(grid_spec, "r")
        grid_dict = yaml.safe_load(ff)
        ff.close()

    source_id = base_ds.attrs["source_id"]
    grid_label = base_ds.attrs["grid_label"]

    # if source_id not in dict, and grid label is gn, warn and ask to submit an issue
    try:
        axis_shift = grid_dict[source_id][grid_label]["axis_shift"]
    except KeyError:
        warnings.warn(
            f"Could not find the source_id/grid_label ({source_id}/{grid_label}) combo in `grid_dict`, returning `None`. Please submit an issue to github: https://github.com/jbusecke/xmip/issues"
        )
        return None

    position = {k: ("center", axis_shift[k]) for k in axis_shift.keys()}

    axis_dict = {"X": "x", "Y": "y"}

    ds_grid = generate_grid_ds(
        base_ds, axis_dict, position=position, boundary_discontinuity={"X": 360}
    )

    # TODO: man parse lev and lev_bounds as center and outer dims.
    # I should also be able to do this with `generate_grid_ds`, but here we
    # have the `lev_bounds` with most models, so that is probably more reliable.
    # cheapest solution right now
    if "lev" in ds_grid.dims:
        ds_grid["lev"].attrs["axis"] = "Z"

    return ds_grid


def combine_staggered_grid(
    ds_base, other_ds=None, recalculate_metrics=False, grid_dict=None, **kwargs
):
    """Combine a reference datasets with a list of other datasets to a full xgcm-compatible staggered grid datasets.


    Parameters
    ----------
    ds_base : xr.Dataset
        The reference ('base') datasets, assumed to be at the tracer position/cell center
    other_ds : list,xr.Dataset, optional
        List of datasets representing different variables. Their grid position will be
        automatically detected relative to `ds_base`. Coordinates and attrs of these added datasets will be lost
        , by default None
    recalculate_metrics : bool, optional
        nables the reconstruction of grid metrics usign simple
        spherical geometry, by default False

        !!! Check your results carefully when using reconstructed values,
        these might differe substantially if the grid geometry is complicated.
    grid_dict : dict, optional
        Dictionary for staggered grid setup. See `create_full_grid` for detauls
        If None (default), will load staggered grid info from internal database, by default None

    Returns
    -------
    xr.Dataset
        Single xgcm-compatible dataset, containing all variables on their respective staggered grid position.
    """
    ds_base = ds_base.copy()
    if isinstance(other_ds, xr.Dataset):
        other_ds = [other_ds]

    ds_g = create_full_grid(ds_base, grid_dict=grid_dict)

    if ds_g is None:
        warnings.warn("Staggered Grid creation failed. Returning `None`")
        return None, None

    # save attrs out for later (something during alignment destroys them)
    dim_attrs_dict = {}
    for di in ds_g.dims:
        dim_attrs_dict[di] = ds_g[di].attrs

    # TODO: metrics and interpolation of metrics if they are parsed

    # parse other variables
    if other_ds is not None:
        for ds_new in other_ds:
            ds_new = ds_new.copy()
            # strip everything but the variable_id (perhaps I would want to
            # loosen this in the future)
            ds_new = ds_new[ds_new.attrs["variable_id"]]

            if not all(
                [
                    len(ds_new[di]) == len(ds_g[di])
                    for di in ds_new.dims
                    if di not in ["member_id", "time"]
                ]
            ):
                warnings.warn(
                    f"Could not parse `{ds_new.name}`, due to a size mismatch. If this is the MRI model, the grid convention is currently not supported."
                )
            else:
                # detect shift and rename accordingly
                rename_dict = {}
                for axis in ["X", "Y"]:
                    shift = detect_shift(ds_base, ds_new, axis)

                    if shift != "center":
                        rename_dict[axis.lower()] = axis.lower() + "_" + shift
                ds_new = ds_new.rename(rename_dict)
                ds_new = ds_new.reset_coords(drop=True)
                # TODO: This needs to be coded more generally, for now hardcode x and y
                force_align_dims = [di for di in ds_new.dims if "x" in di or "y" in di]
                _, ds_new = xr.align(
                    ds_g.copy(),
                    ds_new,
                    join="override",
                    exclude=[di for di in ds_new.dims if di not in force_align_dims],
                )
                additional_dims = [di for di in ds_new.dims if di not in ds_g.dims]
                if len(additional_dims) > 0:
                    raise RuntimeError(
                        f"While trying to parse `{ds_new.name}`, detected dims that are not in the base dataset:[{additional_dims}]"
                    )
                ds_g[ds_new.name] = ds_new

    # Restore dims attrs from the beginning
    for di in ds_g.dims:
        ds_g.coords[di].attrs.update(dim_attrs_dict[di])

    grid_kwargs = {"periodic": ["X"]}
    grid_kwargs.update(kwargs)
    grid = Grid(ds_g, grid_kwargs)

    # if activated calculate metrics
    if recalculate_metrics:
        grid_kwargs.pop(
            "metrics", None
        )  # remove any passed metrics when recalculating them
        # I might be able to refine this more to e.g. allow axes that are not recreated.

        ds_g, metrics_dict = recreate_metrics(ds_g, grid)
        # this might fail in circumstances, where the
        grid_kwargs["metrics"] = metrics_dict
        grid = Grid(ds_g, **grid_kwargs)
    return grid, ds_g
