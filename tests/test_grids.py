import numpy as np
import pytest
import xarray as xr

from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds

from xmip.grids import (
    _interp_vertex_to_bounds,
    _parse_bounds_vertex,
    combine_staggered_grid,
    create_full_grid,
    detect_shift,
    distance,
    distance_deg,
    recreate_metrics,
)


def _add_small_rand(da):
    return da + (np.random.rand(*da.shape) * 0.05)


def _test_data(grid_label="gn", z_axis=True):
    xt = np.arange(4) + 1
    yt = np.arange(5) + 1
    zt = np.arange(6) + 1

    x = xr.DataArray(xt, coords=[("x", xt)])
    y = xr.DataArray(yt, coords=[("y", yt)])
    lev = xr.DataArray(zt, coords=[("lev", zt)])

    # Need to add a tracer here to get the tracer dimsuffix
    coords = [("x", x.data), ("y", y.data)]
    data = np.random.rand(len(xt), len(yt))
    dims = ["x", "y"]

    if z_axis:
        coords.append(("lev", lev.data))
        data = np.random.rand(len(x), len(y), len(lev))
        dims = ["x", "y", "lev"]

    tr = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
    )

    lon_raw = xr.DataArray(xt, coords=[("x", xt)])
    lat_raw = xr.DataArray(yt, coords=[("y", yt)])
    lon = lon_raw * xr.ones_like(lat_raw)
    lat = xr.ones_like(lon_raw) * lat_raw

    lon_bounds_e = lon + 0.5
    lon_bounds_w = lon - 0.5 + (np.random.rand(*lon.shape) * 0.05)
    lat_bounds_n = lat + 0.5 + (np.random.rand(*lon.shape) * 0.05)
    lat_bounds_s = lat - 0.5 + (np.random.rand(*lon.shape) * 0.05)

    lon_bounds = xr.concat(
        [_add_small_rand(lon_bounds_w), _add_small_rand(lon_bounds_w)], dim="bnds"
    )
    lat_bounds = xr.concat(
        [_add_small_rand(lat_bounds_s), _add_small_rand(lat_bounds_n)], dim="bnds"
    )

    if z_axis:
        lev_bounds = xr.concat(
            [_add_small_rand(lev - 0.5), _add_small_rand(lev + 0.5)], dim="bnds"
        )

    lon_vertices = xr.concat(
        [
            _add_small_rand(lon_bounds_e),
            _add_small_rand(lon_bounds_e),
            _add_small_rand(lon_bounds_w),
            _add_small_rand(lon_bounds_w),
        ],
        dim="vertex",
    )
    lat_vertices = xr.concat(
        [
            _add_small_rand(lat_bounds_s),
            _add_small_rand(lat_bounds_n),
            _add_small_rand(lat_bounds_n),
            _add_small_rand(lat_bounds_s),
        ],
        dim="vertex",
    )

    ds = xr.Dataset({"base": tr})

    dataset_coords = dict(
        lon=lon,
        lat=lat,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        lon_vertices=lon_vertices,
        lat_vertices=lat_vertices,
    )

    if z_axis:
        dataset_coords["lev_bounds"] = lev_bounds

    ds = ds.assign_coords(dataset_coords)
    ds.attrs["source_id"] = "test_model"
    ds.attrs["grid_label"] = grid_label
    ds.attrs["variable_id"] = "base"
    return ds


def test_parse_bounds_vertex():
    lon_b = xr.DataArray(np.array([0, 1, 2, 3]), dims=["vertex"])
    lat_b = xr.DataArray(np.array([10, 11, 12, 13]), dims=["vertex"])

    data = np.random.rand(4)

    da = xr.DataArray(
        data, dims=["vertex"], coords={"lon_vertices": lon_b, "lat_vertices": lat_b}
    )
    test = _parse_bounds_vertex(da, "vertex", position=[0, 3])
    print(test)
    expected = (da.isel(vertex=0).load().data, da.isel(vertex=3).load().data)
    print(expected)
    assert test == expected


def test_interp_vertex_to_bounds():
    da = xr.DataArray(np.arange(4), dims=["vertex"])
    # test interp on the y axis
    expected = xr.DataArray(np.array([1.5, 1.5]), dims=["bnds"])
    xr.testing.assert_equal(_interp_vertex_to_bounds(da, "y"), expected)
    # test interp on the x axis
    expected = xr.DataArray(np.array([0.5, 2.5]), dims=["bnds"])
    xr.testing.assert_equal(_interp_vertex_to_bounds(da, "x"), expected)


def test_distance_deg():
    lon0, lat0, lon1, lat1 = 120, 30, 121, 31
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    assert delta_lon == 1.0
    assert delta_lat == 1.0

    lon0, lat0, lon1, lat1 = 360, 30, 1, 31
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    assert delta_lon == 1.0
    assert delta_lat == 1.0

    lon0, lat0, lon1, lat1 = 300, 30, 301, 30.09
    delta_lon, delta_lat = distance_deg(lon0, lat0, lon1, lat1)
    assert delta_lon == 1.0
    assert delta_lat == 0.0


@pytest.mark.parametrize("lon", [0, 90, 120])
@pytest.mark.parametrize("lat", [0, 10, 45])
def test_distance(lon, lat):
    Re = 6.378e6
    # test straight lat line
    lon0, lat0, lon1, lat1 = lon, lat, lon, lat + 1
    dist = distance(lon0, lat0, lon1, lat1)
    np.testing.assert_allclose(dist, Re * (np.pi * 1.0 / 180))

    # test straight lon line
    lon0, lat0, lon1, lat1 = lon, lat, lon + 1, lat
    dist = distance(lon0, lat0, lon1, lat1)
    np.testing.assert_allclose(
        dist, Re * (np.pi * 1.0 / 180) * np.cos(np.pi * lat0 / 180)
    )


# TODO: inner and outer (needs to be implemented in xgcm autogenerate first)
@pytest.mark.parametrize("xshift", ["left", "right"])
@pytest.mark.parametrize("yshift", ["left", "right"])
@pytest.mark.parametrize("z_axis", [True, False])
def test_recreate_metrics(xshift, yshift, z_axis):
    # reconstruct all the metrics by hand and compare to inferred output

    # * For now this is a regular lon lat grid. Might need to add some tests for more complex grids.
    # Then again. This will not do a great job for those....

    # create test dataset
    ds = _test_data(z_axis=z_axis)

    # TODO: generalize so this also works with e.g. zonal average sections (which dont have a X axis)
    coord_dict = {"X": "x", "Y": "y"}
    if z_axis:
        coord_dict["Z"] = "lev"

    ds_full = generate_grid_ds(
        ds,
        coord_dict,
        position={"X": ("center", xshift), "Y": ("center", yshift)},
    )

    grid = Grid(ds_full)

    ds_metrics, metrics_dict = recreate_metrics(ds_full, grid)

    if z_axis:
        # Check that the bound values are intact (previously those got alterd due to unexpected behaviour of .assign_coords())
        assert "bnds" in ds_metrics.lev_bounds.dims

    # compute the more complex metrics (I could wrap this into a function I guess?)
    lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(ds.lon.load(), xshift)
    lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(ds.lat.load(), xshift)
    dx_gx_expected = distance(lon0, lat0, lon1, lat1)

    lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(ds.lon.load(), yshift)
    lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(ds.lat.load(), yshift)
    dy_gy_expected = distance(lon0, lat0, lon1, lat1)

    # corner metrics
    # dx
    if yshift == "left":
        # dx
        lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
            _interp_vertex_to_bounds(ds_metrics.lon_vertices, "y").isel(bnds=0),
            xshift,
        )
        lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
            ds_metrics.lat_bounds.isel(bnds=0), xshift
        )
    elif yshift == "right":
        lon0, lon1 = grid.axes["X"]._get_neighbor_data_pairs(
            _interp_vertex_to_bounds(ds_metrics.lon_vertices, "y").isel(bnds=1),
            xshift,
        )
        lat0, lat1 = grid.axes["X"]._get_neighbor_data_pairs(
            ds_metrics.lat_bounds.isel(bnds=1), xshift
        )
    dx_gxgy_expected = distance(lon0, lat0, lon1, lat1)

    # dy
    if xshift == "left":
        # dx
        lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
            _interp_vertex_to_bounds(ds_metrics.lat_vertices, "x").isel(bnds=0),
            yshift,
        )
        lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
            ds_metrics.lon_bounds.isel(bnds=0), yshift
        )
    elif xshift == "right":
        lat0, lat1 = grid.axes["Y"]._get_neighbor_data_pairs(
            _interp_vertex_to_bounds(ds_metrics.lat_vertices, "x").isel(bnds=1),
            yshift,
        )
        lon0, lon1 = grid.axes["Y"]._get_neighbor_data_pairs(
            ds_metrics.lon_bounds.isel(bnds=1), yshift
        )
    dy_gxgy_expected = distance(lon0, lat0, lon1, lat1)

    if xshift == "left":
        vertex_points = [0, 1]
    else:
        vertex_points = [2, 3]
    lon0, lon1 = (
        ds_metrics.lon_vertices.isel(vertex=vertex_points[0]),
        ds_metrics.lon_vertices.isel(vertex=vertex_points[1]),
    )
    lat0, lat1 = (
        ds_metrics.lat_vertices.isel(vertex=vertex_points[0]),
        ds_metrics.lat_vertices.isel(vertex=vertex_points[1]),
    )
    dy_gx_expected = distance(lon0, lat0, lon1, lat1)

    if yshift == "left":
        vertex_points = [0, 3]
    else:
        vertex_points = [1, 2]
    lon0, lon1 = (
        ds_metrics.lon_vertices.isel(vertex=vertex_points[0]),
        ds_metrics.lon_vertices.isel(vertex=vertex_points[1]),
    )
    lat0, lat1 = (
        ds_metrics.lat_vertices.isel(vertex=vertex_points[0]),
        ds_metrics.lat_vertices.isel(vertex=vertex_points[1]),
    )
    dx_gy_expected = distance(lon0, lat0, lon1, lat1)

    if z_axis:
        dz_t_expected = ds.lev_bounds.diff("bnds").squeeze().data
    else:
        dz_t_expected = None

    for var, expected in [
        ("dz_t", dz_t_expected),
        (
            "dx_t",
            distance(
                ds_metrics.lon_bounds.isel(bnds=0).data,
                ds_metrics.lat.data,
                ds_metrics.lon_bounds.isel(bnds=1).data,
                ds_metrics.lat.data,
            ),
        ),
        (
            "dy_t",
            distance(
                ds_metrics.lon.data,
                ds_metrics.lat_bounds.isel(bnds=0).data,
                ds_metrics.lon.data,
                ds_metrics.lat_bounds.isel(bnds=1).data,
            ),
        ),
        ("dx_gx", dx_gx_expected),
        ("dy_gy", dy_gy_expected),
        ("dy_gx", dy_gx_expected),
        ("dx_gy", dx_gy_expected),
        ("dy_gxgy", dy_gxgy_expected),
        ("dx_gxgy", dx_gxgy_expected),
    ]:
        if expected is not None:
            print(var)
            control = ds_metrics[var].data
            if expected.shape != control.shape:
                control = control.T
            np.testing.assert_allclose(control, expected)

    if z_axis:
        assert set(["X", "Y", "Z"]).issubset(set(metrics_dict.keys()))
    else:
        assert set(["X", "Y"]).issubset(set(metrics_dict.keys()))
        assert "Z" not in list(metrics_dict.keys())


# TODO: inner and outer (needs to be implemented in xgcm autogenerate first)
@pytest.mark.parametrize("xshift", ["left", "center", "right"])
@pytest.mark.parametrize("yshift", ["left", "center", "right"])
def test_detect_shift(xshift, yshift):
    # create base dataset (tracer)
    ds_base = _test_data()

    # create the maybe shifted dataset
    ds = ds_base.copy()
    if xshift == "left":
        ds["lon"] = ds["lon"] - 0.5
    elif xshift == "right":
        ds["lon"] = ds["lon"] + 0.5

    if yshift == "left":
        ds["lat"] = ds["lat"] - 0.5
    elif yshift == "right":
        ds["lat"] = ds["lat"] + 0.5
    assert detect_shift(ds_base, ds, "X") == xshift
    assert detect_shift(ds_base, ds, "Y") == yshift

    # repeat with very small shifts (these should not be detected)
    ds = ds_base.copy()
    if xshift == "left":
        ds["lon"] = ds["lon"] - 0.05
    elif xshift == "right":
        ds["lon"] = ds["lon"] + 0.05

    if yshift == "left":
        ds["lat"] = ds["lat"] - 0.05
    elif yshift == "right":
        ds["lat"] = ds["lat"] + 0.05
    assert detect_shift(ds_base, ds, "X") == "center"
    assert detect_shift(ds_base, ds, "Y") == "center"


@pytest.mark.parametrize("xshift", ["left", "right"])
@pytest.mark.parametrize("yshift", ["left", "right"])
@pytest.mark.parametrize("grid_label", ["gr", "gn"])
def test_create_full_grid(xshift, yshift, grid_label):
    ds_base = _test_data(grid_label=grid_label)
    grid_dict = {"test_model": {grid_label: {"axis_shift": {"X": xshift, "Y": yshift}}}}
    # TODO: This should be specific to the grid_label: e.g grid_dict = {'model':{'gr':{'axis_shift':{'X':'left}}}}

    ds_full = create_full_grid(ds_base, grid_dict=grid_dict)

    shift_dict = {"left": -0.5, "right": 0.5}

    assert ds_full["x"].attrs["axis"] == "X"
    assert ds_full["x_" + xshift].attrs["axis"] == "X"
    assert ds_full["x_" + xshift].attrs["c_grid_axis_shift"] == shift_dict[xshift]
    assert ds_full["y"].attrs["axis"] == "Y"
    assert ds_full["y_" + yshift].attrs["axis"] == "Y"
    assert ds_full["y_" + yshift].attrs["c_grid_axis_shift"] == shift_dict[yshift]
    # TODO: integrate the vertical
    # assert ds_full["lev"].attrs["axis"] == "Z"

    # I might want to loosen this later and switch to a uniform naming
    # E.g. use x_g for the x dimension on the x gridface, no matter if its left or right...
    # TODO: Check upstream in xgcm
    # Once that is done I
    assert "x_" + xshift in ds_full.dims
    assert "y_" + yshift in ds_full.dims

    # test error handling
    with pytest.warns(UserWarning):
        ds_none = create_full_grid(
            ds_base, grid_dict=None
        )  # the synthetic dataset is not in the default dict.
    assert ds_none is None


@pytest.mark.parametrize("recalculate_metrics", [True, False])
@pytest.mark.parametrize("xshift", ["left", "right"])
@pytest.mark.parametrize("yshift", ["left", "right"])
@pytest.mark.parametrize("grid_label", ["gr", "gn"])
def test_combine_staggered_grid(recalculate_metrics, xshift, yshift, grid_label):
    ds_base = _test_data(grid_label=grid_label)

    # create the maybe shifted dataset
    ds = ds_base.copy()
    ds = ds.rename({"base": "other"})
    ds.attrs["variable_id"] = "other"
    if xshift == "left":
        ds["lon"] = ds["lon"] - 0.5
    elif xshift == "right":
        ds["lon"] = ds["lon"] + 0.5

    if yshift == "left":
        ds["lat"] = ds["lat"] - 0.5
    elif yshift == "right":
        ds["lat"] = ds["lat"] + 0.5
    grid_dict = {"test_model": {grid_label: {"axis_shift": {"X": xshift, "Y": yshift}}}}

    for other_ds in [ds, [ds]]:
        grid, ds_combined = combine_staggered_grid(
            ds_base,
            other_ds,
            grid_dict=grid_dict,
            recalculate_metrics=recalculate_metrics,
        )

        for axis, shift in zip(["X", "Y"], [xshift, yshift]):
            # make sure the correct dim is in the added dataset
            assert grid.axes[axis].coords[shift] in ds_combined["other"].dims
            # and also that none of the other are in there
            assert all(
                [
                    di not in ds_combined["other"].dims
                    for dd, di in grid.axes[axis].coords.items()
                    if dd != shift
                ]
            )
        # check if metrics are correctly parsed
        if recalculate_metrics:
            for axis in ["X", "Y"]:
                for metric in ["_t", "_gx", "_gy", "_gxgy"]:
                    assert f"d{axis.lower()}{metric}" in list(ds_combined.coords)

    # Test error handling
    with pytest.warns(UserWarning):
        grid_none, ds_combined_none = combine_staggered_grid(
            ds_base,
            ds,
            grid_dict=None,
            recalculate_metrics=recalculate_metrics,
        )
    assert ds_combined_none is None
    assert grid_none is None
