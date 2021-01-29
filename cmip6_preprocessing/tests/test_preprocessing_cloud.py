# This module tests data directly from the pangeo google cloud storage.
# Tests are meant to be more high level and also serve to document known problems (see skip statements).
import pytest
import xarray as xr
import numpy as np
from cmip6_preprocessing.tests.cloud_test_utils import (
    all_models,
    data,
    diagnose_doubles,
)
from cmip6_preprocessing.preprocessing import combined_preprocessing
from cmip6_preprocessing.grids import combine_staggered_grid

pytest.importorskip("gcsfs")

test_models = ["CESM2-FV2", "GFDL-CM4"]
# test_models = all_models()


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".

    for name in ["vi", "gl", "ei"]:

        option_value = getattr(metafunc.config.option, name)

        if isinstance(option_value, str):
            option_value = [option_value]

        if name in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(name, option_value)


print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")

## Combine the input parameters according to command line input

########################### Most basic test #########################

# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_dim_coord_values_wo_intake(request, gl, vi, ei):
    expected_failures = [
        ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
        ("AWI-ESM-1-1-LR", "thetao", "ssp585", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
        # TODO: would be nice to have a "*" matching...
        # (
        #     "GFDL-CM4",
        #     "thetao",
        #     "historical",
        #     "gn",
        # ),  # this should not fail and should trigger an xpass (I just use this for dev purposes to check
        #     # the strict option)
    ]
    spec = (request.param, vi, ei, gl)
    request.param = spec
    if request.param in expected_failures:
        request.node.add_marker(pytest.mark.xfail(strict=True))
    return request


@pytest.mark.parametrize(
    "spec_check_dim_coord_values_wo_intake", test_models, indirect=True
)
def test_check_dim_coord_values_wo_intake(
    spec_check_dim_coord_values_wo_intake,
):
    (
        source_id,
        variable_id,
        experiment_id,
        grid_label,
    ) = spec_check_dim_coord_values_wo_intake.param

    # there must be a better way to build this at the class level and then tear it down again
    # I can probably get this done with fixtures, but I dont know how atm
    ds, _ = data(source_id, variable_id, experiment_id, grid_label, False)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    ##### Check for dim duplicates
    # check all dims for duplicates
    # for di in ds.dims:
    # for now only test a subset of the dims. TODO: Add the bounds once they
    # are cleaned up.
    for di in ["x", "y", "lev", "time"]:
        if di in ds.dims:
            diagnose_doubles(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))
            if di != "time":  # these tests do not make sense for decoded time
                assert np.all(~np.isnan(ds[di]))
                assert np.all(ds[di].diff(di) >= 0)

    assert ds.lon.min().load() >= 0
    assert ds.lon.max().load() <= 360
    if "lon_bounds" in ds.variables:
        assert ds.lon_bounds.min().load() >= 0
        assert ds.lon_bounds.max().load() <= 360
    assert ds.lat.min().load() >= -90
    assert ds.lat.max().load() <= 90
    # make sure lon and lat are 2d
    assert len(ds.lon.shape) == 2
    assert len(ds.lat.shape) == 2


# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_dim_coord_values(request, gl, vi, ei):
    expected_failures = [
        ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
        ("AWI-ESM-1-1-LR", "thetao", "ssp585", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
        # TODO: would be nice to have a "*" matching...
        (
            "IPSL-CM6A-LR",
            "thetao",
            "historical",
            "gn",
        ),  # IPSL has an issue with `lev` dims concatting
        ("IPSL-CM6A-LR", "o2", "historical", "gn"),
        ("NorESM2-MM", "thetao", "historical", "gn"),
        ("NorESM2-MM", "thetao", "historical", "gr"),
    ]
    spec = (request.param, vi, ei, gl)
    request.param = spec
    if request.param in expected_failures:
        request.node.add_marker(pytest.mark.xfail(strict=True))
    return request


@pytest.mark.parametrize("spec_check_dim_coord_values", test_models, indirect=True)
def test_check_dim_coord_values(
    spec_check_dim_coord_values,
):
    (
        source_id,
        variable_id,
        experiment_id,
        grid_label,
    ) = spec_check_dim_coord_values.param
    # there must be a better way to build this at the class level and then tear it down again
    # I can probably get this done with fixtures, but I dont know how atm
    ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    ##### Check for dim duplicates
    # check all dims for duplicates
    # for di in ds.dims:
    # for now only test a subset of the dims. TODO: Add the bounds once they
    # are cleaned up.
    for di in ["x", "y", "lev", "time"]:
        if di in ds.dims:
            diagnose_doubles(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))
            if di != "time":  # these tests do not make sense for decoded time
                assert np.all(~np.isnan(ds[di]))
                assert np.all(ds[di].diff(di) >= 0)

    assert ds.lon.min().load() >= 0
    assert ds.lon.max().load() <= 360
    if "lon_bounds" in ds.variables:
        assert ds.lon_bounds.min().load() >= 0
        assert ds.lon_bounds.max().load() <= 360
    assert ds.lat.min().load() >= -90
    assert ds.lat.max().load() <= 90
    # make sure lon and lat are 2d
    assert len(ds.lon.shape) == 2
    assert len(ds.lat.shape) == 2


############################### Specific Bound Coords Test ###############################


# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_bounds_verticies(request, gl, vi, ei):
    expected_failures = [
        ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
        ("AWI-ESM-1-1-MR", "thetao", "historical", "gn"),
        ("AWI-ESM-1-1-MR", "thetao", "ssp585", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
        ("CESM2-FV2", "thetao", "historical", "gn"),
        ("FGOALS-f3-L", "thetao", "historical", "gn"),
        ("FGOALS-f3-L", "thetao", "ssp585", "gn"),
        ("FGOALS-g3", "thetao", "historical", "gn"),
        ("FGOALS-g3", "thetao", "ssp585", "gn"),
        ("NorESM2-MM", "thetao", "historical", "gn"),
        ("NorESM2-MM", "thetao", "historical", "gr"),
        ("IPSL-CM6A-LR", "thetao", "historical", "gn"),
        ("IPSL-CM6A-LR", "o2", "historical", "gn"),
    ]
    spec = (request.param, vi, ei, gl)
    request.param = spec
    if request.param in expected_failures:
        request.node.add_marker(pytest.mark.xfail(strict=True))
    return request


@pytest.mark.parametrize("spec_check_bounds_verticies", test_models, indirect=True)
def test_check_bounds_verticies(
    spec_check_bounds_verticies,
):
    (
        source_id,
        variable_id,
        experiment_id,
        grid_label,
    ) = spec_check_bounds_verticies.param
    ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    if "vertex" in ds.dims:
        np.testing.assert_allclose(ds.vertex.data, np.arange(4))

    ####Check for existing bounds and verticies
    for co in ["lon_bounds", "lat_bounds", "lon_verticies", "lat_verticies"]:
        assert co in ds.coords
        # make sure that all other dims are eliminated from the bounds.
        assert (set(ds[co].dims) - set(["bnds", "vertex"])) == set(["x", "y"])

    #### Check the order of the vertex
    # Ill only check these south of the Arctic for now. Up there
    # things are still weird.

    test_ds = ds.sel(y=slice(-40, 40))

    vertex_lon_diff1 = test_ds.lon_verticies.isel(
        vertex=3
    ) - test_ds.lon_verticies.isel(vertex=0)
    vertex_lon_diff2 = test_ds.lon_verticies.isel(
        vertex=2
    ) - test_ds.lon_verticies.isel(vertex=1)
    vertex_lat_diff1 = test_ds.lat_verticies.isel(
        vertex=1
    ) - test_ds.lat_verticies.isel(vertex=0)
    vertex_lat_diff2 = test_ds.lat_verticies.isel(
        vertex=2
    ) - test_ds.lat_verticies.isel(vertex=3)
    for vertex_diff in [vertex_lon_diff1, vertex_lon_diff2]:
        assert (vertex_diff <= 0).sum() <= (3 * len(vertex_diff.y))
        # allowing for a few rows to be negative

    for vertex_diff in [vertex_lat_diff1, vertex_lat_diff2]:
        assert (vertex_diff <= 0).sum() <= (5 * len(vertex_diff.x))
        # allowing for a few rows to be negative
    # This is just to make sure that not the majority of values is negative or zero.

    # Same for the bounds:
    lon_diffs = test_ds.lon_bounds.diff("bnds")
    lat_diffs = test_ds.lat_bounds.diff("bnds")

    assert (lon_diffs <= 0).sum() <= (5 * len(lon_diffs.y))
    assert (lat_diffs <= 0).sum() <= (5 * len(lat_diffs.y))


################################# xgcm grid specific tests ########################################


# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_grid(request, gl, vi, ei):
    expected_failures = [
        ("AWI-ESM-1-1-LR", "thetao", "historical", "gn"),
        ("AWI-ESM-1-1-MR", "thetao", "historical", "gn"),
        ("AWI-ESM-1-1-MR", "thetao", "ssp585", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "historical", "gn"),
        ("AWI-CM-1-1-MR", "thetao", "ssp585", "gn"),
        ("CESM2-FV2", "thetao", "historical", "gn"),
        ("CMCC-CM2-SR5", "thetao", "historical", "gn"),
        ("CMCC-CM2-SR5", "thetao", "ssp585", "gn"),
        ("FGOALS-f3-L", "thetao", "historical", "gn"),
        ("FGOALS-f3-L", "thetao", "ssp585", "gn"),
        ("FGOALS-g3", "thetao", "historical", "gn"),
        ("FGOALS-g3", "thetao", "ssp585", "gn"),
        ("MPI-ESM-1-2-HAM", "thetao", "historical", "gn"),
        ("MPI-ESM-1-2-HAM", "o2", "historical", "gn"),
        ("NorESM2-MM", "thetao", "historical", "gn"),
        ("NorESM2-MM", "thetao", "historical", "gr"),
        ("IPSL-CM6A-LR", "thetao", "historical", "gn"),
        ("IPSL-CM6A-LR", "o2", "historical", "gn"),
    ]
    spec = (request.param, vi, ei, gl)
    request.param = spec
    if request.param in expected_failures:
        request.node.add_marker(pytest.mark.xfail(strict=True))
    return request


@pytest.mark.parametrize("spec_check_grid", test_models, indirect=True)
def test_check_grid(
    spec_check_grid,
):
    source_id, variable_id, experiment_id, grid_label = spec_check_grid.param

    ds, cat = data(source_id, variable_id, experiment_id, grid_label, True)

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    # This is just a rudimentary test to see if the creation works
    staggered_grid, ds_staggered = combine_staggered_grid(ds, recalculate_metrics=True)

    print(ds_staggered)

    assert ds_staggered is not None
    #
    if "lev" in ds_staggered.dims:
        assert "bnds" in ds_staggered.lev_bounds.dims

    for axis in ["X", "Y"]:
        for metric in ["_t", "_gx", "_gy", "_gxgy"]:
            assert f"d{axis.lower()}{metric}" in list(ds_staggered.coords)
    # TODO: Include actual test to combine variables
