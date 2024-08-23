# This module tests data directly from the pangeo google cloud storage.
# Tests are meant to be more high level and also serve to document known problems (see skip statements).

import fsspec
import numpy as np
import pytest
import xarray as xr

from xmip.grids import combine_staggered_grid
from xmip.preprocessing import _desired_units, _drop_coords, combined_preprocessing
from xmip.utils import google_cmip_col, model_id_match


pytest.importorskip("gcsfs")


def diagnose_duplicates(data):
    """displays non-unique entries in data"""
    _, idx = np.unique(data, return_index=True)
    missing = np.array([i for i in np.arange(len(data)) if i not in idx])
    if len(missing) > 0:
        missing_values = data[missing]
        raise ValueError(f"Duplicate Values ({missing_values}) found")


def data(
    source_id, variable_id, experiment_id, grid_label, use_intake_esm, catalog="main"
):
    zarr_kwargs = {
        "consolidated": True,
        # "decode_times": False,
        "decode_times": True,
        "use_cftime": True,
    }

    cat = google_cmip_col(catalog=catalog).search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        # member_id="r1i1p1f1",
        table_id="Omon",
        grid_label=grid_label,
    )

    if len(cat.df["zstore"]) > 0:
        if use_intake_esm:
            ddict = cat.to_dataset_dict(
                zarr_kwargs=zarr_kwargs,
                preprocess=combined_preprocessing,
                storage_options={"token": "anon"},
            )
            _, ds = ddict.popitem()
        else:
            # debugging options
            # @charlesbluca suggested this to make this work in GHA
            # https://github.com/jbusecke/xmip/pull/62#issuecomment-741928365
            mm = fsspec.get_mapper(
                cat.df["zstore"][0]
            )  # think you can pass in storage options here as well?
            ds_raw = xr.open_zarr(mm, **zarr_kwargs)
            ds = combined_preprocessing(ds_raw)
    else:
        ds = None

    return ds, cat


def all_models():
    df = google_cmip_col().df
    all_models = df["source_id"].unique()
    all_models = tuple(np.sort(all_models))
    # all_models = tuple(["EC-Earth3"])
    return all_models


# test_models = ["CESM2-FV2", "GFDL-CM4"]
test_models = all_models()


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".

    for name in ["vi", "gl", "ei", "cat"]:
        option_value = getattr(metafunc.config.option, name)

        if isinstance(option_value, str):
            option_value = [option_value]

        if name in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(name, option_value)


# print(f"\n\n\n\n$$$$$$$ All available models: {all_models()}$$$$$$$\n\n\n\n")

# Combine the input parameters according to command line input

# --- Most basic test --- #

# Try to combine some of the failures

# We dont support these at all
not_supported_failures = [
    ("AWI-ESM-1-1-LR", "*", "*", "gn"),
    ("AWI-CM-1-1-MR", "*", "*", "gn"),
]

# basic problems when trying to concat with intake-esm
intake_concat_failures = [
    (
        "CanESM5",
        [
            "uo",
            "so",
            "thetao",
        ],
        "ssp245",
        "gn",
    ),
    (
        "CanESM5",
        ["zos"],
        [
            "ssp245",
            "ssp585",
        ],
        "gn",
    ),
    (
        "E3SM-1-0",
        ["so", "o2", "zos"],
        ["historical", "ssp585"],
        "gr",
    ),  # issues with time concatenation
    (
        "IPSL-CM6A-LR",
        ["thetao", "o2", "so"],
        "historical",
        "gn",
    ),  # IPSL has an issue with `lev` dims concatting]
    (
        "NorESM2-MM",
        ["uo", "so"],
        "historical",
        "gr",
    ),  # time concatting
    (
        "NorESM2-MM",
        ["so"],
        "historical",
        "gn",
    ),
]


# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_dim_coord_values_wo_intake(request, gl, vi, ei, cat):
    expected_failures = not_supported_failures + [
        ("FGOALS-f3-L", ["thetao"], "piControl", "gn"),
        # (
        #     "GFDL-CM4",
        #     "thetao",
        #     "historical",
        #     "gn",
        # ),  # this should not fail and should trigger an xpass (I just use this for dev purposes to check
        #     # the strict option)
    ]
    spec = (request.param, vi, ei, gl, cat)
    request.param = spec
    if model_id_match(expected_failures, request.param[0:-1]):
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
        catalog,
    ) = spec_check_dim_coord_values_wo_intake.param

    # there must be a better way to build this at the class level and then tear it down again
    # I can probably get this done with fixtures, but I dont know how atm
    ds, _ = data(
        source_id, variable_id, experiment_id, grid_label, False, catalog=catalog
    )

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    # Check for dim duplicates
    # check all dims for duplicates
    # for di in ds.dims:
    # for now only test a subset of the dims. TODO: Add the bounds once they
    # are cleaned up.
    for di in ["x", "y", "lev", "time"]:
        if di in ds.dims:
            diagnose_duplicates(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))
            if di != "time":  # these tests do not make sense for decoded time
                assert np.all(~np.isnan(ds[di]))
                assert np.all(ds[di].diff(di) >= 0)

    assert ds.lon.min().load() >= 0
    assert ds.lon.max().load() <= 360
    if "lon_bounds" in ds.variables:
        assert ds.lon_bounds.min().load() >= 0
        assert ds.lon_bounds.max().load() <= 361
    assert ds.lat.min().load() >= -90
    assert ds.lat.max().load() <= 90
    # make sure lon and lat are 2d
    assert len(ds.lon.shape) == 2
    assert len(ds.lat.shape) == 2
    for co in _drop_coords:
        if co in ds.dims:
            assert co not in ds.coords

    # Check unit conversion
    for var, expected_unit in _desired_units.items():
        if var in ds.variables:
            unit = ds[var].attrs.get("units")
            if unit:
                assert unit == expected_unit


# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_dim_coord_values(request, gl, vi, ei, cat):
    expected_failures = (
        not_supported_failures
        + intake_concat_failures
        + [
            ("NorESM2-MM", ["uo", "zos"], "historical", "gn"),
            ("NorESM2-MM", "thetao", "historical", "gn"),
            ("NorESM2-MM", "thetao", "historical", "gr"),
            ("FGOALS-f3-L", ["thetao"], "piControl", "gn"),
        ]
    )
    spec = (request.param, vi, ei, gl, cat)
    request.param = spec
    if model_id_match(expected_failures, request.param[0:-1]):
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
        catalog,
    ) = spec_check_dim_coord_values.param
    # there must be a better way to build this at the class level and then tear it down again
    # I can probably get this done with fixtures, but I dont know how atm
    ds, cat = data(
        source_id, variable_id, experiment_id, grid_label, True, catalog=catalog
    )

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    # Check for dim duplicates
    # check all dims for duplicates
    # for di in ds.dims:
    # for now only test a subset of the dims. TODO: Add the bounds once they
    # are cleaned up.
    for di in ["x", "y", "lev", "time"]:
        if di in ds.dims:
            diagnose_duplicates(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))
            if di != "time":  # these tests do not make sense for decoded time
                assert np.all(~np.isnan(ds[di]))
                assert np.all(ds[di].diff(di) >= 0)

    assert ds.lon.min().load() >= 0
    assert ds.lon.max().load() <= 360
    if "lon_bounds" in ds.variables:
        assert ds.lon_bounds.min().load() >= 0
        assert ds.lon_bounds.max().load() <= 361
    assert ds.lat.min().load() >= -90
    assert ds.lat.max().load() <= 90
    # make sure lon and lat are 2d
    assert len(ds.lon.shape) == 2
    assert len(ds.lat.shape) == 2
    for co in _drop_coords:
        if co in ds.dims:
            assert co not in ds.coords


# --- Specific Bound Coords Test -----


# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_bounds_vertices(request, gl, vi, ei, cat):
    expected_failures = (
        not_supported_failures
        + intake_concat_failures
        + [
            ("FGOALS-f3-L", ["thetao", "so", "uo", "zos"], "*", "gn"),
            ("FGOALS-g3", ["thetao", "so", "uo", "zos"], "*", "gn"),
            ("NorESM2-MM", ["thetao", "uo", "zos"], "historical", "gn"),
            ("NorESM2-MM", ["thetao", "so"], "historical", "gr"),
            ("IPSL-CM6A-LR", ["thetao", "o2"], "historical", "gn"),
            ("IITM-ESM", ["so", "uo", "thetao"], "piControl", "gn"),
            ("GFDL-CM4", "uo", "*", "gn"),
        ]
    )
    spec = (request.param, vi, ei, gl, cat)
    request.param = spec
    if model_id_match(expected_failures, request.param[0:-1]):
        request.node.add_marker(pytest.mark.xfail(strict=True))
    return request


@pytest.mark.parametrize("spec_check_bounds_vertices", test_models, indirect=True)
def test_check_bounds_vertices(spec_check_bounds_vertices):
    (
        source_id,
        variable_id,
        experiment_id,
        grid_label,
        catalog,
    ) = spec_check_bounds_vertices.param
    ds, cat = data(
        source_id, variable_id, experiment_id, grid_label, True, catalog=catalog
    )

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    if "vertex" in ds.dims:
        np.testing.assert_allclose(ds.vertex.data, np.arange(4))

    # Check for existing bounds and vertices
    for co in ["lon_bounds", "lat_bounds", "lon_vertices", "lat_vertices"]:
        assert co in ds.coords
        # make sure that all other dims are eliminated from the bounds.
        assert (set(ds[co].dims) - set(["bnds", "vertex"])) == set(["x", "y"])

    # Check the order of the vertex
    # Ill only check these south of the Arctic for now. Up there
    # things are still weird.
    test_ds = ds.where(abs(ds.lat) <= 40, drop=True)

    vertex_lon_diff1 = test_ds.lon_vertices.isel(
        vertex=3
    ) - test_ds.lon_vertices.isel(vertex=0)
    vertex_lon_diff2 = test_ds.lon_vertices.isel(
        vertex=2
    ) - test_ds.lon_vertices.isel(vertex=1)
    vertex_lat_diff1 = test_ds.lat_vertices.isel(
        vertex=1
    ) - test_ds.lat_vertices.isel(vertex=0)
    vertex_lat_diff2 = test_ds.lat_vertices.isel(
        vertex=2
    ) - test_ds.lat_vertices.isel(vertex=3)
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


# --- xgcm grid specific tests --- #
# this fixture has to be redifined every time to account for different fail cases for each test
@pytest.fixture
def spec_check_grid(request, gl, vi, ei, cat):
    expected_failures = (
        not_supported_failures
        + intake_concat_failures
        + [
            ("CMCC-ESM2", "*", "*", "gn"),
            ("CMCC-CM2-SR5", "*", "*", "gn"),
            ("CMCC-CM2-HR4", "*", "*", "gn"),
            ("FGOALS-f3-L", "*", "*", "gn"),
            ("FGOALS-g3", "*", "*", "gn"),
            ("E3SM-1-0", ["so", "thetao", "o2"], "*", "gn"),
            (
                "E3SM-1-0",
                ["zos"],
                ["historical", "ssp585", "ssp245", "ssp370", "esm-hist"],
                "gr",
            ),
            (
                "EC-Earth3-AerChem",
                ["so", "thetao", "zos"],
                ["historical", "piControl", "ssp370"],
                "gn",
            ),
            ("EC-Earth3-Veg", "*", "historical", "gr"),
            ("EC-Earth3-CC", "*", "*", "gn"),
            ("MPI-ESM-1-2-HAM", "*", "*", "gn"),
            ("NorESM2-MM", "*", "historical", "gn"),
            ("NorESM2-MM", ["thetao", "so", "uo"], "historical", "gr"),
            ("IITM-ESM", "*", "*", "gn"),
            ("GFDL-CM4", ["uo"], "*", "gn"),
            ("IPSL-CM5A2-INCA", "*", "*", "gn"),
            ("IPSL-CM6A-LR-INCA", "*", "*", "gn"),
        ]
    )
    spec = (request.param, vi, ei, gl, cat)
    request.param = spec
    if model_id_match(expected_failures, request.param[0:-1]):
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=""))
    return request


@pytest.mark.parametrize("spec_check_grid", test_models, indirect=True)
def test_check_grid(
    spec_check_grid,
):
    source_id, variable_id, experiment_id, grid_label, catalog = spec_check_grid.param

    ds, cat = data(
        source_id, variable_id, experiment_id, grid_label, True, catalog=catalog
    )

    if ds is None:
        pytest.skip(
            f"No data found for {source_id}|{variable_id}|{experiment_id}|{grid_label}"
        )

    # This is just a rudimentary test to see if the creation works
    staggered_grid, ds_staggered = combine_staggered_grid(ds, recalculate_metrics=True)

    assert ds_staggered is not None
    #
    if "lev" in ds_staggered.dims:
        assert "bnds" in ds_staggered.lev_bounds.dims

    for axis in ["X", "Y"]:
        for metric in ["_t", "_gx", "_gy", "_gxgy"]:
            assert f"d{axis.lower()}{metric}" in list(ds_staggered.coords)
    # TODO: Include actual test to combine variables
