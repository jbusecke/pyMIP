import functools
import inspect
import warnings

from typing import List, Mapping

import numpy as np
import xarray as xr

from xmip.utils import _key_from_attrs, _maybe_make_list, cmip6_dataset_id


try:
    import xesmf
except ImportError:
    xesmf = None


# define the attrs that are needed to get an 'exact' match
EXACT_ATTRS = [
    "source_id",
    "grid_label",
    "experiment_id",
    "table_id",
    # "version", # for testing
    "variant_label",
    "variable_id",
]


def _match_attrs(ds_a, ds_b, match_attrs):
    """returns the number of matched attrs between two datasets"""
    try:
        n_match = sum([ds_a.attrs[i] == ds_b.attrs[i] for i in match_attrs])
        return n_match
    except KeyError:
        raise ValueError(
            f"Cannot match datasets because at least one of the datasets does not contain all attributes [{match_attrs}]."
        )


def _match_datasets(ds, ds_dict, match_attrs, pop=True, nomatch="ignore", unique=False):
    """Find all datasets in a dictionary of datasets that have a matching set
    of attributes and return a list of datasets for merging/concatting.
    Optionally remove the matching datasets from the input dict.
    """
    datasets = [ds]
    keys = list(ds_dict.keys())
    for k in keys:
        if _match_attrs(ds, ds_dict[k], match_attrs) == len(match_attrs):
            if pop:
                ds_matched = ds_dict.pop(k)
            else:
                ds_matched = ds_dict[k]
            # preserve the original dictionary key of the chosen dataset in attribute.
            ds_matched.attrs["original_key"] = k
            datasets.append(ds_matched)
    if len(datasets) > 2:
        if unique:
            raise ValueError(
                f"Found more than one matching dataset for {cmip6_dataset_id(ds)}. Pass `unique=False` to ignore this."
            )
    nomatch_msg = f"Could not find a matching dataset for {cmip6_dataset_id(ds)}"
    if len(datasets) < 2:
        if nomatch == "ignore":
            pass
        elif nomatch == "warn":
            warnings.warn(nomatch_msg)
        elif nomatch == "raise":
            raise RuntimeError(nomatch_msg)
        else:
            # Could this be done in an annotation? Or some other built in way to do this?
            raise ValueError(
                f"Invalid input ({nomatch}) for `nomatch`, should be `ignore`, `warn`, or `raise`"
            )
    return datasets


def _prune_match_attrs_to_available(
    match_attrs: List[str], ds_dict: Mapping[str, xr.Dataset]
) -> List[str]:
    """prune a set of attrs to only the ones available in every dataset"""
    missing_match_attrs = []
    for ma in match_attrs:
        if all([ma not in ds.attrs.keys() for ds in ds_dict.values()]):
            missing_match_attrs.append(ma)

    if len(missing_match_attrs) > 0:
        warnings.warn(
            f"Match attributes {missing_match_attrs} not found in any of the datasets. \
        This can happen when several combination functions are used and attributes are removed during merging. \
        Double check the results."
        )
        pruned_match_attrs = [ma for ma in match_attrs if ma not in missing_match_attrs]
        return pruned_match_attrs
    else:
        return match_attrs


def combine_datasets(
    ds_dict,
    combine_func,
    combine_func_args=(),
    combine_func_kwargs={},
    match_attrs=EXACT_ATTRS,
):
    """General combination function to combine datasets within a dictionary according to their matching attributes.
    This function provides maximum flexibility, but can be somewhat complex to set up. The postprocessing module provided several
    convenience wrappers like `merge_variables`, `concat_members`, etc.

    Parameters
    ----------
    ds_dict : [type]
        [description]
    combine_func : [type]
        [description]
    combine_func_args : tuple, optional
        [description], by default ()
    combine_func_kwargs : dict, optional
        [description], by default {}
    match_attrs : [type], optional
        [description], by default exact_attrs

    Returns
    -------
    [type]
        [description]
    """
    # make a copy of the input dict, so it is not modified outside of the function
    # ? Not sure this is always desired.
    ds_dict = {k: v for k, v in ds_dict.items()}
    ds_dict_combined = {}

    # Check each of the matching attributes. If attr is not present in any of
    # the input datasets drop from match_attrs and warn
    match_attrs = _prune_match_attrs_to_available(match_attrs, ds_dict)

    while len(ds_dict) > 0:
        # The order does not matter here, so just choose the first key
        k = list(ds_dict.keys())[0]
        ds = ds_dict.pop(k)
        matched_datasets = _match_datasets(ds, ds_dict, match_attrs, pop=True)

        # create new dict key
        new_k = _key_from_attrs(ds, match_attrs, sep=".")

        # for now Ill hardcode the merging. I have been thinking if I should
        # just pass a general `func` to deal with a list of datasets. The user could pass custom stuff
        # And I think that way I can generalize all/most of the `member` level combine functions.
        # Well for now, lets do it the manual way.
        try:
            ds_combined = combine_func(
                matched_datasets, *combine_func_args, **combine_func_kwargs
            )
            ds_dict_combined[new_k] = ds_combined
        except Exception as e:
            warnings.warn(f"{cmip6_dataset_id(ds)} failed to combine with :{e}")

    return ds_dict_combined


# Convenience Wrappers for `combine_datasets`. Less flexible but easier to use and understand.
def merge_variables(
    ds_dict,
    merge_kwargs={},
):
    """Given a dictionary of datasets, this function merges all available data variables (given in seperate datasets) into a single dataset.
    CAUTION: This assumes that all variables are on the same staggered grid position. If you are working with data on the cell edges,
    this function will disregard that information. Use the grids module instead to get an accurate staggered grid representation.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray datasets, that need matching metrics.
    merge_kwargs : dict
        Optional arguments passed to xr.merge.

    Returns
    -------
    dict
        A new dict of xr.Datasets with all datasets from `ds_dict`, but with merged variables and adjusted keys.

    """
    # An exact match is only possible if all attrs in `EXACT_ATTRS` are in every dataset

    match_attrs = [ma for ma in EXACT_ATTRS if ma not in ["variable_id"]]

    # set defaults
    merge_kwargs.setdefault("compat", "override")
    merge_kwargs.setdefault("join", "exact")  # if the size differs throw an error.
    merge_kwargs.setdefault(
        "combine_attrs", "drop_conflicts"
    )  # if the size differs throw an error. Requires xarray >=0.17.0

    return combine_datasets(
        ds_dict, xr.merge, combine_func_kwargs=merge_kwargs, match_attrs=match_attrs
    )


def _construct_and_promote_member_id(ds):
    # construct member_id according to https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk/edit
    sub_experiment_id = ds.attrs.get("sub_experiment_id", "none")
    variant_label = ds.attrs[
        "variant_label"
    ]  # if this should fail in the future, we could build an error/warning in here
    if sub_experiment_id == "none":
        member_id = variant_label
    else:
        member_id = f"{sub_experiment_id}-{variant_label}"

    if "member_id" not in ds.dims:
        ds = ds.expand_dims({"member_id": [member_id]})
    else:
        existing_member_id = ds.member_id.data[0]
        if not existing_member_id == member_id:
            warnings.warn(
                f"{cmip6_dataset_id(ds)} already contained a member_id ({existing_member_id}) but this is different from the reconstructed value ({member_id}). The existing value is not modified, but we recommend checking the input.",
                UserWarning,
            )
    return ds


def concat_members(
    ds_dict,
    concat_kwargs={},
):
    """Given a dictionary of datasets, this function merges all available ensemble members
    (given in seperate datasets) into a single dataset for each combination of attributes,
    like source_id, grid_label, etc. but with concatnated members.
    CAUTION: If members do not have the same dimensions (e.g. longer run time for some members),
    this can result in poor dask performance (see: https://github.com/jbusecke/xmip/issues/58)

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray datasets.
    concat_kwargs : dict
        Optional arguments passed to xr.concat.

    Returns
    -------
    dict
        A new dict of xr.Datasets with all datasets from `ds_dict`, but with concatenated members and adjusted keys.

    """
    match_attrs = [ma for ma in EXACT_ATTRS if ma not in ["variant_label"]]

    # set defaults
    concat_kwargs.setdefault(
        "combine_attrs", "drop_conflicts"
    )  # if the size differs throw an error. Requires xarray >=0.17.0

    # promote variant_label attr to coordinate, to have member_ids as coordinates

    ds_dict = {k: _construct_and_promote_member_id(ds) for k, ds in ds_dict.items()}

    return combine_datasets(
        ds_dict,
        xr.concat,
        combine_func_args=(
            ["member_id"]
        ),  # I dont like this. Its confusing to have two different dimension names
        combine_func_kwargs=concat_kwargs,
        match_attrs=match_attrs,
    )


def _concat_sorted_time(ds_list, **kwargs):
    # extract the first date
    start_dates = [str(ds.time.to_index()[0]) for ds in ds_list]
    sorted_idx = np.argsort(start_dates)
    ds_list_sorted = [ds_list[i] for i in sorted_idx]
    return xr.concat(ds_list_sorted, "time", **kwargs)


def concat_experiments(
    ds_dict,
    concat_kwargs={},
):
    """Given a dictionary of datasets, this function merges all available ensemble members
    (given in seperate datasets) into a single dataset for each combination of attributes,
    like source_id, grid_label, etc. but with concatnated members.
    CAUTION: If members do not have the same dimensions (e.g. longer run time for some members),
    this can result in poor dask performance (see: https://github.com/jbusecke/xmip/issues/58)
    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray datasets.
    concat_kwargs : dict
        Optional arguments passed to xr.concat.
    Returns
    -------
    dict
        A new dict of xr.Datasets with all datasets from `ds_dict`, but with concatenated members and adjusted keys.
    """

    match_attrs = [ma for ma in EXACT_ATTRS if ma not in ["experiment_id"]]

    # set defaults
    concat_kwargs.setdefault(
        "combine_attrs",
        "drop_conflicts",
    )  # if the size differs throw an error. Requires xarray >=0.17.0
    concat_kwargs.setdefault("compat", "override")
    concat_kwargs.setdefault("coords", "minimal")

    return combine_datasets(
        ds_dict,
        _concat_sorted_time,
        combine_func_kwargs=concat_kwargs,
        match_attrs=match_attrs,
    )


def _pick_first_member(ds_list, **kwargs):
    members = [ds.variant_label for ds in ds_list]
    first_member_idx = np.argmin(members)
    return ds_list[first_member_idx]


def pick_first_member(ddict):
    match_attrs = [ma for ma in EXACT_ATTRS if ma not in ["variant_label"]]
    return combine_datasets(
        ddict,
        _pick_first_member,
        match_attrs=match_attrs,
    )


# Matching wrapper specific to combining grid labels via interpolation with xesmf
def requires_xesmf(func):
    @functools.wraps(func)
    def wrapper_requires_xesmf(*args, **kwargs):
        # Check if xesmf is installed
        if xesmf is None:
            raise ValueError(
                "This function needs the optional package xesmf. Please install with `conda install -c conda-forge xesmf`."
            )
        return func(*args, **kwargs)

    return wrapper_requires_xesmf


def _pick_grid_label(ds_list, preferred_grid_label):
    """From a list of datasets this picks the dataset with `ds.attrs['grid_label']==preferred_label`
    if available"""
    # TODO: Implement an option to pick the highest/lowest resolution if grid label dows not match.
    matches = [ds for ds in ds_list if ds.attrs["grid_label"] == preferred_grid_label]
    if len(matches) > 0:
        return matches[0]
    else:
        return ds_list[0]


def drop_duplicate_grid_labels(ddict, preferred_grid_label):
    """Eliminate multiple grid labels for the same dataset, by dropping multiples
    with a preference for `preferred_grid_label`"""
    match_attrs = [ma for ma in EXACT_ATTRS if ma not in ["version", "grid_label"]]
    return combine_datasets(
        ddict,
        _pick_grid_label,
        combine_func_args=(preferred_grid_label,),
        match_attrs=match_attrs,
    )


def _clean_regridder(ds_source, ds_target, method, **xesmf_kwargs):
    def _clean(ds):
        # remove all unnecessary stuff for the regridding
        ds = ds.isel(time=0, lev=0, rho=0, missing_dims="ignore")
        for coord in [
            co for co in ds.coords if co not in ["lon", "lat"]
        ]:  # , "lat_bounds""lon_bounds",
            ds = ds.drop_vars(coord)

        # Ugly Hack to elminate 'seam' when regridding from gr to native grids
        # There is something in xesmf that causes problems with broadcasted regular lon/lat values
        if "gr" in ds.attrs["grid_label"]:
            # actually revert the convention and return the 1d coordinates. ?Should I change that behavior in general?
            ds = ds.assign(lon=ds.lon.isel(y=0))
            ds = ds.assign(lat=ds.lat.isel(x=0))

        # for now just eliminate the attrs here
        # I can solve this more elegantly when I parse proper cf-attributes
        for coord in ds.coords:
            ds[coord].attrs = {}
        #         ds = ds.rename({'lon_bounds':'lon_b', 'lat_bounds':'lat_b'})

        # TODO: Make this work out of the box with lon/lat bounds and method='conservative'
        # TODO: Maybe erase the need for this completely with cf-xarray
        return ds

    ds_source = _clean(ds_source)
    ds_target = _clean(ds_target)
    return xesmf.Regridder(ds_source, ds_target, method, **xesmf_kwargs)


def _regrid_to_target(ds_source, ds_target, regridder):
    ds_regridded = regridder(ds_source, keep_attrs=True)

    # remove all coordinates that involve x and y (these will be merged from the native grid dataset)
    ds_regridded = ds_regridded.reset_coords(drop=True)

    # modify attributes
    ds_regridded.attrs["grid_label"] = ds_target.attrs["grid_label"]
    # identify the variables that are regridded
    for var in ds_regridded.data_vars:
        ds_regridded[var].attrs["xmip_regrid_method"] = ds_regridded.attrs[
            "regrid_method"
        ]

    return ds_regridded


def _interpolate_combine_func(
    ds_list, target_grid_label, method, verbose=False, xesmf_kwargs={}, merge_kwargs={}
):
    grid_labels_raw = [dss.attrs["grid_label"] for dss in ds_list]
    grid_labels = np.unique(grid_labels_raw)

    # special case: All grid labels are the same but none of them is equal to target_grid_label
    if len(grid_labels) == 1:
        return xr.merge(ds_list, **merge_kwargs)
    else:
        if target_grid_label not in grid_labels:
            raise ValueError(
                f"Could not find any variable with the target_grid_label ({target_grid_label}). Found these instead: {grid_labels}"
            )
        else:
            # just take the first one with a matching grid?
            target_grid = [
                ds for ds in ds_list if ds.attrs["grid_label"] == target_grid_label
            ][0]

            # Construct a regridder for each other grid_label
            regridder_dict = {}
            if verbose:
                print(
                    f'Constructing regridders for source_id {target_grid.attrs["source_id"]} ...'
                )
            for gl in grid_labels:
                if gl != target_grid_label:
                    if verbose:
                        print(gl)
                    source_grid = [
                        dss for dss in ds_list if dss.attrs["grid_label"] == gl
                    ][0]  # again just take the first one available
                    regridder_dict[gl] = _clean_regridder(
                        source_grid, target_grid, method, **xesmf_kwargs
                    )
            if verbose:
                print("FINISHED")

            # Now regrid all datasets in the list (dont do anything if already on the target_grid)
            ds_list_new = []
            for ds_raw in ds_list:
                if ds_raw.attrs["grid_label"] != target_grid_label:
                    if verbose:
                        print(f"regridding {cmip6_dataset_id(ds_raw)}")
                    ds_regridded = _regrid_to_target(
                        ds_raw, target_grid, regridder_dict[ds_raw.attrs["grid_label"]]
                    )
                    ds_list_new.append(ds_regridded)
                else:
                    ds_list_new.append(ds_raw)

            # check that horizontal dimensions are compatible.
            xy_dimensions = [(len(ds.x), len(ds.y)) for ds in ds_list_new]
            if not all([xy == xy_dimensions[0] for xy in xy_dimensions]):
                raise ValueError(
                    f"Regridded datasets do not have the same dimensions. Found({xy_dimensions}). This will cause broadcasting problems during merge."
                )
            if verbose:
                print("Merging regridded files")
            return xr.merge(ds_list_new, **merge_kwargs)


@requires_xesmf
def interpolate_grid_label(
    ds_dict,
    target_grid_label="gn",
    method="bilinear",
    xesmf_kwargs={},
    merge_kwargs={},
    verbose=False,
):
    """Combines different grid labels via interpolation with xesmf

    Parameters
    ----------
    ds_dict : dict
        dictonary of input datasets
    target_grid_label : str, optional
        preferred grid_label value. If at least one dataset has this grid_label, otherse are interpolated to it.
        Dataset with this grid label are not modified, by default "gn"
    method : str, optional
        interpolation method for xesmf, by default "bilinear"
    xesmf_kwargs : dict, optional
        optional arguments for building xesmf regridder, by default {}
    merge_kwargs : dict, optional
        optional arguments for the merging of interpolated datasets, by default {}
    verbose : bool, optional
        print output while creating regridder, by default False

    Returns
    -------
    dict
        dictionary of combined datasets (usually will combine across different variable ids)
    """
    match_attrs = [
        ma for ma in EXACT_ATTRS if ma not in ["grid_label", "variable_id"]
    ]  # does this need to be more flexible?

    xesmf_kwargs.setdefault("ignore_degenerate", True)
    xesmf_kwargs.setdefault("periodic", True)

    merge_kwargs.setdefault("combine_attrs", "drop_conflicts")

    # first drop the datasets that might have both the target and another grid_label present
    ds_dict = drop_duplicate_grid_labels(ds_dict, target_grid_label)

    return combine_datasets(
        ds_dict,
        _interpolate_combine_func,
        combine_func_args=(
            target_grid_label,
            method,
        ),
        combine_func_kwargs={
            "xesmf_kwargs": xesmf_kwargs,
            "merge_kwargs": merge_kwargs,
        },
        match_attrs=match_attrs,
    )


# Matching wrapper specific to metric datasets


def _parse_metric(ds, metric, dim_length_conflict="error"):
    """Convenience function to parse a metric dataarry into an existing dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    metric : xr.Data
        Metric dataarray to add to `ds`
    dim_length_conflict : str
        Specifies how to handle dimensions that are not the same lenght.

    Returns
    -------
    xr.Dataset
        `ds` with `metric` added as coordinate.

    """
    dataset_id = cmip6_dataset_id(ds)

    if not isinstance(metric, xr.DataArray):
        raise ValueError(
            f"{dataset_id}:`metric` input must be xarray.DataArray. Got {type(metric)}"
        )

    if metric.name is None:
        warnings.warn(
            f"{dataset_id}:`metric` has no name. This might lead to problems down the line.",
            RuntimeWarning,
        )

    # Check dimension compatibility
    # TODO: Check if xarray has some nifty way to do this
    mismatch_dims = []
    for di in metric.dims:
        if len(ds[di]) != len(metric[di]):
            mismatch_dims.append(di)

    if len(mismatch_dims) > 0:
        metric_str = [str(di) + ":" + str(len(metric[di])) for di in mismatch_dims]
        ds_str = [str(di) + ":" + str(len(ds[di])) for di in mismatch_dims]
        msg = (
            f"{dataset_id}:`metric` dimensions {metric_str} do not match `ds` {ds_str}."
        )
        if dim_length_conflict == "error":
            raise ValueError(msg)
        elif dim_length_conflict == "align":
            warnings.warn(msg + " Aligning the data on `inner`")
            ds, metric = xr.align(ds, metric, join="inner")

    # strip all coordinates from metric
    metric_stripped = metric.reset_coords(drop=True)

    # add attributes
    metric_stripped.attrs["parsed_with"] = (
        f"xmip/postprocessing/{inspect.currentframe().f_code.co_name}"
    )
    # TODO: Get the package and module name without hardcoding.

    ds = ds.assign_coords({metric_stripped.name: metric_stripped})

    return ds


def match_metrics(
    ds_dict,
    metric_dict,
    match_variables,
    match_attrs=["source_id", "grid_label"],
    print_statistics=False,
    dim_length_conflict="error",
):
    """Given two dictionaries of datasets, this function matches metrics from `metric_dict` to
    every datasets in `ds_dict` based on a comparison of the datasets attributes.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray datasets, that need matching metrics.
    metric_dict : dict
        Dictionary of xarray datasets, which contain metrics as data_variables.
    match_variables : list
        Data variables of datasets in `metric_dict` to parse.
    match_attrs : list, optional
        Minimum dataset attributes that need to match, by default ["source_id", "grid_label"].
        Pass "exact" to only allow exact matches using all required attributes.
    print_statistics : bool, optional
        Option to print statistics about matching, by default False
    dim_length_conflict : str
        Defines the behavior when parsing metrics with non-exact matches in dimension size.
        See `parse_metric`.
    Returns
    -------
    dict
        All datasets from `ds_dict`, if match was not possible the input dataset is returned unchanged.

    """
    # metrics should never match the variable
    exact_attrs_wo_var = [ma for ma in EXACT_ATTRS if ma != "variable_id"]

    # TODO: this naming is a big weird. Basically this here is necessary to still get a 'closest match' on whichever of the full
    # set of 'exact' attrs are given on each of the datasets
    pruned_attrs = _prune_match_attrs_to_available(exact_attrs_wo_var, ds_dict)
    # and this also needs to be true for the metrics of course.
    pruned_attrs = _prune_match_attrs_to_available(pruned_attrs, metric_dict)

    match_variables = _maybe_make_list(match_variables)

    # if match is set to exact check all these attributes
    if match_attrs == "exact":
        match_attrs = exact_attrs_wo_var

    # --- set up statistics --- #
    total_datasets = 0
    exact_datasets = {va: 0 for va in match_variables}
    nonexact_datasets = {va: 0 for va in match_variables}
    nomatch_datasets = {va: 0 for va in match_variables}

    ds_dict_parsed = {}
    for k, ds in ds_dict.items():
        total_datasets += 1
        # Filter for the matching attrs
        matched_metrics = _match_datasets(ds, metric_dict, match_attrs, pop=False)

        # Filter for the variable
        for mv in match_variables:
            matched_metric_vars = [ds for ds in matched_metrics if mv in ds.variables]

            if len(matched_metric_vars) == 0:
                warnings.warn(f"No matching metrics found for {mv}")
                nomatch_datasets[mv] += 1
            else:
                # Pick the metric with the most mathching attributes.
                # This ensures that the exact match is picked if available.

                nmatch = [
                    _match_attrs(ds, ds_match, pruned_attrs)
                    for ds_match in matched_metric_vars
                ]
                closest_match = np.argmax(nmatch)
                ds_metric = matched_metric_vars[closest_match]

                # this is a hardcoded check for time variable metrics.
                # These are very likely only valid for exact matches of runs.
                # For instance these could be cell thickness values, which cannot
                # simply be used for all runs of a model.
                exact_match = _match_attrs(ds, ds_metric, pruned_attrs) == len(
                    exact_attrs_wo_var
                )
                if "time" in ds_metric.dims and not exact_match:
                    warnings.warn(
                        "This metric had a time dimension and did not perfectly match. Not parsing anything."
                    )
                else:
                    # parse the original key attr to the dataarray
                    ds_metric[mv].attrs["original_key"] = ds_metric.attrs[
                        "original_key"
                    ]
                    ds = _parse_metric(
                        ds, ds_metric[mv], dim_length_conflict=dim_length_conflict
                    )
                    if exact_match:
                        exact_datasets[mv] += 1
                    else:
                        nonexact_datasets[mv] += 1

        ds_dict_parsed[k] = ds
    if print_statistics:
        print(
            f"Processed {total_datasets} datasets.\nExact matches:{exact_datasets}\nOther matches:{nonexact_datasets}\nNo match found:{nomatch_datasets}"
        )
    return ds_dict_parsed
