from functools import wraps

import numpy as np
import pandas as pd
import torch

from hydromtl.data.cache.cache_base import DataSourceCache
from hydromtl.data.loader.data_scalers import wrap_t_s_dict
from hydromtl.data.source.data_base import DataSourceBase
from hydromtl.utils.hydro_utils import check_np_array_nan, t_range_days


@check_np_array_nan
def read_yxc(data_source: DataSourceBase, data_params: dict, loader_type: str) -> tuple:
    """
    Read output, dynamic inputs and static inputs

    Parameters
    ----------
    data_source
        object for reading source data
    data_params
        parameters for reading source data
    loader_type
        train, vaild or test

    Returns
    -------
    tuple [np.array]
        data_flow, data_forcing, data_attr
    """
    t_s_dict = wrap_t_s_dict(data_source, data_params, loader_type)
    basins_id = t_s_dict["sites_id"]
    t_range_list = t_s_dict["t_final_range"]
    target_cols = data_params["target_cols"]
    relevant_cols = data_params["relevant_cols"]
    constant_cols = data_params["constant_cols"]
    cache_read = data_params["cache_read"]
    other_cols = data_params.get("other_cols")
    if cache_read:
        # Don't wanna the cache impact the implemention of data_sources' read_xxx functions
        # Hence, here we follow "Convention over configuration", and set the cache files' name in DataSourceCache
        data_source_cache = DataSourceCache(
            data_params["cache_path"], loader_type, data_source
        )
        caches = data_source_cache.load_data_source()
        if data_params["vars_data_mask"] is not None:
            data_flow = _mask_flow(
                caches[0],
                data_params["vars_data_mask"],
                basins_id,
                t_range_list,
                target_cols,
            )
        else:
            data_flow = caches[0]
        data_dict = caches[3]
        # judge if the configs are correct
        if not (
            basins_id == data_dict[data_source_cache.key_sites]
            and t_range_list == data_dict[data_source_cache.key_t_range]
            and target_cols == data_dict[data_source_cache.key_target_cols]
            and relevant_cols == data_dict[data_source_cache.key_relevant_cols]
            and constant_cols == data_dict[data_source_cache.key_constant_cols]
        ):
            raise RuntimeError(
                "You chose a wrong cache, please set cache_write=1 to get correct cache or just set cache_read=0"
            )
        if other_cols is not None:
            for key, value in other_cols.items():
                assert value == data_dict[data_source_cache.key_other_cols][key]
            return data_flow, caches[1], caches[2], caches[4]
        return data_flow, caches[1], caches[2]
    if "relevant_types" in data_params:
        forcing_type = data_params["relevant_types"][0]
    else:
        forcing_type = None
    # read streamflow
    data_flow = data_source.read_target_cols(basins_id, t_range_list, target_cols)
    if data_params["vars_data_mask"] is not None:
        data_flow = _mask_flow(
            data_flow,
            data_params["vars_data_mask"],
            basins_id,
            t_range_list,
            target_cols,
        )
    # read forcing
    data_forcing = data_source.read_relevant_cols(
        basins_id, t_range_list, relevant_cols, forcing_type=forcing_type
    )
    # read attributes
    data_attr = data_source.read_constant_cols(basins_id, constant_cols)
    if other_cols is not None:
        # read other data
        data_other = data_source.read_other_cols(basins_id, other_cols=other_cols)
        return data_flow, data_forcing, data_attr, data_other
    return data_flow, data_forcing, data_attr


def _mask_flow(
    data_flow,
    mask,
    basins_id,
    t_range_list,
    target_cols,
):
    """_summary_

    Parameters
    ----------
    data_flow : np.array
        _description_
    mask : dict
        sometimes we assume some data is NaN even they literally are not,
        so we need to mask them, default is None meaning no mask
    basins_id : list
        the basins' ids, used to judge which basins to mask
    t_range_list : list
        the time range, such as ["1990-01-01","2000-01-01"], used to judge which time to mask
    target_cols : list
        the target columns, used to judge which columns to mask

    Returns
    -------
    _type_
        _description_
    """
    basin_id_mask_file = mask["basin_id_mask"]
    if isinstance(basin_id_mask_file, str):
        basin_id_mask = pd.read_csv(basin_id_mask_file, dtype={"GAGE_ID": str})[
            "GAGE_ID"
        ].values.tolist()
    else:
        basin_id_mask = basin_id_mask_file
    t_range_mask = mask["t_range_mask"]
    target_cols_mask = mask["target_cols_mask"]
    # which columns to mask
    basin_id_mask_idx = [basins_id.index(basin_id) for basin_id in basin_id_mask]
    all_t_list = t_range_days(t_range_list).tolist()
    mask_t_list = t_range_days(t_range_mask).tolist()
    t_range_mask_idx = [all_t_list.index(t_) for t_ in mask_t_list if t_ in all_t_list]
    target_cols_mask_idx = [target_cols.index(col) for col in target_cols_mask]
    # Mask the data_flow array
    for basin_idx in basin_id_mask_idx:
        for t_idx in t_range_mask_idx:
            for col_idx in target_cols_mask_idx:
                data_flow[basin_idx, t_idx, col_idx] = np.nan
    return data_flow


def check_data_loader(func):
    """
    Check if the data loader will load an input and output with NaN.
    If NaN exist in inputs, raise an Error;
    If all elements in output are NaN, also raise an Error

    Parameters
    ----------
    func
        function to run

    Returns
    -------
    function(*args, **kwargs)
        the wrapper
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        for a_tensor in result[:-1]:
            if torch.isnan(a_tensor).any():
                raise ValueError(
                    "We don't support an input with NaN value for deep learning models;\n"
                    "Please check your input data"
                )
        if torch.isnan(result[-1]).all():
            raise ValueError(
                "We don't support an output with all NaN value for deep learning models;\n"
                "Please check your output data"
            )
        return result

    return wrapper
