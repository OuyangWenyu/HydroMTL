import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

import definitions
from hydromtl.models.time_model import PyTorchForecast
from hydromtl.models.evaluator import generate_predictions
from hydromtl.data.loader.dataloaders4test import TestDataModel
from hydromtl.models.training_utils import get_the_device
from hydromtl.data.data_dict import data_sources_dict
from hydromtl.data.config import get_config_file
from hydromtl.data.source_pro.data_camels_pro import CamelsPro


def get_trained_model(config):
    """Get the trained LSTM models

    Parameters
    ----------
    config : _type_
        the config file saved in an experiment directory

    Returns
    -------
    _type_
        _description_
    """
    config["model_params"]["continue_train"] = False
    epoch = config["evaluate_params"]["test_epoch"]
    weight_path = os.path.join(
        config["data_params"]["test_path"], "model_Ep" + str(epoch) + ".pth"
    )
    if not os.path.isfile(weight_path):
        weight_path = config["model_params"]["weight_path"]
    config["model_params"]["weight_path"] = weight_path
    config["data_params"]["cache_read"] = True
    config["data_params"]["cache_write"] = False
    if config["data_params"]["cache_path"] is None:
        config["data_params"]["cache_path"] = config["data_params"]["test_path"]
    data_params = config["data_params"]
    config["data_params"]["stat_dict_file"] = None
    data_source_name = data_params["data_source_name"]
    if data_source_name in ["CAMELS", "CAMELS_SERIES"]:
        # there are many different regions for CAMELS datasets
        data_source = data_sources_dict[data_source_name](
            data_params["data_path"],
            data_params["download"],
            data_params["data_region"],
        )
    else:
        data_source = data_sources_dict[data_source_name](
            data_params["data_path"], data_params["download"]
        )
    model = PyTorchForecast(config["model_params"]["model_name"], data_source, config)
    return model


def convert_to_xarray(
    data: np.ndarray, basins: list, time_range: list, key: str, dim=None
) -> xr.Dataset:
    date_range = pd.date_range(start=time_range[0], end=time_range[1], freq="1D")[:-1]
    if dim is None:
        dim = np.arange(data.shape[-1])
    return xr.Dataset(
        {key: (("station_id", "date", "dimension"), data)},
        coords={
            "station_id": basins,
            "date": date_range,
            "dimension": dim,
        },
    )


def get_cell_states(t_model: PyTorchForecast):
    """
    Do a forward pass and store the output c_n for each basin

    Parameters
    ----------
    model : PyTorchForecast
        A PyTorchForecast model containing the LSTM model and data of all basins

    Returns
    -------
    _type_
        _description_
    """
    test_dataset = TestDataModel(t_model.test_data)
    all_data = test_dataset.load_test_data()
    device = get_the_device(t_model.params["training_params"]["device"])
    data_params = t_model.params["data_params"]
    #  get cell state and transform to xarray objects
    cell_states_nc_file = os.path.join(
        data_params["test_path"],
        "cell_states.nc",
    )
    cell_states_npy_file = os.path.join(
        data_params["test_path"],
        "cell_states.npy",
    )
    if os.path.exists(cell_states_nc_file):
        cell_states = xr.open_dataset(cell_states_nc_file)
        return cell_states
    elif os.path.exists(cell_states_npy_file):
        cell_states = np.load(cell_states_npy_file)
    else:
        pred, cell_states = generate_predictions(
            t_model,
            test_dataset,
            *all_data[:-1],
            device=device,
            data_params=data_params,
            return_cell_state=True,
        )
    #  get cell state xarray objects
    basins = data_params["object_ids"]
    time_range = data_params["t_range_test"]
    all_cn_xr = convert_to_xarray(cell_states, basins, time_range, key="c_n")
    all_cn_xr.to_netcdf(cell_states_nc_file)
    return all_cn_xr


def load_cell_states_for_exp(exp):
    """Load cell states for one exp

    Parameters
    ----------
    exp : str
        for example, camels/exp41013

    Returns
    -------
    _type_
        _description_
    """
    two_parts = exp.split("/")
    run_dir = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", two_parts[0], two_parts[1]
    )
    config = get_config_file(run_dir)
    model = get_trained_model(config)
    cell_states = get_cell_states(model)
    return cell_states


def load_ts_var_data_for_exp(exp, var="ET") -> xr.Dataset:
    """Load time-series variable data for one exp

    Parameters
    ----------
    exp : str
        for example, camels/exp41013

    Returns
    -------
    xr.Dataset
        _description_
    """
    # we only support MODIS ET, usgsFlow and SMAP ssm now
    assert var in ["ET", "usgsFlow", "ssm"]
    two_parts = exp.split("/")
    run_dir = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", two_parts[0], two_parts[1]
    )

    config = get_config_file(run_dir)
    obs_nc_file = os.path.join(config["data_params"]["test_path"], "obs_" + var + ".nc")
    if os.path.exists(obs_nc_file):
        obs = xr.open_dataset(obs_nc_file)
        return obs

    basins = config["data_params"]["object_ids"]
    time_range = config["data_params"]["t_range_test"]
    source_path = [
        os.path.join(definitions.DATASET_DIR, "camelsflowet"),
        os.path.join(definitions.DATASET_DIR, "modiset4camels"),
        os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
        os.path.join(definitions.DATASET_DIR, "nldas4camels"),
        os.path.join(definitions.DATASET_DIR, "smap4camels"),
    ]
    camels_pro = CamelsPro(source_path)
    obs_ = camels_pro.read_target_cols(
        basins,
        time_range,
        [var],
    )
    if var == "usgsFlow":
        # Transform to mm/day to avoid basin area's effect
        basin_areas = camels_pro.read_basin_area(basins)
        basin_areas = np.repeat(basin_areas, obs_.shape[1], axis=0).reshape(obs_.shape)
        # ft3/s -> mm/day
        obs_ = obs_ / 35.314666721489 / (basin_areas * 1e6) * 86400

    obs = convert_to_xarray(obs_, basins, time_range, key="obs", dim=[var])
    obs.to_netcdf(obs_nc_file)
    return obs


def normalize_cell_states(
    cell_state: np.ndarray, desc: str = "Normalize"
) -> np.ndarray:
    """Normalize each cell state by DIMENSION"""
    original_shape = cell_state.shape
    store = []
    s = StandardScaler()
    n_dims = len(cell_state.shape)
    # note tested for 3-dim (basins, target_time, dimensions)
    if n_dims == 3:
        for ix in tqdm(range(cell_state.shape[-1]), desc=desc):
            store.append(s.fit_transform(cell_state[:, :, ix]))

        c_state = np.stack(store)
        c_state = c_state.transpose(1, 2, 0)
        assert c_state.shape == original_shape

    elif n_dims == 2:
        for ix in tqdm(range(cell_state.shape[-1]), desc=desc):
            store.append(s.fit_transform(cell_state[:, ix].reshape(-1, 1)))
        c_state = np.stack(store)[:, :, 0]
        c_state = c_state.T
        assert c_state.shape == original_shape

    else:
        raise NotImplementedError

    return c_state


def normalize_xarray_cstate(
    c_state: xr.Dataset, cell_state_var: str = "cell_state", cell_states_nc_file=None
) -> xr.Dataset:
    if cell_states_nc_file is not None and os.path.exists(cell_states_nc_file):
        return xr.open_dataset(cell_states_nc_file)
    #  Normalize all station values in cs_data:
    all_normed = []
    for station in c_state.station_id.values:
        norm_state = normalize_cell_states(
            c_state.sel(station_id=station)[cell_state_var].values
        )
        all_normed.append(norm_state)

    #  stack the normalized numpy arrays
    #   [station_id, time, dimension]
    all_normed_stack = np.stack(all_normed)
    norm_c_state = xr.ones_like(c_state[cell_state_var])
    norm_c_state = norm_c_state * all_normed_stack
    if type(norm_c_state) == xr.DataArray:
        norm_c_state = norm_c_state.to_dataset()
    norm_c_state.to_netcdf(cell_states_nc_file)
    return norm_c_state


def normalize_xr_by_basin(ds: xr.Dataset) -> xr.Dataset:
    return (ds - ds.mean(dim="date", skipna=True)) / ds.std(dim="date", skipna=True)
