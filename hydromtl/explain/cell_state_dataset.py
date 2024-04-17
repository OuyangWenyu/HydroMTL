import copy
from typing import Any, Optional, Tuple, Union
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, Dataset, SubsetRandomSampler


def _fill_gaps_da(da: xr.DataArray, fill_nan: Optional[str] = None) -> xr.DataArray:
    """Fill gaps in a DataArray"""
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    if fill_nan is None:
        return da
    # fill gaps
    if fill_nan == "et_ssm_ignore":
        # only for MODIS ET or SMAP ssm -- ignore normal nan values and interpolate npn-normal nan values
        all_non_nan_idx = []
        for i in range(da.shape[0]):
            non_nan_idx_tmp = np.where(~np.isnan(da[i].values))
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp[0].tolist()
        # some NaN data appear in different dates in different basins
        non_nan_idx = np.unique(all_non_nan_idx).tolist()
        for i in range(da.shape[0]):
            targ_i = da[i][non_nan_idx]
            da[i][non_nan_idx] = targ_i.interpolate_na(
                dim="date", fill_value="extrapolate"
            )
    elif fill_nan == "median":
        # fill median
        da = da.fillna(da.median())
    elif fill_nan == "interpolate":
        # fill interpolation
        for i in range(da.shape[0]):
            da[i] = da[i].interpolate_na(dim="date", fill_value="extrapolate")
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da


def choose_data_like_et_or_ssm(da_input, da_target, var="ET"):
    """select data from ds at the time of ds_target non-nan data
    and we will use them as input and output of a ML model

    Parameters
    ----------
    da_input : xr.DataArray
        data to be chosen
    da_target : xr.DataArray
        the template data
    """
    # we have fill gap for ET to guarantee that all basins nan-data come out in same dates
    # so here just choose data_target[0]
    non_nan_idx = np.where(~np.isnan(da_target[0].values))[0].tolist()
    assert type(non_nan_idx) == list, "non_nan_idx should be list"
    chosen_date = da_target[0].date.values[non_nan_idx]
    input_data = da_input.sel(date=chosen_date)
    output_data = da_target.sel(date=chosen_date)
    if var == "ssm":
        return input_data, output_data
    elif var == "ET":
        # calculate mean of values in multiple dates between two non-nan values for ET
        for i in range(da_input.shape[0]):
            cs_i_sum = np.add.reduceat(da_input[i].values, non_nan_idx, axis=0)
            if non_nan_idx[-1] < da_input[i].date.size:
                idx4mean = non_nan_idx + [da_input[i].date.size]
            else:
                idx4mean = copy.copy(non_nan_idx)
            idx_interval = [y - x for x, y in zip(idx4mean, idx4mean[1:])]
            cs_i_mean = (cs_i_sum.T / idx_interval).T
            input_data[i] = cs_i_mean
    else:
        raise NotImplementedError(f"var {var} not implemented")
    return input_data, output_data


def fill_gaps(
    ds: Union[xr.DataArray, xr.Dataset], fill_nan: Optional[list] = None
) -> Union[xr.DataArray, xr.Dataset]:
    """ET is 8-day data, so we have to fill nan values with the mean of the all 8 days

    Parameters
    ----------
    ds : Union[xr.DataArray, xr.Dataset]
        _description_
    fill : Optional[str], optional
        _description_, by default None

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        _description_
    """
    if fill_nan is None:
        return ds
    if isinstance(ds, xr.Dataset):
        pbar = tqdm(ds.data_vars, desc=f"Filling gaps with method {fill_nan}")
        for v in pbar:
            pbar.set_postfix_str(v)
            for count in range(ds[v]["dimension"].size):
                # basin, date, variable
                ds[v][:, :, count] = _fill_gaps_da(
                    ds[v][:, :, count], fill_nan=fill_nan[count]
                )
    else:
        for count in range(ds["dimension"].size):
            ds[v][:, :, count] = _fill_gaps_da(
                ds[v][:, :, count], fill_nan=fill_nan[count]
            )
    return ds


class CellStateDataset(Dataset):
    def __init__(
        self,
        input_data: xr.DataArray,  #  cell state (`hs` dimensions)
        target_data: xr.DataArray,  #  soil moisture
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        device: str = "cpu",
        variable_str: str = "cell_state",
        fill: Optional[str] = None,
    ):
        assert all(np.isin(["time", "dimension", "station_id"], input_data.dims))

        #  drop missing / non matching basins
        if not all(
            np.isin(input_data.station_id.values, target_data.station_id.values)
        ):
            input_data = input_data.sel(
                station_id=np.isin(
                    input_data.station_id.values, target_data.station_id.values
                )
            )

        self.input_data = input_data
        self.variable_str = variable_str

        #  All times that we have data for
        times = pd.date_range(start_date, end_date, freq="D")
        bool_input_times = np.isin(input_data.time.values, times)
        bool_target_times = np.isin(target_data.time.values, times)
        self.all_times = list(
            set(target_data.time.values[bool_target_times]).intersection(
                set(input_data.time.values[bool_input_times])
            )
        )
        self.all_times = sorted(self.all_times)

        # get input/target data
        input_data = self.input_data.sel(time=self.all_times)
        self.input_data = fill_gaps(input_data, fill_nan=fill)
        target_data = target_data.sel(time=self.all_times)
        self.target_data = fill_gaps(target_data, fill_nan=fill)

        # basins
        self.basins = input_data.station_id.values

        # dimensions
        self.dimensions = len(input_data.dimension.values)

        # create x y pairs
        self.create_samples()

    def __len__(self):
        return len(self.samples)

    def create_samples(self):
        self.samples = []
        self.basin_samples = []
        self.time_samples = []

        for basin in self.basins:
            # read the basin data
            X = self.input_data.sel(station_id=basin).values.astype("float64")
            Y = self.target_data.sel(station_id=basin).values.astype("float64")

            # Ensure time is the 1st (0 index) axis
            X_time_axis = int(
                np.argwhere(~np.array([ax == len(self.all_times) for ax in X.shape]))
            )
            if X_time_axis != 1:
                X = X.transpose(1, 0)

            # drop nans over time (1st axis)
            finite_indices = np.logical_and(np.isfinite(Y), np.isfinite(X).all(axis=1))
            X, Y = X[finite_indices], Y[finite_indices]
            times = self.input_data["time"].values[finite_indices].astype(float)

            # convert to Tensors
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()

            # create unique samples [(units_num,), (1,)]
            samples = [(x, y.reshape(-1)) for (x, y) in zip(X, Y)]
            self.samples.extend(samples)
            self.basin_samples.extend([basin for _ in range(len(samples))])
            self.time_samples.extend(times)

        #  SORT DATA BY TIME (important for train-test split)
        sort_idx = np.argsort(self.time_samples)
        self.time_samples = np.array(self.time_samples)[sort_idx]
        try:
            self.samples = np.array(self.samples)[sort_idx]
        except TypeError:
            self.samples = np.array(list(self.samples))[sort_idx]
        self.basin_samples = np.array(self.basin_samples)[sort_idx]

    def __getitem__(self, item: int) -> Tuple[Tuple[str, Any], Tuple[torch.Tensor]]:
        basin = str(self.basin_samples[item])
        time = self.time_samples[item]
        x, y = self.samples[item]
        return {"x_d": x, "y": y, "meta": {"basin": basin, "time": time}}


def train_validation_split(
    dataset: Dataset,
    validation_split: float = 0.1,
    shuffle_dataset: bool = True,
    random_seed: int = 42,
) -> Tuple[SubsetRandomSampler]:
    #  SubsetRandomSampler = https://stackoverflow.com/a/50544887
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


# train-test split
def get_train_test_dataset(
    dataset: Dataset, test_proportion: float = 0.2
) -> Tuple[Subset, Subset]:
    #  Subset = https://stackoverflow.com/a/59414029
    #  random_split = https://stackoverflow.com/a/51768651
    all_data_size = len(dataset)
    train_size = int((1 - test_proportion) * all_data_size)
    test_size = all_data_size - train_size
    test_index = all_data_size - int(np.floor(test_size))

    #  test data is from final_sequence : end
    test_dataset = Subset(dataset, range(test_index, all_data_size))
    # train data is from start : test_index
    train_dataset = Subset(dataset, range(test_index))
    #  sense-check
    assert len(train_dataset) + len(test_dataset) == all_data_size

    return train_dataset, test_dataset
