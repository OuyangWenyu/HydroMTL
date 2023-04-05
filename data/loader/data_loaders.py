"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-01-18 11:55:07
LastEditors: Wenyu Ouyang
Description: Self-made Data sets and loaders for DL models; references to https://github.com/mhpi/hydroDL
FilePath: /HydroSPB/hydroSPB/data/loader/data_loaders.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from abc import ABC
from functools import wraps
from typing import Union

import numpy as np
import torch
import pandas as pd

from hydroSPB.data.cache.cache_base import DataSourceCache
from hydroSPB.data.loader.dataloader_utils import read_yxc, check_data_loader
from hydroSPB.data.source.data_base import DataSourceBase
from hydroSPB.data.loader.data_scalers import ScalerHub, wrap_t_s_dict
from hydroSPB.utils.hydro_utils import (
    select_subset,
    select_subset_batch_first,
    check_np_array_nan,
)


class HydroDlTsDataModel(ABC):
    """time series data model in hydrodl of the MHPI group -- https://github.com/mhpi/hydroDL"""

    def __init__(self, data_source: DataSourceBase):
        """
        Parameters
        ----------
        data_source
            object for reading from data source
        """
        self.data_source = data_source

    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor]:
        """
        Read data from data source and compose a mini-batch

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]
        Returns
        -------
        tuple[torch.Tensor]
            a mini-batch tensor
        """
        raise NotImplementedError


class XczDataModel(HydroDlTsDataModel):
    """x,c,z are all inputs, where z are some special inputs, such as FDC"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        """
        Read source data (x,c,z are inputs; y is output) and normalize them

        Parameters
        ----------
        data_source
            object for reading source data
        data_params
            parameters for reading source data
        loader_type
            train, vaild or test
        """
        super().__init__(data_source)
        data_flow, data_forcing, data_attr, data_other = read_yxc(
            data_source, data_params, loader_type
        )
        # normalization
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            other_vars=data_other,
            data_params=data_params,
            loader_type=loader_type,
            data_source=data_source,
        )
        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        # different XczDataModel has different way to handle with z values, here only handle with one-key case
        one_value = list(scaler_hub.z.items())[0][1]
        if one_value.ndim == 3 and one_value.shape[-1] == 1:
            # if the 3rd dim is just 1, it must be expanded for normalization, and it will be used as kernel,
            # which we will use it as 2d var
            one_value = one_value.reshape(one_value.shape[0], one_value.shape[1])
        self.z = one_value
        self.target_scaler = scaler_hub.target_scaler

    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one mini-batch

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]

        Returns
        -------
        tuple
            a mini-batch data; x_train, z_train, y_train
        """
        if batch_first:
            x_train = select_subset_batch_first(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            # no need for y_train to set warmup period as loss is not calculated for warmup periods
            y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
            z_train = select_subset_batch_first(self.z, i_grid, i_t=None, rho=None)
        else:
            x_train = select_subset(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            y_train = select_subset(self.y, i_grid, i_t, rho)
            z_train = select_subset(self.z, i_grid, i_t=None, rho=None)
        # y_train must be the final!!!!!!
        return x_train, z_train, y_train


class DplDataModel(HydroDlTsDataModel):
    """Basin's rainfall-runoff mini-batch data model for Differential parameter learning"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        """
        Parameters
        ----------
        data_source
            object for reading source data
        data_params
            parameters for reading source data
        loader_type
            train, vaild or test
        """
        super().__init__(data_source)
        data_flow, data_forcing, data_attr = read_yxc(
            data_source, data_params, loader_type
        )
        # we need unnormalized data for the physical model;
        # data_flow, data_forcing and data_attr will be referred, so they may be changed
        self.x = np.copy(data_forcing)
        self.c = np.copy(data_attr)
        # the final output of physical model is unnormalized output;
        # we don't use y_un_norm as its name because in the main function we will use "y"
        # transform streamflow (ft^3/s) to (mm/day) then there is no need to transform it during model forwarding
        y_data = np.copy(data_flow)
        t_s_dict = wrap_t_s_dict(data_source, data_params, loader_type)
        basins_id = t_s_dict["sites_id"]
        basin_area = data_source.read_basin_area(basins_id)
        basin_areas = np.repeat(basin_area, y_data.shape[1], axis=0).reshape(
            y_data.shape
        )
        self.y = 0.0283168 * 24 * 3600 * 1e3 * y_data / (basin_areas * 1e6)
        x_rm_nan = data_params["relevant_rm_nan"]
        c_rm_nan = data_params["constant_rm_nan"]
        y_rm_nan = data_params["target_rm_nan"]
        if x_rm_nan:
            self.x[np.where(np.isnan(self.x))] = 0
        if c_rm_nan:
            self.c[np.where(np.isnan(self.c))] = 0
        if y_rm_nan and loader_type == "train":
            # for streamflow, we now provide a interpolation way, but only for training
            y_df = pd.DataFrame(
                self.y.reshape(self.y.shape[0], self.y.shape[1]).T, columns=basins_id
            )
            y_df_intepolate = y_df.interpolate(
                method="linear", limit_direction="forward", axis=0
            )
            self.y = y_df_intepolate.values.T.reshape(self.y.shape)
        # there may be NaN values in x and c
        # For physical hydrologic models, we need warmup, hence the target values should exclude data in warmup period
        self.warmup_length = data_params["warmup_length"]
        # we also need normalized data for the DL model
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            data_params=data_params,
            loader_type=loader_type,
            data_source=data_source,
        )
        # perform the norm
        self.x_norm = scaler_hub.x
        self.y_norm = scaler_hub.y
        self.c_norm = scaler_hub.c
        self.target_scaler = scaler_hub.target_scaler
        self.target_as_input = data_params["target_as_input"]
        self.constant_only = data_params["constant_only"]
        if self.target_as_input and (not self.train_mode):
            # TODO: to be debugged to check if this code works
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = DplDataModel(
                data_source, data_params, loader_type="train"
            )

    @check_data_loader
    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one mini-batch for dPL (differential parameter learning) model

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]

        Returns
        -------
        tuple
            a mini-batch data;
            x_train (not normalized forcing), z_train (normalized data for DL model), y_train (not normalized output)
        """
        if self.target_as_input:
            # when target_as_input is True, we need use training data to generate pbm params
            if batch_first:
                xc_norm = select_subset_batch_first(
                    self.train_dataset.x_norm,
                    i_grid,
                    i_t,
                    rho,
                    warmup_length=warmup_length,
                    c=self.c_norm,
                )
                y_norm = select_subset_batch_first(
                    self.train_dataset.y_norm,
                    i_grid,
                    i_t,
                    rho,
                    warmup_length=warmup_length,
                )
                # forcing data used in physical model are nor normalized
                x_train = select_subset_batch_first(
                    self.x, i_grid, i_t, rho, warmup_length=warmup_length
                )
                # output are also not normalized, because the final putout comes from physical model；
                # but we don't need warmup for y_train, because we only calculate loss for formal period
                y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
            else:
                xc_norm = select_subset(
                    self.train_dataset.x_norm,
                    i_grid,
                    i_t,
                    rho,
                    warmup_length=warmup_length,
                    c=self.c_norm,
                )
                y_norm = select_subset(
                    self.train_dataset.y_norm,
                    i_grid,
                    i_t,
                    rho,
                    warmup_length=warmup_length,
                )
                x_train = select_subset(
                    self.x, i_grid, i_t, rho, warmup_length=warmup_length
                )
                y_train = select_subset(self.y, i_grid, i_t, rho)
            # y_morn and xc_norm are concatenated and used for DL model
            # the order of xc_norm and y_norm matters, please be careful!
            z_train = torch.cat((xc_norm, y_norm), 2)
        else:
            if self.constant_only:
                # only use attributes data for DL model
                z_train = select_subset_batch_first(
                    self.x_norm,
                    i_grid,
                    None,
                    rho,
                    warmup_length=warmup_length,
                    c=self.c_norm,
                )
                if batch_first:
                    # forcing data used in physical model are nor normalized
                    x_train = select_subset_batch_first(
                        self.x, i_grid, i_t, rho, warmup_length=warmup_length
                    )
                    # output are also not normalized, because the final putout comes from physical model；
                    # but we don't need warmup for y_train, because we only calculate loss for formal period
                    y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
                else:
                    x_train = select_subset(
                        self.x, i_grid, i_t, rho, warmup_length=warmup_length
                    )
                    y_train = select_subset(self.y, i_grid, i_t, rho)
            else:
                # no streamflow data, only attribute/forcing data are available for models
                if batch_first:
                    z_train = select_subset_batch_first(
                        self.x_norm,
                        i_grid,
                        i_t,
                        rho,
                        warmup_length=warmup_length,
                        c=self.c_norm,
                    )
                    # forcing data used in physical model are nor normalized
                    x_train = select_subset_batch_first(
                        self.x, i_grid, i_t, rho, warmup_length=warmup_length
                    )
                    # output are also not normalized, because the final putout comes from physical model；
                    # but we don't need warmup for y_train, because we only calculate loss for formal period
                    y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
                else:
                    z_train = select_subset(
                        self.x_norm,
                        i_grid,
                        i_t,
                        rho,
                        warmup_length=warmup_length,
                        c=self.c_norm,
                    )
                    x_train = select_subset(
                        self.x, i_grid, i_t, rho, warmup_length=warmup_length
                    )
                    y_train = select_subset(self.y, i_grid, i_t, rho)
        return x_train, z_train, y_train


class BasinFlowDataModel(HydroDlTsDataModel):
    """Basic basin's rainfall-runoff mini-batch data model"""

    def __init__(
        self, data_source: DataSourceBase, data_params: dict, loader_type: str
    ):
        """
        Parameters
        ----------
        data_source
            object for reading source data
        data_params
            parameters for reading source data
        loader_type
            train, vaild or test
        """
        super().__init__(data_source)
        data_flow, data_forcing, data_attr = read_yxc(
            data_source, data_params, loader_type
        )
        # normalization
        scaler_hub = ScalerHub(
            data_flow,
            data_forcing,
            data_attr,
            data_params=data_params,
            loader_type=loader_type,
            data_source=data_source,
        )
        # perform the norm
        self.x = scaler_hub.x
        self.y = scaler_hub.y
        self.c = scaler_hub.c
        self.target_scaler = scaler_hub.target_scaler

    @check_data_loader
    def get_item(
        self, i_grid, i_t, rho, warmup_length=0, batch_first=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get one mini-batch tensor from np.array data samples

        Parameters
        ----------
        i_grid
            i-th basin/grid/...
        i_t
            i-th period
        rho
            time_length of a time-sequence
        warmup_length
            time length of warmup period
        batch_first
            if True, the batch data's dim is [batch, seq, feature]; else [seq, batch, feature]

        Returns
        -------
        tuple
            a mini-batch data; x_train (x concat with c), y_train
        """
        if batch_first:
            x_train = select_subset_batch_first(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            # y_train don't need warmup period since loss is only calculated for formal periods
            y_train = select_subset_batch_first(self.y, i_grid, i_t, rho)
        else:
            x_train = select_subset(
                self.x, i_grid, i_t, rho, warmup_length=warmup_length, c=self.c
            )
            y_train = select_subset(self.y, i_grid, i_t, rho)
        return x_train, y_train
