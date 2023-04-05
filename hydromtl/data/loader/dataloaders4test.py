from typing import Union

import numpy as np
import pandas as pd
import torch

from hydroSPB.data.loader.data_loaders import (
    BasinFlowDataModel,
    XczDataModel,
    DplDataModel,
)
from hydroSPB.data.loader.data_sets import BasinFlowDataset


class TestDataModel(object):
    """Data model for test or validation (denormalization)"""

    def __init__(
        self, test_data: Union[BasinFlowDataModel, XczDataModel, BasinFlowDataset]
    ):
        """test_data is the data loader when initializing TimeSeriesModel"""
        self.test_data = test_data
        self.target_scaler = test_data.target_scaler

    def inverse_scale(
        self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    ) -> np.array:
        """
        denormalization of data

        Parameters
        ----------
        result_data
            The data you want to unscale can handle multiple data types.

        Returns
        -------
        np.array
            Returns the unscaled data as np.array.
        """
        if isinstance(result_data, (pd.Series, pd.DataFrame)):
            result_data_np = result_data.values
        elif isinstance(result_data, torch.Tensor):
            # TODO: not tested, useful when validating
            if len(result_data.shape) > 2:
                result_data = result_data.permute(2, 0, 1).reshape(
                    result_data.shape[2], -1
                )
                result_data = result_data.permute(1, 0)
            result_data_np = result_data.numpy()
        elif isinstance(result_data, np.ndarray):
            result_data_np = result_data
        else:
            raise TypeError("No such data type for denormalization!")
        np_target_denorm = self.target_scaler.inverse_transform(result_data_np)
        return np_target_denorm

    def load_test_data(self):
        # don't test for warmup period yet as no pbm use it now
        x = self.test_data.x
        c = self.test_data.c
        y = self.test_data.y
        if hasattr(self.test_data, "z"):
            z = self.test_data.z
            # y must be the final!!!
            return x, c, z, y
        return x, c, y


class TestDplDataModel(object):
    """DplDataModel for test"""

    def __init__(self, train_data: DplDataModel, test_data: DplDataModel):
        """test_data is the data loader when initializing TimeSeriesModel"""
        self.train_data = train_data
        self.test_data = test_data

    def inverse_scale(
        self, result_data: Union[torch.Tensor, pd.Series, np.ndarray]
    ) -> np.array:
        """
        This function didn't perform any calculation, just return input.

        Its purpose is to keep consistent with all test data models

        Parameters
        ----------
        result_data
            result from model

        Returns
        -------
        np.array
            result from model
        """
        return result_data

    def load_test_data(self):
        warmup_length = self.test_data.warmup_length
        x = self.test_data.x
        c = self.test_data.c
        y = self.test_data.y
        # we use x_norm, c_norm and y_norm from training periods to generate PBM's parameters
        x_norm = self.train_data.x_norm
        c_norm = self.train_data.c_norm
        y_norm = self.train_data.y_norm
        # if we need dynamic parameters for lstm dPLt model, we need x y c in test_data
        x_test_norm = self.test_data.x_norm
        c_test_norm = self.test_data.c_norm
        y_test_norm = self.test_data.y_norm
        # Only y need warmup. For example, input: x (length = 10), warmup = 5, then output y(length=10-5=5)
        return (
            x_norm,  # normalized training forcing
            c_norm,  # normalized training attrs
            y_norm,  # normalized training target
            x,  # not normalized test forcing
            c,  # not normalized test attrs
            x_test_norm,  # normalized test forcing
            c_test_norm,  # normalized test attrs
            y_test_norm,  # normalized test target
            y[:, warmup_length:, :],  # not normalized test target
        )


class TestDplDataset(object):
    """It is a little different for dataset to test for dpl model, so we set this class"""

    def __init__(self, train_data, test_data):
        """test_data is the data loader when initializing TimeSeriesModel"""
        self.train_data = train_data
        self.test_data = test_data

    def inverse_scale(self, result_data) -> np.array:
        """
        This function didn't perform any calculation, just return input.

        Its purpose is to keep consistent with all test data models

        Parameters
        ----------
        result_data
            result from model

        Returns
        -------
        np.array
            result from model
        """
        return result_data

    def load_test_data(self):
        warmup_length = self.test_data.warmup_length
        x_origin = self.test_data.x_origin
        c_origin = self.test_data.c_origin
        y_origin = self.test_data.y_origin
        x = self.train_data.x
        c = self.train_data.c
        y = self.train_data.y
        # if we need dynamic parameters for lstm dPLt model, we need x y c in test_data
        x_test_norm = self.test_data.x
        c_test_norm = self.test_data.c
        y_test_norm = self.test_data.y
        return (
            x,  # normalized training forcing
            c,  # normalized training attributes
            y,  # normalized training output
            x_origin,  # not normalized forcing
            c_origin,  # not normalized attributes
            x_test_norm,  # normalized forcing for test
            c_test_norm,  # normalized attributes for test
            y_test_norm,  # normalized output for test
            y_origin[:, warmup_length:, :],  # not normalized output
        )
