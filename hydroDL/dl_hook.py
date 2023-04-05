"""
Author: Wenyu Ouyang
Date: 2022-11-23 15:33:52
LastEditTime: 2022-11-24 16:37:07
LastEditors: Wenyu Ouyang
Description: Some hooks for DL models, refer to: https://zhuanlan.zhihu.com/p/279903361
FilePath: /HydroSPB/hydroSPB/hydroDL/dl_hook.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import numpy as np
import xarray as xr
from collections import OrderedDict
import torch
from torch import nn, Tensor
from typing import Dict, Iterable, Callable


class VerboseExecution(nn.Module):
    """A simple wrapper for DL model to print something during forward calculation

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        # what a hook looks like could be learned from: https://www.youtube.com/watch?v=1ZbLA7ofasY
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        y = self.model(x)
        return y, self._features


# class SaveModelParams(nn.Module):
#     """Deprecated: this will lead to a strange error when forwarding"""

#     def __init__(self, model: nn.Module, layers: Iterable[str], **kwargs):
#         """Save weights and bias of chosen layers

#         Parameters
#         ----------
#         model : nn.Module
#             the model to be wrapped
#         layers : Iterable[str]
#             chosen layers
#         """
#         super().__init__()
#         self.model = model
#         self.layers = layers
#         self._params = {layer: torch.empty(0) for layer in layers}
#         self.save_iter = 1
#         if "save_iter" in kwargs:
#             self.save_iter = kwargs["save_iter"]

#         for layer_id in layers:
#             layer = dict([*self.model.named_modules()])[layer_id]
#             layer.register_forward_hook(self.save_weights_hook(layer_id))

#     def save_weights_hook(self, layer_id: str) -> Callable:
#         def fn(module, _, __):
#             # https://discuss.pytorch.org/t/whats-the-difference-between-module-parameters-vs-module-parameters/108924/3
#             self._params[layer_id] = OrderedDict()
#             for name, param in module.named_parameters():
#                 self._params[layer_id][name] = param.detach().cpu().numpy()

#         return fn

#     def forward(self, x: Tensor) -> Dict[Tensor, dict]:
#         y = self.model(x)
#         return y, self._params


def trans_model_params(saved_params) -> Dict[str, np.ndarray]:
    """Trans weights and bias Tensor to Numpy array

    Parameters
    ----------
    saved_params : Dict[str, Tensor]
        saved weights and bias
    """
    trans_param = {}
    for layer_id, params in saved_params.items():
        trans_param[layer_id] = OrderedDict({})
        for param_name, param in params.items():
            trans_param[layer_id][param_name] = param.detach().cpu().numpy()
    return trans_param


def concat_model_params(saved_params: Dict[str, np.ndarray]):
    """Trans weights and bias Tensor to Numpy array

    Parameters
    ----------
    saved_params : Dict[str, np.ndarray]
        saved weights and bias
    """
    layer_id_dict = OrderedDict({})
    for layer_id, param in saved_params[0].items():
        layer_id_dict[layer_id] = OrderedDict({})
        for param_name, param in param.items():
            layer_id_dict[layer_id][param_name] = []
    for params in saved_params:
        for layer_id, param in params.items():
            for param_name, param in param.items():
                layer_id_dict[layer_id][param_name].append(param)
    for layer_id, param in saved_params[0].items():
        for param_name, param in param.items():
            layer_id_dict[layer_id][param_name] = np.array(
                layer_id_dict[layer_id][param_name]
            )
    return layer_id_dict


def save_model_params(params, save_path, save_epoch=1, save_iter=1):
    """Save weights and bias of params

    Parameters
    ----------
    params
        the model parameters during training
    save_path : str
        path to save the weights and bias
    save_epoch : int, optional
        save weights and bias every save_epoch, by default 1
    save_iter : int, optional
        save weights and bias every save_iter, by default 1
    """
    for layer_id, param in params.items():
        for param_name, param in param.items():
            xr_param = xr.DataArray(
                param,
                dims=["epoch", "iter"]
                + ["dim" + str(i) for i in range(param.ndim - 2)],
            ).to_dataset(name=f"{layer_id}_{param_name}")
            xr_param["epoch"] = np.arange(1, param.shape[0] + 1) * save_epoch
            xr_param["iter"] = np.arange(1, param.shape[1] + 1) * save_iter
            xr_param.to_netcdf(
                os.path.join(save_path, f"{layer_id}_{param_name}_during_training.nc")
            )


def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val))

    return model
