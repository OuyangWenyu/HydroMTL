"""
Author: Wenyu Ouyang
Date: 2022-11-20 20:52:08
LastEditTime: 2022-11-24 11:31:08
LastEditors: Wenyu Ouyang
Description: Some nn models for LSTM probe
FilePath: /HydroSPB/hydroSPB/explore/cell_state_model.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from sklearn.base import RegressorMixin
from hydromtl.explain.cell_state_dataset import (
    CellStateDataset,
    train_validation_split,
    get_train_test_dataset,
)
from torch.utils.data import DataLoader


class LinearModel(nn.Module):
    def __init__(self, D_in: int, dropout: float = 0.0, **kwargs):
        super(LinearModel, self).__init__(**kwargs)

        #  number of weights == number of dimensions in cell state vector (cfg.hidden_size)
        self.D_in = D_in
        self.model = nn.Sequential(nn.Linear(self.D_in, 1, bias=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data: Dict[str, torch.Tensor]):
        return self.model(self.dropout(data["x_d"].squeeze()))


def train_model(
    model: nn.Module,
    train_dataset: CellStateDataset,
    device,
    learning_rate: float = 1e-2,
    n_epochs: int = 5,
    l2_penalty: float = 0,
    val_split: bool = False,
    desc: str = "Training Epoch",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[Any, List[float], List[float]]:
    #  GET loss function
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # GET optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_penalty
    )
    #  TRAIN
    train_losses_ALL = []
    val_losses_ALL = []
    for epoch in tqdm(range(n_epochs), desc=desc):
        train_losses = []
        val_losses = []

        #  new train-validation split each epoch
        if val_split:
            #  create a unique test, val set (random) for each ...
            train_sampler, val_sampler = train_validation_split(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

        for data in train_loader:
            data["x_d"] = data["x_d"].to(device)
            data["y"] = data["y"].to(device)
            y_pred = model(data)
            y = data["y"][
                :,
            ]
            loss = loss_fn(y_pred, y)

            # train/update the weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())

        # VALIDATE
        if val_split:
            model.eval()
            with torch.no_grad():
                for (basin, time), data in val_loader:
                    X, y = data
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                    val_losses.append(loss.detach().cpu().numpy())

            #  save the epoch-mean losses
            val_losses_ALL = np.mean(val_losses)

        train_losses_ALL.append(np.mean(train_losses))

    return model, train_losses_ALL, val_losses_ALL


def to_xarray(predictions: Dict[str, List]) -> xr.Dataset:
    return pd.DataFrame(predictions).set_index(["time", "station_id"]).to_xarray()


def calculate_predictions(model, loader: DataLoader, device) -> xr.Dataset:
    predictions = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            X, y = data["x_d"], data["y"]
            data["x_d"] = X.to(device)
            data["y"] = y.to(device)
            basin, time = data["meta"]["basin"], data["meta"]["time"]
            y_hat = model(data)

            #  Coords / Dimensions
            predictions["time"].extend(pd.to_datetime(time))
            predictions["station_id"].extend(basin)

            # Variables
            predictions["y_hat"].extend(y_hat.detach().cpu().numpy().flatten())
            predictions["y"].extend(y.detach().cpu().numpy().flatten())

    return to_xarray(predictions)


#  ALL Training Process
def train_model_loop(
    input_data: xr.DataArray,
    target_data: xr.DataArray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_test: bool = True,
    train_val: bool = False,
    return_loaders: bool = True,
    desc: str = "Training Epoch",
    dropout: float = 0.0,
    device: str = "cpu",
    l2_penalty: float = 0,
    num_workers: int = 4,
):
    #  1. create dataset (input, target)
    dataset = CellStateDataset(
        input_data=input_data,
        target_data=target_data,
        device=device,
        start_date=start_date,
        end_date=end_date,
    )
    print("Data Loaded")

    #  2. create train-test split
    if train_test:
        #  build the train, test, validation
        train_dataset, test_dataset = get_train_test_dataset(dataset)
        test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=num_workers
        )
    else:
        train_dataset = dataset
        test_dataset = dataset
        test_loader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=num_workers
        )
    print("Train-Test-Val Split")

    #  3. initialise the model
    model = LinearModel(D_in=dataset.dimensions, dropout=dropout)
    model = model.to(device)

    # 4. Run training loop (iterate over batches)
    print("Start Training")
    model, train_losses, _ = train_model(
        model,
        train_dataset,
        device,
        learning_rate=1e-3,
        n_epochs=20,
        val_split=train_val,
        desc=desc,
        l2_penalty=l2_penalty,
    )

    # 5. Save outputs (losses: List[float], model: BaseModel, dataloader: DataLoader)
    if return_loaders:
        return train_losses, model, test_loader
    else:
        return train_losses, model, None


def get_model_weights(model: Union[LinearModel, RegressorMixin]) -> Tuple[np.ndarray]:
    if isinstance(model, LinearModel):
        parameters = list(model.parameters())
        w = parameters[0].cpu().detach().numpy()
        b = parameters[1].cpu().detach().numpy()
    elif isinstance(model, RegressorMixin):
        w = model.coef_
        b = model.intercept_
    else:
        assert False, "Only works with Pytorch and Sklearn models"
    return w, b


def get_all_models_weights(models: List[nn.Linear]) -> Tuple[np.ndarray]:
    model_outputs = defaultdict(dict)
    for sw_ix in range(len(models)):
        w, b = get_model_weights(models[sw_ix])
        model_outputs[f"swvl{sw_ix+1}"]["w"] = w
        model_outputs[f"swvl{sw_ix+1}"]["b"] = b
    ws_np = np.stack([model_outputs[swl]["w"] for swl in model_outputs.keys()])
    # TODO: not fully test for multiple models
    ws = ws_np.reshape(len(models), ws_np.shape[-1])
    bs = np.stack([model_outputs[swl]["b"] for swl in model_outputs.keys()])
    return ws, bs


def calculate_raw_correlations(
    input_data: xr.DataArray,
    target_data: xr.DataArray,
    device: str = "cpu",
    time_dim: str = "time",
):
    """Calculate the correlation coefficient of input_data and target_data
    using: `np.corrcoef`.

    Parameters
    ----------
    input_data : xr.DataArray
        The input cell state data
    target_data : xr.DataArray
        The target variable data (1D)
    device : str, optional
        _description_, by default "cpu"
    time_dim : str, optional
        _description_, by default "time"

    Returns
    -------
    tuple(np.ndarray, xr.DataArray)
        correlation coefficient for each cell state
        correlation coefficient for each cell state and each basin
    """
    #  Create the datasets for each feature
    start_date = pd.to_datetime(input_data[time_dim].min().values)
    end_date = pd.to_datetime(input_data[time_dim].max().values)
    dataset = CellStateDataset(
        input_data=input_data,
        target_data=target_data,
        device=device,
        start_date=start_date,
        end_date=end_date,
    )

    # trans to int for easier plotting
    target_data["station_id"] = [int(sid) for sid in target_data["station_id"]]
    input_data["station_id"] = [int(sid) for sid in input_data["station_id"]]
    all_basin_correlations = xr.corr(input_data, target_data, dim="time")
    # Calculate the correlations for each level
    all_correlations = np.zeros(256)

    #  get the DATA for that feature
    all_cs_data = np.array(
        [data["x_d"].detach().cpu().numpy() for data in DataLoader(dataset)]
    )
    all_var_data = np.array(
        [data["y"].detach().cpu().numpy() for data in DataLoader(dataset)]
    )
    Y = all_var_data.reshape(-1, 1)
    correlations = []
    for cs in tqdm(np.arange(all_cs_data.shape[-1])):
        # correlations for each cell state, all basins' data are flattened
        X = all_cs_data[:, :, cs]
        correlations.append(np.corrcoef(X, Y, rowvar=False)[0, 1])
    #  save correlations
    correlations = np.array(correlations)
    all_correlations += correlations

    return all_correlations, all_basin_correlations
