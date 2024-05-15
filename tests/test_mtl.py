"""
Author: Wenyu Ouyang
Date: 2024-05-15 18:28:53
LastEditTime: 2024-05-15 18:37:43
LastEditors: Wenyu Ouyang
Description: Test for multioutput model
FilePath: \HydroMTL\tests\test_mtl.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import definitions
from hydromtl.models.trainer import train_and_evaluate
from hydromtl.data.config import cmd, update_cfg, default_config_file


def test_data_syn_flow_et():
    """
    Test for flow and et

    Parameters
    ----------
    config_data

    Returns
    -------

    """
    project_name = os.path.join("test_camels", "exp001")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_path=[
            os.path.join(definitions.DATASET_DIR, "camelsflowet"),
            os.path.join(definitions.DATASET_DIR, "modiset4camels"),
            os.path.join(definitions.DATASET_DIR, "camels", "camels_us"),
            os.path.join(definitions.DATASET_DIR, "nldas4camels"),
            os.path.join(definitions.DATASET_DIR, "smap4camels"),
        ],
        source="CAMELS_FLOW_ET",
        download=0,
        ctx=[0],
        model_name="KuaiLSTMMultiOut",
        model_param={
            "n_input_features": 23,
            "n_output_features": 2,
            "n_hidden_states": 256,
            # when layer_hidden_size is None, it is same with KuaiLSTM
            # "layer_hidden_size": None,
            "layer_hidden_size": 128,
        },
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 2],
            "item_weight": [0.5, 0.5],
        },
        cache_write=1,
        cache_read=1,
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "01055000",
            "01057000",
            "01170100",
        ],
        batch_size=5,
        rho=30,  # batch_size=100, rho=365,
        var_t=[
            "temperature",
            "specific_humidity",
            "shortwave_radiation",
            "potential_energy",
            "potential_evaporation",
            "total_precipitation",
        ],
        var_t_type=["nldas"],
        var_out=["usgsFlow", "ET"],
        train_period=["2001-10-01", "2002-10-01"],
        test_period=["2002-10-01", "2003-10-01"],
        data_loader="StreamflowDataModel",
        scaler="DapengScaler",
        scaler_params={
            "basin_norm_cols": [
                "usgsFlow",
                "ET",
            ],
            "gamma_norm_cols": [
                "prcp",
                "pr",
                "total_precipitation",
                "pre",
                # pet may be negative, but we set negative as 0 because of gamma_norm_cols
                # https://earthscience.stackexchange.com/questions/12031/does-negative-reference-evapotranspiration-make-sense-using-fao-penman-monteith
                "pet",
                "potential_evaporation",
                "PET",
            ],
        },
        train_epoch=20,
        te=20,
        fill_nan=["no", "mean"],
        n_output=2,
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
