"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-04-06 21:48:50
LastEditors: Wenyu Ouyang
Description: Entrypoint for training all ML models
FilePath: /HydroMTL/hydromtl/scripts/train_evaluate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from hydromtl.data.config import default_config_file, cmd, update_cfg
from hydromtl.models.trainer import train_and_evaluate


def main(args):
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    config_data = default_config_file()
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")


# python camels_analysis.py --sub camels/exp2 --download 0 --ctx 1 --model_name LSTM --model_param {\"nx\":23\,\"ny\":1\,\"hidden_size\":256\,\"num_layers\":2\,\"dr\":0.5} --opt Adadelta --loss_func RMSESum --rs 1234 --cache_read 0 --cache_write 0 --train_period 1990-01-01 2000-01-01 --test_period 2000-01-01 2010-01-01 --scaler DapengScaler --data_loader StreamflowDataModel --train_epoch 300 --save_epoch 20 --te 300 --batch_size 100 --rho 365 --var_t dayl prcp srad tmax tmin vp
if __name__ == "__main__":
    print("Begin\n")
    cmd_args = cmd()
    main(cmd_args)
    print("End\n")
