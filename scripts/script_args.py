"""
Author: Wenyu Ouyang
Date: 2022-11-09 15:46:41
LastEditTime: 2023-04-07 10:46:23
LastEditors: Wenyu Ouyang
Description: Try cmds
FilePath: /HydroMTL/scripts/script_args.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
#################################################
# to test if the arg-parse commands are correct

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from hydromtl.data.config import default_config_file, cmd, update_cfg


def main(config_data, args):
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    args.gage_id = ["01013500"]
    update_cfg(config_data, args)
    print("Process is now complete.")


# python script_args.py --sub camels/exp40 --download 0 --model_name DplHbv --opt Adadelta --rs 1234 --cache_write 0 --cache_read 0 --model_param {\"n_input_features\":24\,\"n_output_features\":14\,\"n_hidden_states\":256\,\"warmup_length\":30} --scaler DapengScaler --warmup_length 10 --data_loader DplDataModel --batch_size 100 --rho 365 --train_epoch 300 --var_t total_precipitation potential_evaporation temperature specific_humidity shortwave_radiation potential_energy --train_period 1985-10-01 1995-10-01 --test_period 1995-10-01 2005-10-01
if __name__ == "__main__":
    print("Begin\n")
    config = default_config_file()
    cmd_args = cmd()
    main(config, cmd_args)
    print(config["model_params"])
    print("End\n")
