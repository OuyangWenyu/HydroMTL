<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-04-05 20:10:24
 * @LastEditTime: 2024-06-19 17:18:38
 * @LastEditors: Wenyu Ouyang
 * @Description: README for HydroMTL
 * @FilePath: \HydroMTL\README.md
 * Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
-->
# HydroMTL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10024011.svg)](https://doi.org/10.5281/zenodo.10024011)

## What is HydroMTL

Multi-task deep learning models for basin hydrological modeling

All the code of this repository is also available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10024011).

If you feel it useful, please cite:

```bibtex
@misc{ouyang2021hydroMTL,
  author = {Wenyu Ouyang},
  title = {OuyangWenyu/HydroMTL},
  year = {2021},
  publisher = {Zenodo},
  journal = {Zenodo},
  howpublished = {\url{https://doi.org/10.5281/zenodo.10024011}}
}
```

## How to run

**NOTICE: a NVIDIA GPU is required to run the code!**

### Clone the repository

Fork this repository and clone it to your local machine.

```bash
# xxx is your github username
git clone git@github.com:xxxx/HydroMTL.git
cd HydroMTL
```

### Install dependencies

```bash
# if you have mamaba installed, it's faster to use mamba to create a new environment than conda
mamba env create -f environment.yml
# after the environment is created, activate it
conda activate MTL
# check if packages are installed correctly and HydroMTL is runnable
pytest tests
```

The code for Results and Discussion of the manuscript is in the scripts directory.

### Prepare data

Firstly, download data manually from [Kaggle](https://www.kaggle.com/datasets/owenyy/hydromtl) or [Zenodo](https://doi.org/10.5281/zenodo.10024011) .

Then, put the data in a folder and set this fold in definitions.py.
 
A recommeded way to config the data path is to create a file named `definitions_private.py` in the root directory of the project, and set the data path in it.

You can set the data path in `definitions_private.py` as follows:

```python
# xxx is your path
DATASET_DIR = xxx # This is your Data source directory
RESULT_DIR = xxx # This is your result directory
```

Put the downloaded data into the directory you set and unzip it.

The directory structure should be like this:

```bash
# xxx is your dataset directory
xxx
├── camels
│   ├── camels_us
│   ├──   |── basin_set_full_res
│   ├──   |── basin_timeseries_v1p2_metForcing_obsFlow
│   ├──   |── camels_streamflow
│   ├──   |── camels_attributes_v2.0.xlsx
│   ├──   |── camels_clim.txt
│   ├──   |── camels_geol.txt
│   ├──   |── camels_hydro.txt
│   ├──   |── camels_name.txt
│   ├──   |── camels_soil.txt
│   ├──   |── camels_topo.txt
│   ├──   |── camels_vege.txt
│── modiset4camels
│   ├── basin_mean_forcing
│   |   ├── MOD16A2_006_CAMELS
│   |   |  ├── 01
│   |   |  ├── 02
│   |   |  ├── ……
│   |   ├── MOD16A2GF_061_CAMELS
│   |   |  ├── 01
│   |   |  ├── 02
│   |   |  ├── ……
|   |   ├── ……
├── nldas4camels
│   ├── basin_mean_forcing
│   |   ├── 01
│   |   ├── 02
│   |   ├── ……
│── smap4camels
│   ├── NASA_USDA_SMAP_CAMELS
│   |   ├── 01
│   |   ├── 02
│   |   ├── ……
│   ├── SMAP_CAMELS
│   |   ├── 01
│   |   ├── 02
│   |   ├── ……
```

### Train

After data is ready, run the following command to train the model.

```bash
# if not in the scripts folder, cd to it
# cd scripts
# train MTL models, you can choose one to try
# for cache_path, mine is /home/ouyangwenyu/code/HydroMTL/results/camels/expmtl001
python run_task.py --exp expmtl001 --loss_weight 0.5 0.5 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 0 --random 1234
python run_task.py --exp expmtl002 --loss_weight 0.33 0.66 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 0 --random 1234 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None
python run_task.py --exp expmtl003 --loss_weight 0.75 0.25 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 0 --random 1234 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None
python run_task.py --exp expmtl004 --loss_weight 0.88 0.11 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 0 --random 1234 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None
python run_task.py --exp expmtl005 --loss_weight 0.96 0.04 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 0 --random 1234 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None
# train a STL (streamflow) model
python run_task.py --exp expstlq001 --loss_weight 1.0 0.0 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 1 --random 1234 --limit_part 1
# train a STL (evapotranspiration) model
python run_task.py --exp expstlet001 --loss_weight 0.0 1.0 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 1 --random 1234 --limit_part 0
```

The trained model will be saved in `./results/` folder.

### Test

One can use the trained model to test in any period.

```bash
# if not in the scripts folder, cd to it
# cd scripts
# for weight_path, mine is /home/ouyangwenyu/code/HydroMTL/results/camels/expstlq001/weights/07_April_202311_52AM_model.pth
# NOTE: We set test exp as trainexp+"0", for example, train exp is expmtl001, then, test exp is expmtl0010
python evaluate_task.py --exp expstlq0010 --loss_weight 1 0  --test_period 2016-10-01 2021-10-01 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None --weight_path /your/path/to/trained_model_pth_file
```

We provide a script to evaluate all trained models.

```bash
# if not in the scripts folder, cd to it
# cd scripts
python evaluate_all_tasks.py
```
