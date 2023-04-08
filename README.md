<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-04-05 20:10:24
 * @LastEditTime: 2023-04-08 10:42:50
 * @LastEditors: Wenyu Ouyang
 * @Description: README for HydroMTL
 * @FilePath: /HydroMTL/README.md
 * Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
-->
# HydroMTL

## What is HydroMTL

Multi-task deep learning models for basin hydrological modeling

If you feel it useful, please cite our paper:

```bibtex
@misc{ouyang2021hydroMTL,
  author = {Wenyu Ouyang},
  title = {HydroMTL: Multi-task deep learning models for basin hydrological modeling},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{}}
}
```

All the code of this repository is also available on [Zenodo](https://zenodo.org/record/5550000).

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

### Prepare data

Firstly, download data manually from [Kaggle]() or [Zenodo]().

Then, put the data in a folder and set this fold in defination.py.

```python
# defination.py
DATASET_DIR = 'path/to/your/data/source/folder'
```

Run the following command to prepare the data.

```bash
cd scripts
python prepare_data.py
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
# train a STL (streamflow) model
python run_task.py --exp expstlq001 --loss_weight 1.0 0.0 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 1 --random 1234 --limit_part 1
# train a STL (evapotranspiration) model
python run_task.py --exp expstlet001 --loss_weight 0.0 1.0 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 1 --random 1234 --limit_part 0
```

The trained model will be saved in `./results/` folder.

If you don't want to train the model, you can download the trained model from [Kaggle]() or [Zenodo]().

### Test

One can use the trained model to test in any period.

```bash
# if not in the scripts folder, cd to it
# cd scripts
# for weight_path, mine is /home/ouyangwenyu/code/HydroMTL/results/camels/expstlq001/weights/07_April_202311_52AM_model.pth
python evaluate_task.py --exp expstlq0010 --loss_weight 1 0  --test_period 2016-10-01 2021-10-01 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None --weight_path /your/path/to/trained_model_pth_file
```
