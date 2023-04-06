<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-04-05 20:10:24
 * @LastEditTime: 2023-04-06 16:52:49
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

## How to run

**NOTICE: a NVIDIA GPU is required to run the code!**

### Install dependencies

```bash
# if you have mamaba installed, it's faster to use mamba to create a new environment than conda
mamba env create -f environment.yml
# after the environment is created, activate it
conda activate MTL
# check if your pytorch is installed correctly, if cuda is installed, it should return True
python -c "import torch; print(torch.cuda.is_available())"
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

```bash
python train_mtl.py
```

### Test

```bash
python test.py
```
