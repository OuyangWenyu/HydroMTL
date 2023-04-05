<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-04-05 20:10:24
 * @LastEditTime: 2023-04-05 20:48:57
 * @LastEditors: Wenyu Ouyang
 * @Description: README for HydroMTL
 * @FilePath: \HydroMTL\README.md
 * Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
-->
# HydroMTL

## What is HydroMTL

Multi-task deep learning models for basin hydrological modeling

## How to run

### Install dependencies

```bash
mamba env create -f environment.yml
```

### Prepare data

```bash
python prepare_data.py
```

### Train

```bash
python train.py
```

### Test

```bash
python test.py
```

## Cite

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
