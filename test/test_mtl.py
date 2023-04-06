"""
Author: Wenyu Ouyang
Date: 2023-04-06 14:45:34
LastEditTime: 2023-04-06 15:16:05
LastEditors: Wenyu Ouyang
Description: Test the multioutput model
FilePath: /HydroMTL/test/test_mtl.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import torch
from hydromtl.models.cudnnlstm import CudnnLstmModelMultiOutput

def test_cudnnlstm_multioutput(device):
    # batch, seq, features
    x = torch.rand(20, 30, 10)
    model = CudnnLstmModelMultiOutput(10, 2, 32)
    # there must be a gpu to perform calculation
    x = x.to(device)
    model.to(device)
    output = model(x)
    np.testing.assert_array_equal(output.shape, (20, 30, 2))