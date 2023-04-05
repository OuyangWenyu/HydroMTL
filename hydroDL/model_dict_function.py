"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2022-11-24 16:44:04
LastEditors: Wenyu Ouyang
Description: Dicts including models (which are seq-first), losses, and optims
FilePath: /HydroSPB/hydroSPB/hydroDL/model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydroSPB.hydroDL.hydros.tl4dpl import LinearDplModel
from hydroSPB.hydroDL.rnns.cudnnlstm import (
    CudnnLstmModel,
    LinearCudnnLstmModel,
    CNN1dLCmodel,
    CudnnLstmModelLstmKernel,
    CudnnLstmModelMultiOutput,
    KuaiLstm,
)
from hydroSPB.hydroDL.cnns.cnnlstm import (
    CnnLstmEra,
    SppCudnnLSTM,
    TimeDisCNNEraLSTM,
    SppLstm,
)
from hydroSPB.hydroDL.rnns.lstm_vanilla import CudaLSTM, OfficialLstm
from torch.optim import Adam, SGD, Adadelta
from hydroSPB.hydroDL.rnns.darnn_model import DARNN
from hydroSPB.hydroDL.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,
    UncertaintyWeights,
    DynamicTaskPrior,
    MultiOutWaterBalanceLoss,
)

from hydroSPB.hydroDL.hydros.dpl4hbv import DplLstmHbv, DplAnnHbv
from hydroSPB.hydroDL.hydros.dpl4xaj import DplLstmXaj, DplAnnXaj
from hydroSPB.hydroDL.hydros.dpl4gr4j import DplLstmGr4j, DplAnnGr4j
from hydroSPB.hydroDL.hydros.dpl4xaj_varparam import DplLstmVarParamXaj
from hydroSPB.hydroDL.hydros.dpl4xaj_nn4et import DplLstmNnModuleXaj

"""
Utility dictionaries to map a string to a class.
"""
# now only those models support sequence-first, others are batch-first
sequence_first_model_lst = [
    OfficialLstm,
    CudnnLstmModel,
    KuaiLstm,
    LinearCudnnLstmModel,
    CNN1dLCmodel,
    CudnnLstmModelLstmKernel,
    SppCudnnLSTM,
    CudnnLstmModelMultiOutput,
    DplLstmXaj,
    DplLstmVarParamXaj,
    DplLstmNnModuleXaj,
    DplLstmHbv,
    DplLstmGr4j,
    DplAnnGr4j,
    DplAnnHbv,
    DplAnnXaj,
    LinearDplModel,
]

pytorch_model_dict = {
    "LSTM": OfficialLstm,
    "FreddyLSTM": CudaLSTM,
    "KuaiLSTM": CudnnLstmModel,
    "KuaiLstm": KuaiLstm,
    "KaiTlLSTM": LinearCudnnLstmModel,
    "DapengCNNLSTM": CNN1dLCmodel,
    "LSTMKernel": CudnnLstmModelLstmKernel,
    "KuaiLSTMMultiOut": CudnnLstmModelMultiOutput,
    "DARNN": DARNN,
    "CnnLstmEra": CnnLstmEra,
    "CnnEraLstm": TimeDisCNNEraLSTM,
    "SppLstm": SppLstm,
    "SppCudnnLSTM": SppCudnnLSTM,
    "DplXaj": DplLstmXaj,
    "DplVarParamXaj": DplLstmVarParamXaj,
    "DplNnModuleXaj": DplLstmNnModuleXaj,
    "DplHbv": DplLstmHbv,
    "DplGr4j": DplLstmGr4j,
    "DplAttrGr4j": DplAnnGr4j,
    "DplAttrHbv": DplAnnHbv,
    "DplAttrXaj": DplAnnXaj,
    "TlDplModel": LinearDplModel,
}

pytorch_model_wrapper_dict = {}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
    "UncertaintyWeights": UncertaintyWeights,
    "DynamicTaskPrior": DynamicTaskPrior,
    "MultiOutWaterBalanceLoss": MultiOutWaterBalanceLoss,
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
