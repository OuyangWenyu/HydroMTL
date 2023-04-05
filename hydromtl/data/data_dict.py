"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2022-03-20 11:30:05
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: /HydroSPB/hydroSPB/data/data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydroSPB.data.source.data_camels import Camels

# more data types which cannot be easily treated same with attribute or forcing data
from hydroSPB.data.source.data_daymet4camels import Daymet4Camels
from hydroSPB.data.source.data_era5land4camels import Era5Land4Camels
from hydroSPB.data.source.data_gages import Gages
from hydroSPB.data.loader.data_loaders import (
    BasinFlowDataModel,
    XczDataModel,
    DplDataModel,
)
from hydroSPB.data.loader.data_sets import (
    BasinFlowDataset,
    BasinSingleFlowDataset,
    DplDataset,
)
from hydroSPB.data.source.data_modiset4camels import ModisEt4Camels
from hydroSPB.data.source.data_nldas4camels import Nldas4Camels
from hydroSPB.data.source_pro.camels_series import CamelsSeries
from hydroSPB.data.source_pro.data_camels_pro import CamelsPro
from hydroSPB.data.source_pro.data_gages_pro import GagesPro
from hydroSPB.data.loader.xr_dataloader import XarrayDataModel

other_data_source_list = ["RES_STOR_HIST", "GAGES_TS", "FDC", "DI", "WATER_SURFACE"]

data_sources_dict = {
    "CAMELS": Camels,
    "GAGES": Gages,
    "GAGES_PRO": GagesPro,
    "CAMELS_DAYMET_V4": Daymet4Camels,
    "NLDAS_CAMELS": Nldas4Camels,
    "ERA5LAND_CAMELS": Era5Land4Camels,
    "MODIS_ET_CAMELS": ModisEt4Camels,
    "CAMELS_FLOW_ET": CamelsPro,
    "CAMELS_SERIES": CamelsSeries,
}

dataloaders_dict = {
    "StreamflowDataset": BasinFlowDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "StreamflowDataModel": BasinFlowDataModel,
    "KernelFlowDataModel": XczDataModel,
    "XarrayDataModel": XarrayDataModel,
    "DplDataModel": DplDataModel,
}
