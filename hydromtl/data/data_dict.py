"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-04-06 17:10:02
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: /HydroMTL/hydromtl/data/data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from hydromtl.data.source.data_camels import Camels

# more data types which cannot be easily treated same with attribute or forcing data
from hydromtl.data.source.data_daymet4camels import Daymet4Camels
from hydromtl.data.source.data_era5land4camels import Era5Land4Camels
from hydromtl.data.loader.data_loaders import (
    BasinFlowDataModel,
    XczDataModel,
)
from hydromtl.data.loader.data_sets import (
    BasinFlowDataset,
    BasinSingleFlowDataset,
)
from hydromtl.data.source.data_modiset4camels import ModisEt4Camels
from hydromtl.data.source.data_nldas4camels import Nldas4Camels
from hydromtl.data.source_pro.camels_series import CamelsSeries
from hydromtl.data.source_pro.data_camels_pro import CamelsPro

other_data_source_list = ["RES_STOR_HIST", "GAGES_TS", "FDC", "DI", "WATER_SURFACE"]

data_sources_dict = {
    "CAMELS": Camels,
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
    "StreamflowDataModel": BasinFlowDataModel,
    "KernelFlowDataModel": XczDataModel,
}
