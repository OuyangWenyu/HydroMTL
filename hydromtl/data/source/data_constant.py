"""
Author: Wenyu Ouyang
Date: 2022-12-02 17:52:48
LastEditTime: 2023-04-06 16:00:23
LastEditors: Wenyu Ouyang
Description: Some constant for data processing
FilePath: /HydroMTL/hydromtl/data/source/data_constant.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
# TODO: only for CAMELS_CC and CAMELS_US data
from hydromtl.utils.hydro_constant import HydroVar


DAYMET_NAME = "daymet"
NLDAS_NAME = "nldas"
ERA5LAND_NAME = "era5land"

SSM_SMAP_NAME = "ssm"
ET_MODIS_NAME = "ET"
ET_ERA5LAND_NAME = "total_evaporation"

Q_CAMELS_US_NAME = "usgsFlow"
# Q_CAMELS_CC_NAME = "Q_fix"
Q_CAMELS_CC_NAME = "Q"
PRCP_DAYMET_NAME = "prcp"
PRCP_ERA5LAND_NAME = "total_precipitation"
PRCP_NLDAS_NAME = "total_precipitation"
PET_MODIS_NAME = "PET"
PET_DAYMET_NAME = "PET"
PET_ERA5LAND_NAME = "potential_evaporation"
PET_NLDAS_NAME = "potential_evaporation"

streamflow_camels_us = HydroVar(
    name=Q_CAMELS_US_NAME,
    unit="ft3/s",
    ChineseName="径流",
)
evapotranspiration_modis_camels_us = HydroVar(
    name=ET_MODIS_NAME,
    unit="mm/day",
    ChineseName="蒸散发",
)
surface_soil_moisture_smap_camels_us = HydroVar(
    name=SSM_SMAP_NAME,
    unit="mm/day",
    ChineseName="表层土壤含水量",
)
