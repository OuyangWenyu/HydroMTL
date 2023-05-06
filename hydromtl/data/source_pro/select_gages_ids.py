"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2023-04-28 15:19:31
LastEditors: Wenyu Ouyang
Description: Select sites according to some conditions
FilePath: /HydroMTL/hydromtl/data/source_pro/select_gages_ids.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from typing import Union
import numpy as np

from hydromtl.data.source_pro.camels_series import CamelsSeries
from hydromtl.data.source.data_camels import Camels
from hydromtl.data.source.data_constant import Q_CAMELS_US_NAME
from hydromtl.data.source.data_gages import Gages


def usgs_screen_streamflow(
    gages: Union[Camels, CamelsSeries],
    usgs_ids: list,
    time_range: list,
    flow_type=Q_CAMELS_US_NAME,
    **kwargs,
) -> list:
    """
    according to the criteria and its ancillary condition--thresh of streamflow data,
    choose appropriate ones from the given usgs sites

    Parameters
    ----------
    gages
        Camels, CamelsSeries object
    usgs_ids: list
        given sites' ids
    time_range: list
        chosen time range
    flow_type
        flow's name in data file; default is usgsFlow for CAMELS-US
    kwargs
        all criteria

    Returns
    -------
    list
        sites_chosen: [] -- ids of chosen gages

    Examples
    --------
        >>> usgs_screen_streamflow(gages, ["02349000","08168797"], ["1995-01-01","2015-01-01"], **{'missing_data_ratio': 0, 'zero_value_ratio': 1})
    """
    usgs_values = gages.read_target_cols(usgs_ids, time_range, [flow_type])[:, :, 0]
    sites_index = np.arange(usgs_values.shape[0])
    sites_chosen = np.ones(usgs_values.shape[0])
    for i in range(sites_index.size):
        # loop for every site
        runoff = usgs_values[i, :]
        for criteria, thresh in kwargs.items():
            # if any criteria is not matched, we can filter this site
            if sites_chosen[sites_index[i]] == 0:
                break
            if criteria == "missing_data_ratio":
                nan_length = runoff[np.isnan(runoff)].size
                sites_chosen[sites_index[i]] = (
                    0 if nan_length / runoff.size > thresh else 1
                )
            elif criteria == "zero_value_ratio":
                zero_length = runoff.size - np.count_nonzero(runoff)
                thresh = kwargs[criteria]
                sites_chosen[sites_index[i]] = (
                    0 if zero_length / runoff.size > thresh else 1
                )
            else:
                print(
                    "Oops! That is not valid value. Try missing_data_ratio or zero_value_ratio ..."
                )
    return [usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0]


def choose_sites_in_ecoregion(
    gages: Gages, site_ids: list, ecoregion: Union[list, tuple]
) -> list:
    """
    Choose sites in ecoregions

    Parameters
    ----------
    gages : Gages
        Only gages dataset has ecoregion attribute
    site_ids : list
        all ids of sites
    ecoregion : Union[list, tuple]
        which ecoregions

    Returns
    -------
    list
        chosen sites' ids

    Raises
    ------
    NotImplementedError
        PLease choose 'ECO2_CODE' or 'ECO3_CODE'
    NotImplementedError
        must be in EC02 code list
    NotImplementedError
        must be in EC03 code list
    """
    if ecoregion[0] not in ["ECO2_CODE", "ECO3_CODE"]:
        raise NotImplementedError("PLease choose 'ECO2_CODE' or 'ECO3_CODE'")
    if ecoregion[0] == "ECO2_CODE":
        ec02_code_lst = [
            5.2,
            5.3,
            6.2,
            7.1,
            8.1,
            8.2,
            8.3,
            8.4,
            8.5,
            9.2,
            9.3,
            9.4,
            9.5,
            9.6,
            10.1,
            10.2,
            10.4,
            11.1,
            12.1,
            13.1,
        ]
        if ecoregion[1] not in ec02_code_lst:
            raise NotImplementedError(
                f"No such EC02 code, please choose from {ec02_code_lst}"
            )
        attr_name = "ECO2_BAS_DOM"
    elif ecoregion[1] in np.arange(1, 85):
        attr_name = "ECO3_BAS_DOM"
    else:
        raise NotImplementedError("No such EC03 code, please choose from 1 - 85")
    attr_lst = [attr_name]
    data_attr = gages.read_constant_cols(site_ids, attr_lst)
    eco_names = data_attr[:, 0]
    return [site_ids[i] for i in range(eco_names.size) if eco_names[i] == ecoregion[1]]
