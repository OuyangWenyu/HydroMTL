"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2022-12-03 16:51:22
LastEditors: Wenyu Ouyang
Description: Select sites according to some conditions
FilePath: \HydroSPB\hydroSPB\data\source_pro\select_gages_ids.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from typing import Union
import numpy as np

from hydroSPB.data.source.data_gages import Gages, get_dor_values
from hydroSPB.data.source_pro.camels_series import CamelsSeries
from hydroSPB.data.source_pro.data_gages_pro import GagesPro
from hydroSPB.data.source.data_camels import Camels
from hydroSPB.data.source.data_constant import Q_CAMELS_US_NAME
from hydroSPB.utils.hydro_utils import is_any_elem_in_a_lst


def dor_reservoirs_chosen(gages: Union[Gages, GagesPro], usgs_id, dor_chosen) -> list:
    """
    choose basins of small DOR(calculated by NOR_STORAGE/RUNAVE7100)

    """

    if type(gages) == GagesPro:
        gages = gages.gages
    dors = get_dor_values(gages, usgs_id)
    if type(dor_chosen) == list or type(dor_chosen) == tuple:
        # right half-open range
        chosen_id = [
            usgs_id[i]
            for i in range(dors.size)
            if dor_chosen[0] <= dors[i] < dor_chosen[1]
        ]
    else:
        if dor_chosen < 0:
            chosen_id = [usgs_id[i] for i in range(dors.size) if dors[i] < -dor_chosen]
        else:
            chosen_id = [usgs_id[i] for i in range(dors.size) if dors[i] >= dor_chosen]

    assert all(x < y for x, y in zip(chosen_id, chosen_id[1:]))
    return chosen_id


def usgs_screen_streamflow(
        gages: Union[Camels, Gages, GagesPro, CamelsSeries],
        usgs_ids: list,
        time_range: list,
        flow_type=Q_CAMELS_US_NAME,
        **kwargs
) -> list:
    """
    according to the criteria and its ancillary condition--thresh of streamflow data,
    choose appropriate ones from the given usgs sites

    Parameters
    ----------
    gages
        Camels, CamelsSeries, Gages or GagesPro object
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
        for criteria in kwargs:
            # if any criteria is not matched, we can filter this site
            if sites_chosen[sites_index[i]] == 0:
                break
            if criteria == "missing_data_ratio":
                nan_length = runoff[np.isnan(runoff)].size
                # then calculate the length of consecutive nan
                thresh = kwargs[criteria]
                if nan_length / runoff.size > thresh:
                    sites_chosen[sites_index[i]] = 0
                else:
                    sites_chosen[sites_index[i]] = 1

            elif criteria == "zero_value_ratio":
                zero_length = runoff.size - np.count_nonzero(runoff)
                thresh = kwargs[criteria]
                if zero_length / runoff.size > thresh:
                    sites_chosen[sites_index[i]] = 0
                else:
                    sites_chosen[sites_index[i]] = 1
            else:
                print(
                    "Oops! That is not valid value. Try missing_data_ratio or zero_value_ratio ..."
                )
    gages_chosen_id = [
        usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0
    ]
    # assert all(x < y for x, y in zip(gages_chosen_id, gages_chosen_id[1:]))
    return gages_chosen_id


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
        if not ecoregion[1] in ec02_code_lst:
            raise NotImplementedError(
                "No such EC02 code, please choose from " + str(ec02_code_lst)
            )
        attr_name = "ECO2_BAS_DOM"
    else:
        if not ecoregion[1] in np.arange(1, 85):
            raise NotImplementedError("No such EC03 code, please choose from 1 - 85")
        attr_name = "ECO3_BAS_DOM"
    attr_lst = [attr_name]
    data_attr = gages.read_constant_cols(site_ids, attr_lst)
    eco_names = data_attr[:, 0]
    chosen_id = [
        site_ids[i] for i in range(eco_names.size) if eco_names[i] == ecoregion[1]
    ]
    return chosen_id


def choose_basins_with_area(
        gages: Union[Camels, Gages, GagesPro, CamelsSeries],
        usgs_ids: list,
        smallest_area: float,
        largest_area: float,
) -> list:
    """
    choose basins with not too large or too small area

    Parameters
    ----------
    gages
        Camels, CamelsSeries, Gages or GagesPro object
    usgs_ids: list
        given sites' ids
    smallest_area
        lower limit; unit is km2
    largest_area
        upper limit; unit is km2

    Returns
    -------
    list
        sites_chosen: [] -- ids of chosen gages

    """
    basins_areas = gages.read_basin_area(usgs_ids).flatten()
    sites_index = np.arange(len(usgs_ids))
    sites_chosen = np.ones(len(usgs_ids))
    for i in range(sites_index.size):
        # loop for every site
        if basins_areas[i] < smallest_area or basins_areas[i] > largest_area:
            sites_chosen[sites_index[i]] = 0
        else:
            sites_chosen[sites_index[i]] = 1
    gages_chosen_id = [
        usgs_ids[i] for i in range(len(sites_chosen)) if sites_chosen[i] > 0
    ]
    # assert all(x < y for x, y in zip(gages_chosen_id, gages_chosen_id[1:]))
    return gages_chosen_id


def diversion_chosen(gages, usgs_id):
    diversion_strs = ["diversion", "divert"]
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    attr_lst = ["WR_REPORT_REMARKS", "SCREENING_COMMENTS"]
    data_attr = gages.read_attr_origin(usgs_id, attr_lst)
    diversion_strs_lower = [elem.lower() for elem in diversion_strs]
    data_attr0_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[0]])
    data_attr1_lower = np.array([elem.lower() if type(elem) == str else elem for elem in data_attr[1]])
    data_attr_lower = np.vstack((data_attr0_lower, data_attr1_lower)).T
    chosen_id = [usgs_id[i] for i in range(len(usgs_id)) if
                 is_any_elem_in_a_lst(diversion_strs_lower, data_attr_lower[i], include=True)]
    return chosen_id


def dam_num_chosen(gages, usgs_id, dam_num):
    """choose basins of dams"""
    assert (all(x < y for x, y in zip(usgs_id, usgs_id[1:])))
    attr_lst = ["NDAMS_2009"]
    data_attr = gages.read_constant_cols(usgs_id, attr_lst)
    if type(dam_num) == list:
        chosen_id = [usgs_id[i] for i in range(data_attr.size) if dam_num[0] <= data_attr[:, 0][i] < dam_num[1]]
    else:
        chosen_id = [usgs_id[i] for i in range(data_attr.size) if data_attr[:, 0][i] == dam_num]
    return chosen_id
