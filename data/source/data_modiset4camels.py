"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2022-11-11 10:55:47
LastEditors: Wenyu Ouyang
Description: A data source class for ET basin mean data (from MODIS) of CAMELS-basins
FilePath: /HydroSPB/hydroSPB/data/source/data_modiset4camels.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import collections
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from hydroSPB.data.source.data_base import DataSourceBase
from hydroSPB.data.source.data_camels import Camels
from hydroSPB.utils import hydro_utils


class ModisEt4Camels(DataSourceBase):
    """
    A datasource class for geo attributes data, MODIS ET data of basins in CAMELS.

    Attributes data come from CAMELS.
    ET data include:
        PMLV2 (https://doi.org/10.1016/j.rse.2018.12.031)
        MODIS16A2v006 (https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2?hl=en#bands)
        MODIS16A2v105 (https://developers.google.com/earth-engine/datasets/catalog/MODIS_NTSG_MOD16A2_105?hl=en#description)
    """

    def __init__(self, data_path: list, download=False):
        """
        Initialize a ModisEt4Camels instance.

        Parameters
        ----------
        data_path
            a list including the data file directory for the instance and CAMELS's path
        download
            if True we will download the data

        """
        super().__init__(data_path[0])
        self.camels = Camels(data_path[1])
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.camels671_sites = self.read_site_info()

    def get_name(self):
        return "MODIS_ET_CAMELS"

    def set_data_source_describe(self):
        et_db = self.data_source_dir
        # ET
        et_basin_mean_dir = os.path.join(et_db, "basin_mean_forcing")
        modisa16v105_dir = os.path.join(et_basin_mean_dir, "MOD16A2_105_CAMELS")
        modisa16v006_dir = os.path.join(et_basin_mean_dir, "MOD16A2_006_CAMELS")
        pmlv2_dir = os.path.join(et_basin_mean_dir, "PML_V2_CAMELS")
        if not os.path.isdir(et_basin_mean_dir):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        return collections.OrderedDict(
            MODIS_ET_CAMELS_DIR=et_db,
            MODIS_ET_CAMELS_MEAN_DIR=et_basin_mean_dir,
            MOD16A2_CAMELS_DIR=modisa16v006_dir,
            PMLV2_CAMELS_DIR=pmlv2_dir,
        )

    def download_data_source(self):
        logging.warning("Please use HydroBench to download them!")

    def get_constant_cols(self) -> np.array:
        return self.camels.get_constant_cols()

    def get_relevant_cols(self) -> np.array:
        return self.camels.get_relevant_cols()  # MODIS16AV105

    def get_target_cols(self) -> np.array:
        return np.array(
            [  # PMLV2
                "GPP",
                "Ec",
                "Es",
                "Ei",
                "ET_water",
                # PML_V2's ET = Ec + Es + Ei
                "ET_sum",
                # MODIS16A2
                "ET",
                "LE",
                "PET",
                "PLE",
                "ET_QC",
            ]
        )

    def get_other_cols(self) -> dict:
        pass

    def read_site_info(self) -> pd.DataFrame:
        return self.camels.read_site_info()

    def read_object_ids(self, object_params=None) -> np.array:
        return self.camels.read_object_ids()

    def read_target_cols(
        self,
        usgs_id_lst=None,
        t_range=None,
        target_cols=None,
        reduce_way="mean",
        **kwargs
    ):
        """
        Read ET data.

        Parameters
        ----------
        usgs_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        target_cols
            the forcing var types
        reduce_way
            how to do "reduce" -- mean or sum; the default is "mean"

        Returns
        -------
        np.array
            return an np.array
        """

        assert len(t_range) == 2
        assert all(x < y for x, y in zip(usgs_id_lst, usgs_id_lst[1:]))
        # Data is not daily. For convenience, we fill NaN values in gap periods.
        # For example, the data is in period 1 (1-8 days), then there is one data in the 1st day while the rest are NaN
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(usgs_id_lst), nt, len(target_cols)])
        for k in tqdm(range(len(usgs_id_lst)), desc="Read MODIS ET data for CAMELS-US"):
            # two way to read data are provided:
            # 1. directly read data: the data is sum of 8 days
            # 2. calculate daily mean value of 8 days
            data = self.read_basin_mean_modiset(
                usgs_id_lst[k], target_cols, t_range_list, reduce_way=reduce_way
            )
            x[k, :, :] = data
        return x

    def read_basin_mean_modiset(
        self, usgs_id, var_lst, t_range_list, reduce_way
    ) -> np.array:
        """
        Read modis ET from PMLV2 and MOD16A2

        Parameters
        ----------
        usgs_id
            ids of basins
        var_lst
            et variables from PMLV2 or/and MOD16A2
        t_range_list
            daily datetime list
        reduce_way
            how to do "reduce" -- mean or sum; the default is "sum"

        Returns
        -------
        np.array
            ET data
        """
        logging.debug("reading %s forcing data", usgs_id)
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        modis16a2_data_folder = self.data_source_description["MOD16A2_CAMELS_DIR"]
        pmlv2_data_folder = self.data_source_description["PMLV2_CAMELS_DIR"]
        pmlv2_data_file = os.path.join(
            pmlv2_data_folder, huc, "%s_lump_pmlv2_et.txt" % usgs_id
        )
        modis16a2_data_file = os.path.join(
            modis16a2_data_folder, huc, "%s_lump_modis16a2v006_et.txt" % usgs_id
        )
        pmlv2_data_temp = pd.read_csv(pmlv2_data_file, header=None, skiprows=1)
        modis16a2_data_temp = pd.read_csv(modis16a2_data_file, header=None, skiprows=1)
        pmlv2_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "GPP",
            "Ec",
            "Es",
            "Ei",
            "ET_water",
            "ET_sum",
        ]  # PMLV2
        modis16a2_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "ET",
            "LE",
            "PET",
            "PLE",
            "ET_QC",
        ]  # MODIS16A2
        df_date_pmlv2 = pmlv2_data_temp[[0, 1, 2]]
        df_date_pmlv2.columns = ["year", "month", "day"]
        df_date_modis16a2 = modis16a2_data_temp[[0, 1, 2]]
        df_date_modis16a2.columns = ["year", "month", "day"]
        ind1_pmlv2, ind2_pmlv2, t_range_final_pmlv2 = self.date_intersect(
            df_date_pmlv2, t_range_list
        )
        (
            ind1_modis16a2,
            ind2_modis16a2,
            t_range_final_modis16a2,
        ) = self.date_intersect(df_date_modis16a2, t_range_list)

        nf = len(var_lst)
        nt = len(t_range_list)
        out = np.full([nt, nf], np.nan)

        for k in range(nf):
            if var_lst[k] in pmlv2_lst:
                if len(t_range_final_pmlv2) == 0:
                    # no data, just skip this var
                    continue
                if var_lst[k] == "ET_sum":
                    # No such item in original PML_V2 data
                    et_3components = self.read_basin_mean_modiset(
                        usgs_id, ["Ec", "Es", "Ei"], t_range_list, reduce_way
                    )
                    # it si equal to sum of 3 components
                    out[:, k] = np.sum(et_3components, axis=-1)
                    continue
                ind = pmlv2_lst.index(var_lst[k])
                if reduce_way == "sum":
                    out[ind2_pmlv2, k] = pmlv2_data_temp[ind].values[ind1_pmlv2]
                elif reduce_way == "mean":
                    days_interval = [y - x for x, y in zip(ind2_pmlv2, ind2_pmlv2[1:])]
                    if (
                        t_range_final_pmlv2[-1].item().month == 12
                        and t_range_final_pmlv2[-1].item().day == 31
                    ):
                        final_timedelta = (
                            t_range_final_pmlv2[-1].item()
                            - t_range_final_pmlv2[ind2_pmlv2[-1]].item()
                        )
                        final_day_interval = [final_timedelta.days]
                    else:
                        final_day_interval = [8]
                    days_interval = np.array(days_interval + final_day_interval)
                    # there may be some missing data, so that some interval will be larger than 8
                    days_interval[np.where(days_interval > 8)] = 8
                    out[ind2_pmlv2, k] = (
                        pmlv2_data_temp[ind].values[ind1_pmlv2] / days_interval
                    )
                else:
                    raise NotImplementedError("We don't have such a reduce way")
            elif var_lst[k] in modis16a2_lst:
                if len(t_range_final_modis16a2) == 0:
                    # no data, just skip this var
                    continue
                ind = modis16a2_lst.index(var_lst[k])
                if reduce_way == "sum":
                    out[ind2_modis16a2, k] = modis16a2_data_temp[ind].values[
                        ind1_modis16a2
                    ]
                elif reduce_way == "mean":
                    days_interval = [
                        y - x for x, y in zip(ind2_modis16a2, ind2_modis16a2[1:])
                    ]
                    if (
                        t_range_final_modis16a2[-1].item().month == 12
                        and t_range_final_modis16a2[-1].item().day == 31
                    ):
                        final_timedelta = (
                            t_range_final_modis16a2[-1].item()
                            - t_range_final_modis16a2[ind2_modis16a2[-1]].item()
                        )
                        final_day_interval = [final_timedelta.days]
                    else:
                        final_day_interval = [8]
                    days_interval = np.array(days_interval + final_day_interval)
                    # there may be some missing data, so that some interval will be larger than 8
                    days_interval[np.where(days_interval > 8)] = 8
                    out[ind2_modis16a2, k] = (
                        modis16a2_data_temp[ind].values[ind1_modis16a2] / days_interval
                    )
                else:
                    raise NotImplementedError("We don't have such a reduce way")
            else:
                raise NotImplementedError("No such var type now")
        # unit is 0.1mm/day(or 8/5/6days), so multiply it with 0.1 to transform to mm/day(or 8/5/6days))
        # TODO: only valid for MODIS, for PMLV2, we need to check the unit 
        out = out * 0.1
        return out

    @staticmethod
    def date_intersect(df_date, t_range_list):
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        if (
            np.datetime64(str(date[-1].astype(object).year) + "-12-31")
            > date[-1]
            > np.datetime64(str(date[-1].astype(object).year) + "-12-24")
        ):
            final_date = np.datetime64(str(date[-1].astype(object).year + 1) + "-01-01")
        else:
            final_date = date[-1] + np.timedelta64(8, "D")
        date_all = hydro_utils.t_range_days(
            hydro_utils.t_days_lst2range([date[0], final_date])
        )
        t_range_final = np.intersect1d(date_all, t_range_list)
        [c, ind1, ind2] = np.intersect1d(date, t_range_final, return_indices=True)
        return ind1, ind2, t_range_final

    def read_relevant_cols(
        self,
        usgs_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs
    ) -> np.array:
        """
        Read CAMELS forcing data

        Parameters
        ----------
        usgs_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        var_lst
            the forcing var types

        Returns
        -------
        np.array
            3-dim forcing data array -- [seq, batch, feature]
        """

        return self.camels.read_relevant_cols(usgs_id_lst, t_range, var_lst)

    def read_constant_cols(self, usgs_id_lst=None, var_lst=None, is_return_dict=False):
        return self.camels.read_constant_cols(usgs_id_lst, var_lst, is_return_dict)

    def read_other_cols(self, object_ids=None, other_cols: dict = None, **kwargs):
        pass

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(
            object_ids, ["area_gages2"], is_return_dict=False
        )

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)
