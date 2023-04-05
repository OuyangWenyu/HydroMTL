"""
A data source class for SMAP basin mean data of CAMELS-basins
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


class Smap4Camels(DataSourceBase):
    """
    A datasource class for geo attributes data, forcing data, and SMAP data of basins in CAMELS.
    """

    def __init__(self, data_path: list, download=False):
        """
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
        return "SMAP_CAMELS"

    def set_data_source_describe(self):
        # forcing
        smap_db = self.data_source_dir
        if not os.path.isdir(smap_db):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        smap_data_dir = os.path.join(smap_db, "NASA_USDA_SMAP_CAMELS")
        return collections.OrderedDict(
            SMAP_CAMELS_DIR=smap_db, SMAP_CAMELS_MEAN_DIR=smap_data_dir
        )

    def download_data_source(self):
        logging.warning("Please use HydroBench to download them!")

    def get_constant_cols(self) -> np.array:
        return self.camels.get_constant_cols()

    def get_relevant_cols(self) -> np.array:
        return self.camels.get_relevant_cols()

    def get_target_cols(self) -> np.array:
        return np.array(["ssm", "susm", "smp", "ssma", "susma"])

    def get_other_cols(self) -> dict:
        pass

    def read_site_info(self) -> pd.DataFrame:
        return self.camels.read_site_info()

    def read_object_ids(self, object_params=None) -> np.array:
        return self.camels.read_object_ids()

    def read_target_cols(
        self, usgs_id_lst=None, t_range=None, target_cols=None, **kwargs
    ):
        """
        Read SMAP basin mean data

        More detials about NASA-USDA Enhanced SMAP data could be seen in:
        https://explorer.earthengine.google.com/#detail/NASA_USDA%2FHSL%2FSMAP10KM_soil_moisture

        Parameters
        ----------
        usgs_id_lst
            the ids of gages in CAMELS
        t_range
            the start and end periods
        target_cols
            the var types

        Returns
        -------
        np.array
            return an np.array
        """
        # Data is not daily. For convenience, we fill NaN values in gap periods.
        # For example, the data is in period 1 (1-3 days), then there is one data in the 1st day while the rest are NaN
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(usgs_id_lst), nt, len(target_cols)])
        for k in tqdm(range(len(usgs_id_lst)), desc="Read NSDA-SMAP data for CAMELS-US"):
            # two way to read data are provided:
            # 1. directly read data: the data is sum of 8 days
            # 2. calculate daily mean value of 8 days
            data = self.read_basin_mean_smap(usgs_id_lst[k], target_cols, t_range_list)
            x[k, :, :] = data
        return x

    def read_basin_mean_smap(self, usgs_id, var_lst, t_range_list):
        logging.debug("reading %s forcing data", usgs_id)
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["SMAP_CAMELS_MEAN_DIR"]
        data_file = os.path.join(
            data_folder, huc, "%s_lump_nasa_usda_smap.txt" % usgs_id
        )
        data_temp = pd.read_csv(data_file, sep=",", header=None, skiprows=1)
        smap_var_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "ssm",
            "susm",
            "smp",
            "ssma",
            "susma",
        ]
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)

        nf = len(var_lst)
        nt = len(t_range_list)
        out = np.full([nt, nf], np.nan)

        for k in range(nf):
            ind = smap_var_lst.index(var_lst[k])
            out[ind2, k] = data_temp[ind].values[ind1]
        return out

    def read_relevant_cols(
        self,
        usgs_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs
    ) -> np.array:
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
