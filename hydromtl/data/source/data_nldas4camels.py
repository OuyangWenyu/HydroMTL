"""
A data source class for NLDAS basin mean data of CAMELS-basins
"""
import collections
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from hydromtl.data.source.data_base import DataSourceBase
from hydromtl.data.source.data_camels import Camels
from hydromtl.utils import hydro_utils


class Nldas4Camels(DataSourceBase):
    """
    A datasource class for geo attributes data, NLDAS v2 forcing data, and streamflow data of basins in CAMELS.

    The forcing data are basin mean values. Attributes and streamflow data come from CAMELS.
    """

    def __init__(self, data_path: list, download=False):
        """
        Initialize a Nldas4Camels instance.

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
        return "NLDAS_CAMELS"

    def set_data_source_describe(self):
        nldas_db = self.data_source_dir
        # shp file of basins
        camels671_shp = self.camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
        # forcing
        nldas_forcing_basin_mean_dir = os.path.join(nldas_db, "basin_mean_forcing")
        if not os.path.isdir(nldas_forcing_basin_mean_dir):
            raise NotADirectoryError(
                "Please check if you have downloaded the data and put it in the correct dir"
            )
        return collections.OrderedDict(
            NLDAS_CAMELS_DIR=nldas_db,
            CAMELS_SHP_FILE=camels671_shp,
            NLDAS_CAMELS_MEAN_DIR=nldas_forcing_basin_mean_dir,
        )

    def download_data_source(self):
        logging.warning(
            "The data files are too large. Please use HydroBench to download them!"
        )

    def get_constant_cols(self) -> np.array:
        return self.camels.get_constant_cols()

    def get_relevant_cols(self) -> np.array:
        return np.array(
            [
                "temperature",
                "specific_humidity",
                "pressure",
                "wind_u",
                "wind_v",
                "longwave_radiation",
                "convective_fraction",
                "shortwave_radiation",
                "potential_energy",
                "potential_evaporation",
                "total_precipitation",
            ]
        )

    def get_target_cols(self) -> np.array:
        return self.camels.get_target_cols()

    def get_other_cols(self) -> dict:
        pass

    def read_site_info(self) -> pd.DataFrame:
        return self.camels.read_site_info()

    def read_object_ids(self, object_params=None) -> np.array:
        return self.camels.read_object_ids()

    def read_target_cols(
            self, usgs_id_lst=None, t_range=None, target_cols=None, **kwargs
    ):
        return self.camels.read_target_cols(usgs_id_lst, t_range, target_cols)

    def read_basin_mean_nldas(self, usgs_id, var_lst, t_range_list):
        logging.debug("reading %s forcing data", usgs_id)
        gage_id_df = self.camels671_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.data_source_description["NLDAS_CAMELS_MEAN_DIR"]
        data_file = os.path.join(
            data_folder, huc, "%s_lump_nldas_forcing_leap.txt" % usgs_id
        )
        data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, skiprows=1)
        forcing_lst = [
            "Year",
            "Mnth",
            "Day",
            "Hr",
            "temperature",
            "specific_humidity",
            "pressure",
            "wind_u",
            "wind_v",
            "longwave_radiation",
            "convective_fraction",
            "shortwave_radiation",
            "potential_energy",
            "potential_evaporation",
            "total_precipitation",
        ]
        # unit of potential_evaporation and total_precipitation is kg/m^2 (for a day),
        # we use rho=1000kg/m^3 as water's density to transform these two variables‘ unit to mm/day
        # so it's 10^-3 m /day and it is just mm/day, hence we don't need to transform actually
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            ind = forcing_lst.index(var_lst[k])
            out[:, k] = data_temp[ind].values[ind1]
        return out

    def read_relevant_cols(
            self,
            usgs_id_lst: list = None,
            t_range: list = None,
            var_lst: list = None,
            **kwargs
    ) -> np.array:
        """
        Read forcing data.

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
            return an np.array
        """

        assert len(t_range) == 2
        assert all(x < y for x, y in zip(usgs_id_lst, usgs_id_lst[1:]))

        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
        for k in tqdm(range(len(usgs_id_lst)), desc="Read NLDAS forcing data for CAMELS-US"):
            data = self.read_basin_mean_nldas(usgs_id_lst[k], var_lst, t_range_list)
            x[k, :, :] = data
        return x

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
