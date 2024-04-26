"""
Author: Wenyu Ouyang
Date: 2022-01-01 23:35:07
LastEditTime: 2024-04-26 21:04:58
LastEditors: Wenyu Ouyang
Description: A data source class for CAMELS with more data (such as NLDAS, MODIS-ET, SMAP) model
FilePath: \HydroMTL\hydromtl\data\source_pro\data_camels_pro.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import collections

import numpy as np

from hydromtl.data.source.data_base import DataSourceBase
from hydromtl.data.source.data_modiset4camels import ModisEt4Camels
from hydromtl.data.source.data_nldas4camels import Nldas4Camels
from hydromtl.data.source.data_smap4camels import Smap4Camels
from hydromtl.utils import hydro_utils


class CamelsPro(DataSourceBase):
    def __init__(self, data_path: list, download=False, et_product="MOD16A2V006"):
        """
        A data source class for camels attrs, forcings and streamflow + ET data + SMAP data

        Parameters
        ----------
        data_path
            the paths of all necessary data
        download
            if True, download the dataset
        """
        # data_path is a list: now the Convention --
        # 0->camels_flow_et 1->camels, 2->modiset4camels, 3->nldas4camels, 4->smap4camels
        # camels_flow_et is a virtual data_source, not from any real data sources, just for this class
        super().__init__(data_path[0])
        self.modiset4basins = ModisEt4Camels(
            [data_path[1], data_path[2]], download=download, et_product=et_product
        )
        self.nldas4camels = Nldas4Camels([data_path[3], data_path[2]])
        self.smap4camels = Smap4Camels([data_path[4], data_path[2]])
        self.data_source_description = self.set_data_source_describe()

    def get_name(self):
        # TODO: old name, should be updated
        return "CAMELS_FLOW_ET"

    def set_data_source_describe(self):
        camels_data_source_description = (
            self.modiset4basins.camels.data_source_description
        )
        modiset4basins_data_source_description = (
            self.modiset4basins.data_source_description
        )
        nldas4camels_data_source_description = self.nldas4camels.data_source_description
        smap4camels_data_source_description = self.smap4camels.data_source_description
        return collections.OrderedDict(
            **camels_data_source_description,
            **modiset4basins_data_source_description,
            **nldas4camels_data_source_description,
            **smap4camels_data_source_description
        )

    def download_data_source(self):
        self.modiset4basins.camels.download_data_source()
        self.modiset4basins.download_data_source()
        self.nldas4camels.download_data_source()
        self.smap4camels.download_data_source()

    def read_object_ids(self, object_params=None) -> np.array:
        return self.modiset4basins.read_object_ids()

    def read_target_cols(
        self, object_ids=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        """
        outputs include camels' streamflow and/or modiset's ET and/or SMAP

        et_data's time interval is different from streamflow_data's, so we need to fix it
        for MODIS16A2 data:
        according to this introduction -- https://developers.google.com/earth-engine/datasets/catalog/MODIS_NTSG_MOD16A2_105?hl=en#description,
        the period of coverage is 8 days (with the exception of the last period at the end of the year which is either 5 or 6 days)
        and the data is aggregated for period of coverage, such as ET(evapotranspiration) and PET
        or averaged daily over the period of coverage, such as LE and PLE
        for PML_V2 data:
        the paper (https://doi.org/10.1016/j.rse.2018.12.031) said it use 8-day average or aggregated data.
        According to this user guide: https://nsidc.org/data/MOD10A2/versions/6
        the 8-day means the following eight days, for example, period 1 means day 1 to 8.
        This Q&A also mentioned this:
        https://www.researchgate.net/post/Does-the-8day-composite-period-of-MODIS-ET-PET-values-refer-to-the-8days-prior-to-the-reported-date-or-following

        The unit of ETs is 0.1mm/8day, so we should multiply 0.1 to get the real ET/PET values in mm/8day (if we want to use the 8-day average data, it is mm/day)

        Here we only fill the gap periods with NaN values, later in model training,
        we need calculate the average or aggregated prediction values

        for SMAP data:
        its Cadence is 3 days which means one value every 3 days
        https://explorer.earthengine.google.com/#detail/NASA_USDA%2FHSL%2FSMAP10KM_soil_moisture

        Parameters
        ----------
        object_ids
            ids of sites
        t_range_list
            period range
        target_cols
            streamflow and ET
        kwargs
            Optional

        Returns
        -------
        np.array
            streamflow and ET data
        """
        nt = hydro_utils.t_range_days(t_range_list).shape[0]
        q_or_et_or_smap = np.empty([len(object_ids), nt, len(target_cols)])
        flow_vars = self.modiset4basins.camels.get_target_cols()
        et_vars = self.modiset4basins.get_target_cols()
        smap_vars = self.smap4camels.get_target_cols()
        et_product = kwargs.get("et_product", "MOD16A2V006")
        for i in range(len(target_cols)):
            if target_cols[i] in flow_vars:
                q_or_et_or_smap[:, :, i : i + 1] = (
                    self.modiset4basins.camels.read_target_cols(
                        object_ids, t_range_list, target_cols[i : i + 1]
                    )
                )
            elif target_cols[i] in et_vars:
                q_or_et_or_smap[:, :, i : i + 1] = self.modiset4basins.read_target_cols(
                    object_ids,
                    t_range_list,
                    target_cols[i : i + 1],
                    et_product=et_product,
                )
            elif target_cols[i] in smap_vars:
                q_or_et_or_smap[:, :, i : i + 1] = self.smap4camels.read_target_cols(
                    object_ids, t_range_list, target_cols[i : i + 1]
                )
            else:
                raise NotImplementedError(
                    "This type is not in included; Please check your input!"
                )
        return q_or_et_or_smap

    def read_relevant_cols(
        self, object_ids=None, t_range_list=None, relevant_cols=None, **kwargs
    ) -> np.array:
        """
        Read basin mean nldas/daymet forcing data.

        Parameters
        ----------
        object_ids
            the ids of gages in CAMELS
        t_range_list
            the start and end periods
        relevant_cols
            the forcing var types

        Returns
        -------
        np.array
            basin mean nldas data
        """
        if kwargs["forcing_type"] == "nldas":
            return self.nldas4camels.read_relevant_cols(
                object_ids, t_range_list, relevant_cols
            )
        elif kwargs["forcing_type"] == "daymet":
            return self.modiset4basins.read_relevant_cols(
                object_ids, t_range_list, relevant_cols
            )
        else:
            raise NotImplementedError(
                "This forcing type is not provided yet!!Please choose daymet or nldas"
            )

    def read_constant_cols(
        self, object_ids=None, constant_cols: list = None, **kwargs
    ) -> np.array:
        return self.modiset4basins.read_constant_cols(object_ids, constant_cols)

    def get_constant_cols(self) -> np.array:
        return self.modiset4basins.get_constant_cols()

    def get_relevant_cols(self) -> np.array:
        # NLDAS + daymet
        forcing_cols_nldas = self.nldas4camels.get_relevant_cols()
        forcing_cols_daymet = self.modiset4basins.get_relevant_cols()
        return np.array(forcing_cols_nldas.tolist() + forcing_cols_daymet.tolist())

    def get_target_cols(self):
        # flow + et + sm
        return np.array(
            self.nldas4camels.get_target_cols().tolist()
            + self.modiset4basins.get_target_cols().tolist()
            + self.smap4camels.get_target_cols().tolist()
        )

    def get_other_cols(self) -> dict:
        pass

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs):
        pass

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(
            object_ids, ["area_gages2"], is_return_dict=False
        )

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)
