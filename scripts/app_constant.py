DAYMET_NAME = "daymet"
NLDAS_NAME = "nldas"
ERA5LAND_NAME = "era5land"

SSM_SMAP_NAME = "ssm"
ET_MODIS_NAME = "ET"
ET_ERA5LAND_NAME = "total_evaporation"

# unify streamflow names
Q_CAMELS_US_NAME = "streamflow"
# Q_CAMELS_CC_NAME = "Q_fix"
Q_CAMELS_CC_NAME = "Q"
PRCP_DAYMET_NAME = "prcp"
PRCP_ERA5LAND_NAME = "total_precipitation"
PRCP_NLDAS_NAME = "total_precipitation"
PET_MODIS_NAME = "PET"
PET_DAYMET_NAME = "PET"
PET_ERA5LAND_NAME = "potential_evaporation"
PET_NLDAS_NAME = "potential_evaporation"

VAR_C_CHOSEN_FROM_CAMELS_US = [
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "geol_1st_class",
    "geol_2nd_class",
    "geol_porostiy",
    "geol_permeability",
]

VAR_C_CHOSEN_FROM_GAGES_II = [
    "DRAIN_SQKM",
    "ELEV_MEAN_M_BASIN",
    "SLOPE_PCT",
    "DEVNLCD06",
    "FORESTNLCD06",
    "PLANTNLCD06",
    "WATERNLCD06",
    "SNOWICENLCD06",
    "BARRENNLCD06",
    "SHRUBNLCD06",
    "GRASSNLCD06",
    "WOODYWETNLCD06",
    "EMERGWETNLCD06",
    "AWCAVE",
    "PERMAVE",
    "RFACT",
    "ROCKDEPAVE",
    "GEOL_REEDBUSH_DOM",
    "GEOL_REEDBUSH_DOM_PCT",
    "STREAMS_KM_SQ_KM",
    "NDAMS_2009",
    "STOR_NOR_2009",
    "RAW_DIS_NEAREST_MAJ_DAM",
    "CANALS_PCT",
    "RAW_DIS_NEAREST_CANAL",
    "FRESHW_WITHDRAWAL",
    "POWER_SUM_MW",
    "PDEN_2000_BLOCK",
    "ROADS_KM_SQ_KM",
    "IMPNLCD06",
]

VAR_C_CHOSEN_FROM_CAMELS_CC_FEWER = [
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "high_prec_freq",
    "high_prec_timing",
    "low_prec_freq",
    "low_prec_timing",
    "frac_snow_daily",
    "MixedForest",
    "permeability",
    "porosity",
    "root_depth_99",
    "SNDPPT",
    "CLYPPT",
    "elev",
    "slope",
    "Area",
]

VAR_C_CHOSEN_FROM_CAMELS_CC = [
    "p_mean",
    "pet_mean",
    "aet_mean",
    "aridity",
    "p_seasonality",
    "high_prec_freq",
    "high_prec_dur",
    "high_prec_timing",
    "low_prec_freq",
    "low_prec_dur",
    "low_prec_timing",
    "frac_snow_daily",
    "geol_class_1st",
    "geol_class_1st_frac",
    "MixedForest",
    "permeability",
    "porosity",
    "root_depth_50",
    "root_depth_99",
    "BDTICM_M_250m_ll",
    "SNDPPT",
    "CLYPPT",
    "elev",
    "slope",
    "Area",
]

VAR_T_CHOSEN_FROM_DAYMET = [
    "dayl",
    PRCP_DAYMET_NAME,
    "srad",
    "swe",
    "tmax",
    "tmin",
    "vp",
]

VAR_T_CHOSEN_FROM_NLDAS = [
    PRCP_NLDAS_NAME,
    PET_NLDAS_NAME,
    "temperature",
    "specific_humidity",
    "shortwave_radiation",
    "potential_energy",
]

VAR_T_CHOSEN_FROM_ERA5LAND = [
    PRCP_ERA5LAND_NAME,
    PET_ERA5LAND_NAME,
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_net_solar_radiation",
    "snow_depth_water_equivalent",
]
