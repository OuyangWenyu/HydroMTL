import os
from torchhydro import SETTING

DATA_ORIGIN_DIR = SETTING["local_data_path"]["datasets-origin"]
DATA_INTERIM_DIR = SETTING["local_data_path"]["datasets-interim"]
SOURCE_CFGS = (
    {
        "source_names": [
            "usgs4camels",
            "modiset4camels",
            "nldas4camels",
            "smap4camels",
        ],
        "source_paths": [
            os.path.join(DATA_ORIGIN_DIR, "camels", "camels_us"),
            os.path.join(DATA_INTERIM_DIR, "camels_us", "modiset4camels"),
            os.path.join(DATA_INTERIM_DIR, "camels_us", "nldas4camels"),
            os.path.join(DATA_INTERIM_DIR, "camels_us", "smap4camels"),
        ],
    },
)
NLDASCAMELS_CFGS = {
    "source_names": ["usgs4camels", "nldas4camels"],
    "source_paths": [
        os.path.join(DATA_ORIGIN_DIR, "camels", "camels_us"),
        os.path.join(DATA_INTERIM_DIR, "camels_us", "nldas4camels"),
    ],
}
