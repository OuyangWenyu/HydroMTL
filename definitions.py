import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
path = Path(ROOT_DIR)
DATASET_DIR = os.path.join(
    path.parent.parent.absolute(), "data"
)  # This is your Data source directory
RESULT_DIR = os.path.join(
    ROOT_DIR, "results"
)  # This is your result directory
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
print("Please Check your directory:")
print("ROOT_DIR of the repo: ", ROOT_DIR)
print("DATA_SOURCE_DIR of the repo: ", DATASET_DIR)
print("RESULT_DIR of the repo: ", RESULT_DIR)
