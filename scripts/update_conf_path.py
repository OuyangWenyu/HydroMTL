"""
Author: Wenyu Ouyang
Date: 2024-05-12 11:49:21
LastEditTime: 2024-05-25 16:11:50
LastEditors: Wenyu Ouyang
Description: Update paths in the JSON file to the new path as we may use the saved files in different computers
FilePath: \\HydroMTL\\scripts\\update_conf_path.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import json
import os
import argparse


def update_paths(data, old_path, new_path, old_separator, new_separator):
    """Recursively update paths in the dictionary, returns True if any changes are made."""
    changes_made = False
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and old_path in value:
                data[key] = value.replace(old_path, new_path).replace(
                    old_separator, new_separator
                )
                changes_made = True
            elif isinstance(value, (list, dict)):
                if update_paths(
                    value, old_path, new_path, old_separator, new_separator
                ):
                    changes_made = True
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str) and old_path in item:
                data[i] = item.replace(old_path, new_path).replace(
                    old_separator, new_separator
                )
                changes_made = True
            elif isinstance(item, (dict, list)):
                if update_paths(item, old_path, new_path, old_separator, new_separator):
                    changes_made = True
    return changes_made


def process_json_files(
    directory,
    data_path_old,
    data_path_new,
    code_path_old,
    code_path_new,
    old_separator,
    new_separator,
):
    """Process all JSON files in the directory and its subdirectories."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                process_file(
                    file_path,
                    data_path_old,
                    data_path_new,
                    code_path_old,
                    code_path_new,
                    old_separator,
                    new_separator,
                )


def process_file(
    file_path,
    data_path_old,
    data_path_new,
    code_path_old,
    code_path_new,
    old_separator,
    new_separator,
):
    """Load, update paths, and save the JSON file if changes were made."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Update paths and check if any changes were made
        changes_made = update_paths(
            data, data_path_old, data_path_new, old_separator, new_separator
        )
        changes_made |= update_paths(
            data, code_path_old, code_path_new, old_separator, new_separator
        )

        # Save the modified data only if changes were made
        if changes_made:
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Updated {file_path}")
        else:
            print(f"No changes needed for {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update JSON file paths")
    parser.add_argument(
        "--main_directory",
        type=str,
        default="C:\\Users\\wenyu\\OneDrive\\Research\\paper3-mtl\\results\\camels",
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--data_path_old",
        type=str,
        default="/mnt/data/owen411/data/",
        help="Old data path",
    )
    parser.add_argument(
        "--data_path_new",
        type=str,
        default="C:\\Users\\wenyu\\OneDrive\\data\\",
        help="New data path",
    )
    parser.add_argument(
        "--code_path_old",
        type=str,
        default="/mnt/sdc/owen/code/HydroMTL/",
        help="Old code path",
    )
    parser.add_argument(
        "--code_path_new",
        type=str,
        default="C:\\Users\\wenyu\\OneDrive\\Research\\paper3-mtl\\",
        help="New code path",
    )
    parser.add_argument(
        "--old_separator", type=str, default="/", help="Old path separator"
    )
    parser.add_argument(
        "--new_separator", type=str, default="\\", help="New path separator"
    )

    args = parser.parse_args()

    process_json_files(
        args.main_directory,
        args.data_path_old,
        args.data_path_new,
        args.code_path_old,
        args.code_path_new,
        args.old_separator,
        args.new_separator,
    )
