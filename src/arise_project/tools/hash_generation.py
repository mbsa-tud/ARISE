# -*- coding: utf-8 -*-

"""
ICM ARISE Factory Simulation - A modular software platform that decouples simulation from scheduling and enables fair
benchmarking of heterogeneous multi-objective optimization methods.

Copyright (C) 2026 Institute of Industrial Automation and Software Engineering, University of Stuttgart
Primary Author: Patrick Fischer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

----------------------------------------------------------------------------------------------------------------------

Module defining a function used for generating the canonical hash of a JSON file.
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import json
import hashlib

from pathlib import Path

REMOVE_KEY_LIST_SCENARIO_JSON = ["scenario_name", "scenario_note"]
REMOVE_KEY_LIST_DQN_CONFIG_JSON = []


def get_canonical_hash(file_path: Path, char_length: int = 8, remove_key_list: list[str] | None = None) -> str:
    """
    First, open the JSON and load the content as a Python dictionary. Next, remove content that may be variable
    but that doesn't impact the relevant content like spaces or the order of keys. Remove keys that are not
    relevant if changed, for example the date and time of the latest update. Calculate the SHA-256 hex digest.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    if remove_key_list is not None:

        # Remove irrelevant keys
        for key in remove_key_list:
            data_dict.pop(key, None)

    # Remove irrelevant changes in formatting
    canonical_dict = json.dumps(data_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    # Calculate hash
    hash_hex_str = hashlib.sha256(canonical_dict).hexdigest()

    # Limit output characters if value is valid
    if 0 < char_length < len(hash_hex_str):
        hash_hex_str = hash_hex_str[:char_length]

    return hash_hex_str


def get_canonical_hash_scenario_json(file_path: Path, char_length: int = 8) -> str:

    return get_canonical_hash(file_path=file_path,
                              char_length=char_length,
                              remove_key_list=REMOVE_KEY_LIST_SCENARIO_JSON)


def get_canonical_hash_dqn_config_json(file_path: Path, char_length: int = 8) -> str:

    return get_canonical_hash(file_path=file_path,
                              char_length=char_length,
                              remove_key_list=REMOVE_KEY_LIST_DQN_CONFIG_JSON)


def get_scenario_data_dir_path(scenario_file_path: Path) -> Path:
    """
    Calculate canonical hash, create directory name and make directory if it doesn't exist yet.
    """

    hash_str = get_canonical_hash_scenario_json(file_path=scenario_file_path)
    output_dir_path = scenario_file_path.parent / f"sc_data_{hash_str}"

    return output_dir_path
