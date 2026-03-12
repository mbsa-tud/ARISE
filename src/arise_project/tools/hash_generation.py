# -*- coding: utf-8 -*-

"""
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

REMOVE_KEY_LIST_SCENARIO_JSON = ["scenario_name"]
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
