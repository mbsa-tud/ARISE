# -*- coding: utf-8 -*-

"""
Module containing path constants for easy reference throughout the whole project

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from pathlib import Path

DIR_PROJECT_PATH = Path(__file__).parent.parent

DIR_DATA_PATH = DIR_PROJECT_PATH / "data"

DIR_DATA_OUTPUT_PATH = DIR_DATA_PATH / "output"

DIR_GUI_ICONS_PATH = DIR_PROJECT_PATH / "gui" / "icons"
FILE_GUI_ICON_PATH = DIR_GUI_ICONS_PATH / "arise_icon_original.png"

DIR_DATA_INPUT_JSON_SCHEMAS_PATH = DIR_DATA_PATH / "schemas"
FILE_SCENARIO_JSON_SCHEMA_PATH = DIR_DATA_INPUT_JSON_SCHEMAS_PATH / "scenario_schema.json"
FILE_DQN_CONFIG_JSON_PATH = DIR_DATA_INPUT_JSON_SCHEMAS_PATH / "dqn_config_schema.json"


DIR_DATA_INPUT_SCENARIOS_JSON_PATH = DIR_DATA_PATH / "scenarios"

DIR_NAME_DQN = "dqn"
DIR_NAME_LOGS = "logs"
DIR_NAME_MODELS = "models"
DIR_NAME_BACKUP = "backup"
FILE_NAME_BEST_MODEL = "best_model.zip"
FILE_NAME_FINAL_MODEL = "final_model.zip"
FILE_NAME_DQN_CONFIG = "dqn_config.json"

FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "sc_simple_plate_factory" / "sc_simple_plate_factory.json"
FILE_SCENARIO_SIMPLE_PLATE_FACTORY_RR_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "sc_simple_plate_factory_rr" / "sc_simple_plate_factory_rr.json"
