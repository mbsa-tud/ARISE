# -*- coding: utf-8 -*-

"""
Module containing path constants for easy reference throughout the whole project

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from pathlib import Path

DIR_PROJECT_PATH = Path(__file__).parent.parent

DIR_DATA_PATH = DIR_PROJECT_PATH / "data"

DIR_DATA_OUTPUT_PATH = DIR_DATA_PATH / "output"
DIR_DATA_INPUT_PATH = DIR_DATA_PATH / "input"

DIR_GUI_ICONS_PATH = DIR_PROJECT_PATH / "gui" / "icons"
FILE_GUI_ICON_PATH = DIR_GUI_ICONS_PATH / "arise_icon_original.png"

DIR_DATA_INPUT_SCENARIOS_JSON_PATH = DIR_DATA_INPUT_PATH / "scenarios"
FILE_SCENARIO_JSON_SCHEMA_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_schema.json"

DIR_DATA_OUTPUT_DQN_MODELS_PATH = DIR_DATA_OUTPUT_PATH / "dqn_models"
