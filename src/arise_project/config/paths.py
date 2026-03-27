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

DIR_NAME_OPT_RESULTS = "opt_results"

FILE_NAME_OPT_RESULT_A_STAR_PKL = "opt_result_a_star.pkl"
FILE_NAME_OPT_RESULT_DFS_PKL = "opt_result_dfs.pkl"
FILE_NAME_OPT_RESULT_IDDFS_PKL = "opt_result_iddfs.pkl"
FILE_NAME_OPT_RESULT_NSGA2_PKL = "opt_result_nsga2.pkl"
FILE_NAME_OPT_RESULT_NSGA3_PKL = "opt_result_nsga3.pkl"
FILE_NAME_OPT_RESULT_RL_DQN_PKL = "opt_result_rl_dqn.pkl"
FILE_NAME_OPT_RESULT_HUMAN_PKL = "opt_result_human.pkl"

FILE_NAME_OPT_RESULT_TASKS_A_STAR_CSV = "opt_result_tasks_a_star.csv"
FILE_NAME_OPT_RESULT_TASKS_DFS_CSV = "opt_result_tasks_dfs.csv"
FILE_NAME_OPT_RESULT_TASKS_IDDFS_CSV = "opt_result_tasks_iddfs.csv"
FILE_NAME_OPT_RESULT_TASKS_NSGA2_CSV = "opt_result_tasks_nsga2.csv"
FILE_NAME_OPT_RESULT_TASKS_NSGA3_CSV = "opt_result_tasks_nsga3.csv"
FILE_NAME_OPT_RESULT_TASKS_RL_DQN_CSV = "opt_result_tasks_rl_dqn.csv"
FILE_NAME_OPT_RESULT_TASKS_HUMAN_CSV = "opt_result_tasks_human.csv"

FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "sc_plate_1P_1S-2D-2C-2M-1A" / "sc_plate_1P_1S-2D-2C-2M-1A.json"
FILE_SCENARIO_SIMPLE_PLATE_FACTORY_RR_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "sc_plate_1P_1S-2D-2C-2M-1A_RR" / "sc_plate_1P_1S-2D-2C-2M-1A_RR.json"
FILE_SCENARIO_SIMPLE_PLATE_FACTORY_DUAL_AGV_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "sc_plate_1P_1S-2D-2C-2M-2A" / "sc_plate_1P_1S-2D-2C-2M-2A.json"
FILE_SCENARIO_COMPLEX_PLATE_FACTORY_PATH = DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "sc_plate_2P_2S-3D-3C-3M-2A" / "sc_plate_2P_2S-3D-3C-3M-2A.json"
