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

Module containing the optimization result class.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.arise_project.config.paths import FILE_NAME_OPT_RESULT_A_STAR_PKL, FILE_NAME_OPT_RESULT_DIJKSTRA_PKL, \
    FILE_NAME_OPT_RESULT_DFS_PKL, FILE_NAME_OPT_RESULT_IDDFS_PKL, FILE_NAME_OPT_RESULT_NSGA2_PKL, \
    FILE_NAME_OPT_RESULT_NSGA3_PKL, FILE_NAME_OPT_RESULT_RL_DQN_PKL, FILE_NAME_OPT_RESULT_LLM_AGENT_PKL, \
    FILE_NAME_OPT_RESULT_HUMAN_PKL, FILE_NAME_OPT_RESULT_TASKS_A_STAR_CSV, FILE_NAME_OPT_RESULT_TASKS_DIJKSTRA_CSV, \
    FILE_NAME_OPT_RESULT_TASKS_DFS_CSV, FILE_NAME_OPT_RESULT_TASKS_IDDFS_CSV, FILE_NAME_OPT_RESULT_TASKS_NSGA2_CSV, \
    FILE_NAME_OPT_RESULT_TASKS_NSGA3_CSV, FILE_NAME_OPT_RESULT_TASKS_RL_DQN_CSV, \
    FILE_NAME_OPT_RESULT_TASKS_LLM_AGENT_CSV, FILE_NAME_OPT_RESULT_TASKS_HUMAN_CSV

from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.task_results import TaskResult
from src.arise_project.tools.output_timestamp import print_with_timestamp
from src.arise_project.tools.energy_format import joules_to_wh

OPTIMIZATION_RESULT_FILE_NAME_PKL = {OptimizationMethod.OPT_A_STAR: FILE_NAME_OPT_RESULT_A_STAR_PKL,
                                     OptimizationMethod.OPT_DIJKSTRA: FILE_NAME_OPT_RESULT_DIJKSTRA_PKL,
                                     OptimizationMethod.OPT_DFS: FILE_NAME_OPT_RESULT_DFS_PKL,
                                     OptimizationMethod.OPT_IDDFS: FILE_NAME_OPT_RESULT_IDDFS_PKL,
                                     OptimizationMethod.OPT_NSGA2: FILE_NAME_OPT_RESULT_NSGA2_PKL,
                                     OptimizationMethod.OPT_NSGA3: FILE_NAME_OPT_RESULT_NSGA3_PKL,
                                     OptimizationMethod.OPT_RL_DQN: FILE_NAME_OPT_RESULT_RL_DQN_PKL,
                                     OptimizationMethod.OPT_LLM_AGENT: FILE_NAME_OPT_RESULT_LLM_AGENT_PKL,
                                     OptimizationMethod.OPT_HUMAN: FILE_NAME_OPT_RESULT_HUMAN_PKL}

OPTIMIZATION_RESULT_TASKS_FILE_NAME_CSV = {OptimizationMethod.OPT_A_STAR: FILE_NAME_OPT_RESULT_TASKS_A_STAR_CSV,
                                           OptimizationMethod.OPT_DIJKSTRA: FILE_NAME_OPT_RESULT_TASKS_DIJKSTRA_CSV,
                                           OptimizationMethod.OPT_DFS: FILE_NAME_OPT_RESULT_TASKS_DFS_CSV,
                                           OptimizationMethod.OPT_IDDFS: FILE_NAME_OPT_RESULT_TASKS_IDDFS_CSV,
                                           OptimizationMethod.OPT_NSGA2: FILE_NAME_OPT_RESULT_TASKS_NSGA2_CSV,
                                           OptimizationMethod.OPT_NSGA3: FILE_NAME_OPT_RESULT_TASKS_NSGA3_CSV,
                                           OptimizationMethod.OPT_RL_DQN: FILE_NAME_OPT_RESULT_TASKS_RL_DQN_CSV,
                                           OptimizationMethod.OPT_LLM_AGENT: FILE_NAME_OPT_RESULT_TASKS_LLM_AGENT_CSV,
                                           OptimizationMethod.OPT_HUMAN: FILE_NAME_OPT_RESULT_TASKS_HUMAN_CSV}

COL_OPT_RES_PRODUCT = "Product"
COL_OPT_RES_TASK = "Task"
COL_OPT_RES_MACHINE = "Machine"
COL_OPT_RES_SKILL = "Skill"
COL_OPT_RES_SKILL_TYPE = "SkillType"
COL_OPT_RES_TIME = "Time (s)"
COL_OPT_RES_ENERGY = "Energy (Wh)"
COL_OPT_RES_RELIABILITY = "Reliability"
COL_OPT_RES_ACTION_IDX = "Action"
COL_OPT_RES_NOTE = "Note"


class OptimizationResult:

    def __init__(self, action_idx_sequence: list[int], task_result_list: list[TaskResult],
                 total_time: float, total_energy: float, sequence_reliability: float,
                 objective_function: ObjectiveFunction, other_params_dict: dict[str, Any],
                 total_duration_seconds: float, opt_method: OptimizationMethod):

        self._action_idx_sequence = action_idx_sequence
        self._task_result_list = task_result_list

        self._steps = len(self._action_idx_sequence)
        self._total_time = total_time
        self._total_energy = total_energy
        self._sequence_reliability = sequence_reliability
        self._objective_function = objective_function

        self._total_cost = objective_function(time_cost=self._total_time,
                                              energy_cost=self._total_energy,
                                              reliability=self._sequence_reliability)

        self._other_params_dict = other_params_dict
        self._total_duration_seconds = total_duration_seconds

        self._opt_method = opt_method
        self._timestamp_dt = datetime.now()

    @staticmethod
    def pickle_load(scenario_dir: Path, opt_method: OptimizationMethod) -> OptimizationResult | None:

        opt_result = None

        pickle_file = scenario_dir / OPTIMIZATION_RESULT_FILE_NAME_PKL[opt_method]

        if pickle_file.exists():

            with open(pickle_file, "rb") as f:
                opt_result = pickle.load(f)

        else:

            print_with_timestamp(f"Loading '{opt_method.value}' result failed, file: '{pickle_file.name}' does not exist")

        if not isinstance(opt_method, OptimizationMethod):

            print_with_timestamp(f"Loading '{opt_method.value}' result  failed, pickle file does not contain an 'OptimizationMethod' object")

        print_with_timestamp(f"Loaded '{opt_method.value}' result '{pickle_file.name}' successfully")

        # TODO handle cases by raising an error instead of returning None
        return opt_result

    @property
    def action_idx_sequence(self) -> list[int]:
        return self._action_idx_sequence

    @property
    def task_result_list(self) -> list[TaskResult]:
        return self._task_result_list

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def total_energy(self) -> float:
        return self._total_energy

    @property
    def sequence_reliability(self) -> float:
        return self._sequence_reliability

    @property
    def objective_function(self) -> ObjectiveFunction:
        return self._objective_function

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def other_params_dict(self) -> dict[str, Any]:
        return self._other_params_dict

    @property
    def total_duration_seconds(self) -> float:
        return self._total_duration_seconds

    # TODO remove this temporary workaround as this attribute should not be able to be set
    @total_duration_seconds.setter
    def total_duration_seconds(self, value: int) -> None:
        self._total_duration_seconds = value

    @property
    def opt_method(self) -> OptimizationMethod:
        return self._opt_method

    @property
    def timestamp_dt(self) -> datetime:
        return self._timestamp_dt

    def get_timestamp_str(self) -> str:
        return self._timestamp_dt.strftime('%d.%m.%Y %H:%M:%S')

    def print_task_result_history(self, show_numerical_index: bool = False, show_action_index: bool = False) -> None:

        for idx, action_idx in enumerate(self._action_idx_sequence):

            line_str = ""

            if show_numerical_index:
                 line_str += f"{idx+1}. "

            line_str += f"{self._task_result_list[idx]} "

            if show_action_index:
                line_str += f"[{action_idx}] "

            print(line_str)

    def to_dataframe(self) -> pd.DataFrame:

        action_data_dict_list = [{COL_OPT_RES_PRODUCT: task_result.product.unique_id,
                                  COL_OPT_RES_TASK: task_result.task.unique_id,
                                  COL_OPT_RES_MACHINE: task_result.machine.unique_id,
                                  COL_OPT_RES_SKILL: f"{task_result.skill.unique_id}",
                                  COL_OPT_RES_SKILL_TYPE: task_result.skill.type_name(),
                                  COL_OPT_RES_TIME: round(task_result.total_time, 3),
                                  COL_OPT_RES_ENERGY: round(joules_to_wh(task_result.total_energy), 3),
                                  COL_OPT_RES_RELIABILITY: task_result.skill.reliability,
                                  COL_OPT_RES_ACTION_IDX: self._action_idx_sequence[idx],
                                  COL_OPT_RES_NOTE: task_result.task.get_description_short()}

                                 for idx, task_result in enumerate(self._task_result_list)]

        return pd.DataFrame(data=action_data_dict_list)

    def to_csv(self, output_directory: Path) -> None:

        output_directory.mkdir(parents=True, exist_ok=True)
        file_output_path = output_directory / OPTIMIZATION_RESULT_TASKS_FILE_NAME_CSV[self._opt_method]

        output_df = self.to_dataframe()
        output_df.to_csv(file_output_path, sep=";")

    def pickle_dump(self, output_directory: Path) -> None:

        output_directory.mkdir(parents=True, exist_ok=True)
        file_output_path = output_directory / OPTIMIZATION_RESULT_FILE_NAME_PKL[self._opt_method]

        with open(file_output_path, "wb") as f:
            pickle.dump(self, f)
