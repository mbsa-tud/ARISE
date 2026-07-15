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

Module defining the class of processing tasks

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import math
import random
from abc import ABC, abstractmethod

from src.arise_project.model.execution_mode import ExecutionMode
from src.arise_project.model.task_results import TaskResult
from src.arise_project.model.skills import Skill, DrillingSkill, CuttingSkill, MillingSkill, TransportSkill


class Task(ABC):
    """
    Abstract class defining an abstract general task
    """

    # Class instantiation counter for unique id generation (not used in abstract class)
    _unique_id_ctr: int = 0
    _ABBREVIATION = "XX"

    def __init__(self, unique_id: str) -> None:
        self._unique_id = unique_id

    def __eq__(self, other) -> bool:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Comparing objects by unique identifier.
        :param other: Object to compare
        :return: True if unique identifier matches, False otherwise
        """
        return isinstance(other, self.__class__) and self._unique_id == other.unique_id

    def __hash__(self) -> int:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Hashing the unique identifier to make sure hash is unique as well.
        :return: hash value of unique identifier (int)
        """
        return hash(self._unique_id)

    def __lt__(self, other):
        return self._unique_id < other.unique_id

    def __repr__(self) -> str:
        return f"{self._unique_id} ({self.__class__.__name__})"

    @classmethod
    def _generate_unique_id(cls) -> str:

        # Warning: Add semaphore for access control when implementing multiprocessing in the future
        # to avoid race conditions, otherwise unique id may not be unique
        cls._unique_id_ctr += 1

        unique_id = f"{cls._ABBREVIATION}{cls._unique_id_ctr:02d}"

        return unique_id

    @classmethod
    def reset_unique_id_ctr(cls) -> None:
        cls._unique_id_ctr = 0

    @classmethod
    def get_abbreviation(cls) -> str:
        return cls._ABBREVIATION

    @abstractmethod
    def get_params_dict(self) -> dict:
        pass

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @abstractmethod
    def execute(self, selected_skill: Skill, mode: ExecutionMode = ExecutionMode.RANDOM) -> TaskResult:
        """
        Use the selected skill to execute the task and return the time and energy cost as well as information
        regarding whether the task was a success or a failure.
        :param selected_skill: Skill to complete (Skill)
        :param mode: Execution mode (ExecutionMode)
        :return: task result (TaskResult)
        """
        pass

    @abstractmethod
    def get_description_short(self) -> str:
        pass

    @abstractmethod
    def get_description_long(self) -> str:
        pass


class ProcessingTask(Task):
    """
    Abstract class for processing tasks based on the abstract "Task" class
    """

    def __init__(self, unique_id: str, possible_skill_types: list[type[Skill]]) -> None:

        super().__init__(unique_id=unique_id)

        self._possible_skill_types = possible_skill_types
        self._precondition_completed_task_id_set = set()

    @property
    def possible_skill_types(self) -> list[type[Skill]]:
        return self._possible_skill_types

    @property
    def precondition_completed_task_id_set(self) -> set[str]:
        return self._precondition_completed_task_id_set

    def add_precondition_task_id(self, task_id: str) -> None:
        self._precondition_completed_task_id_set.add(task_id)

    @abstractmethod
    def get_params_dict(self) -> dict:
        return {}

    @abstractmethod
    def execute(self, selected_skill: Skill, mode: ExecutionMode = ExecutionMode.RANDOM) -> TaskResult:
        """
        Use the selected skill to execute the task and return the time and energy cost as well as information
        regarding whether the task was a success or a failure.
        :param selected_skill: Skill to complete (Skill)
        :param mode: Execution mode (ExecutionMode)
        :return: task result (TaskResult)
        """
        pass

    @abstractmethod
    def get_description_short(self) -> str:
        pass

    @abstractmethod
    def get_description_long(self) -> str:
        pass


class DrillingTask(ProcessingTask):
    """
    A processing task that makes use of a drilling skill
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "DT"

    def __init__(self, center_x: int, center_y: int, radius: int) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = DrillingTask._generate_unique_id()

        super().__init__(unique_id=unique_id, possible_skill_types=[DrillingSkill])

        self._center_x = center_x
        self._center_y = center_y
        self._radius = radius

    @property
    def center_x(self) -> int:
        return self._center_x

    @property
    def center_y(self) -> int:
        return self._center_y

    @property
    def radius(self) -> int:
        return self._radius

    def get_params_dict(self) -> dict:
        return {"center_x": self.center_x,
                "center_y": self.center_y,
                "radius": self.radius}

    def execute(self, selected_skill: Skill, mode: ExecutionMode = ExecutionMode.RANDOM) -> tuple[float, float, bool]:

        if type(selected_skill) not in self._possible_skill_types:
            raise ValueError(f"Skill {selected_skill.unique_id} can't be used for processing task {self._unique_id}.")

        # Calculate the time cost specifically for this task (geometry / speed = time)
        time_cost = self._radius / selected_skill.execution_speed

        # Introduce noise based on process variability defined individually for each skill (noise sim)
        match mode:

            case ExecutionMode.RANDOM:

                total_time_cost = selected_skill.process_variability.time_with_variability(base_time=time_cost)

            case ExecutionMode.BEST_CASE:

                total_time_cost = selected_skill.process_variability.time_best_case(base_time=time_cost)

            case ExecutionMode.WORST_CASE:

                total_time_cost = selected_skill.process_variability.time_worst_case(base_time=time_cost)

            case _:

                raise ValueError("Unknown execution mode.")

        # Energy is derived from power draw over the (already noisy) time actually spent, not its own factor
        total_energy_cost = selected_skill.nominal_power_draw * total_time_cost

        rounded_total_time_cost = round(total_time_cost, 3)
        rounded_total_energy_cost = round(total_energy_cost, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return rounded_total_time_cost, rounded_total_energy_cost, success_bool

    def get_description_short(self) -> str:
        return f"R: {self.radius:.2f}"

    def get_description_long(self) -> str:
        return f"X: {self._center_x:.2f}, Y: {self._center_y:.2f}, R: {self.radius:.2f}"


class MillingTask(ProcessingTask):
    """
    A processing task that makes use of a milling skill
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "MT"

    def __init__(self, center_x: int, center_y: int, radius: int) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = MillingTask._generate_unique_id()

        super().__init__(unique_id=unique_id, possible_skill_types=[MillingSkill])

        self._center_x = center_x
        self._center_y = center_y
        self._radius = radius

    @property
    def center_x(self) -> int:
        return self._center_x

    @property
    def center_y(self) -> int:
        return self._center_y

    @property
    def radius(self) -> int:
        return self._radius

    def get_params_dict(self) -> dict:
        return {"center_x": self.center_x,
                "center_y": self.center_y,
                "radius": self.radius}

    def execute(self, selected_skill: Skill, mode: ExecutionMode = ExecutionMode.RANDOM) -> tuple[float, float, bool]:

        if type(selected_skill) not in self._possible_skill_types:
            raise ValueError(f"Skill {selected_skill.unique_id} can't be used for processing task {self._unique_id}.")

        # Calculate the time cost specifically for this task (geometry / speed = time)
        time_cost = self._radius / selected_skill.execution_speed

        # Introduce noise based on process variability defined individually for each skill (noise sim)
        match mode:

            case ExecutionMode.RANDOM:

                total_time_cost = selected_skill.process_variability.time_with_variability(base_time=time_cost)

            case ExecutionMode.BEST_CASE:

                total_time_cost = selected_skill.process_variability.time_best_case(base_time=time_cost)

            case ExecutionMode.WORST_CASE:

                total_time_cost = selected_skill.process_variability.time_worst_case(base_time=time_cost)

            case _:

                raise ValueError("Unknown execution mode.")

        # Energy is derived from power draw over the (already noisy) time actually spent, not its own factor
        total_energy_cost = selected_skill.nominal_power_draw * total_time_cost

        rounded_total_time_cost = round(total_time_cost, 3)
        rounded_total_energy_cost = round(total_energy_cost, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return rounded_total_time_cost, rounded_total_energy_cost, success_bool

    def get_description_short(self) -> str:
        return f"A: {self._radius:.2f}"

    def get_description_long(self) -> str:
        return f"A: {self._radius:.2f}"


class CuttingTask(ProcessingTask):
    """
    A processing task that makes use of a cutting skill
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "CT"

    def __init__(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = CuttingTask._generate_unique_id()

        super().__init__(unique_id=unique_id, possible_skill_types=[CuttingSkill])

        self._start_x = start_x
        self._start_y = start_y
        self._end_x = end_x
        self._end_y = end_y

        # Pre-calculation for faster reference
        self._total_length = math.hypot(self._start_x - self._end_x, self._start_y - self._end_y)

    @property
    def start_x(self) -> int:
        return self._start_x

    @property
    def start_y(self) -> int:
        return self._start_y

    @property
    def end_x(self) -> int:
        return self._end_x

    @property
    def end_y(self) -> int:
        return self._end_y

    @property
    def total_length(self) -> float:
        return self._total_length

    def get_params_dict(self) -> dict:
        return {"total length": self._total_length}

    def execute(self, selected_skill: Skill, mode: ExecutionMode = ExecutionMode.RANDOM) -> tuple[float, float, bool]:

        if type(selected_skill) not in self._possible_skill_types:
            raise ValueError(f"Skill {selected_skill.unique_id} can't be used for processing task {self._unique_id}.")

        # Calculate the time cost specifically for this task (geometry / speed = time)
        time_cost = self._total_length / selected_skill.execution_speed

        # Introduce noise based on process variability defined individually for each skill (noise sim)
        match mode:

            case ExecutionMode.RANDOM:

                total_time_cost = selected_skill.process_variability.time_with_variability(base_time=time_cost)

            case ExecutionMode.BEST_CASE:

                total_time_cost = selected_skill.process_variability.time_best_case(base_time=time_cost)

            case ExecutionMode.WORST_CASE:

                total_time_cost = selected_skill.process_variability.time_worst_case(base_time=time_cost)

            case _:

                raise ValueError("Unknown execution mode.")

        # Energy is derived from power draw over the (already noisy) time actually spent, not its own factor
        total_energy_cost = selected_skill.nominal_power_draw * total_time_cost

        rounded_total_time_cost = round(total_time_cost, 3)
        rounded_total_energy_cost = round(total_energy_cost, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return rounded_total_time_cost, rounded_total_energy_cost, success_bool

    def get_description_short(self) -> str:
        return f"L: {self._total_length:.2f}"

    def get_description_long(self) -> str:
        return f"L: {self._total_length:.2f}"

class TransportTask(Task):
    """
    Abstract class for transport tasks
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "TT"

    def __init__(self, source_machine_id: str, target_machine_id: str, distance: float) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = TransportTask._generate_unique_id()

        super().__init__(unique_id=unique_id)

        self._source_machine_id = source_machine_id
        self._target_machine_id = target_machine_id
        self._distance = distance

    @property
    def source_machine_id(self) -> str:
        return self._source_machine_id

    @property
    def target_machine_id(self) -> str:
        return self._target_machine_id

    @property
    def distance(self) -> float:
        return self._distance

    def get_params_dict(self) -> dict:
        return {"distance": self._distance}

    def execute(self, selected_skill: Skill, mode: ExecutionMode = ExecutionMode.RANDOM) -> tuple[float, float, bool]:

        if not isinstance(selected_skill, TransportSkill):
            raise ValueError(f"Skill can't be used for transport task {self._unique_id}.")

        # Calculate the time cost specifically for this task (geometry / speed = time)
        time_cost = self._distance / selected_skill.execution_speed

        # Introduce noise based on process variability defined individually for each skill (noise sim)
        match mode:

            case ExecutionMode.RANDOM:

                total_time_cost = selected_skill.process_variability.time_with_variability(base_time=time_cost)

            case ExecutionMode.BEST_CASE:

                total_time_cost = selected_skill.process_variability.time_best_case(base_time=time_cost)

            case ExecutionMode.WORST_CASE:

                total_time_cost = selected_skill.process_variability.time_worst_case(base_time=time_cost)

            case _:

                raise ValueError("Unknown execution mode.")

        # Energy is derived from power draw over the (already noisy) time actually spent, not its own factor
        total_energy_cost = selected_skill.nominal_power_draw * total_time_cost

        rounded_total_time_cost = round(total_time_cost, 3)
        rounded_total_energy_cost = round(total_energy_cost, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return rounded_total_time_cost, rounded_total_energy_cost, success_bool

    def get_description_short(self) -> str:
        return f"-> {self._target_machine_id}"

    def get_description_long(self) -> str:
        return f"{self._source_machine_id} -> {self._target_machine_id} ({self._distance})"
