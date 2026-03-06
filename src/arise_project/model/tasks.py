# -*- coding: utf-8 -*-

"""
Module defining the class of processing tasks

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

import random
from abc import ABC, abstractmethod
from typing import Type

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
    def execute(self, selected_skill: Skill) -> TaskResult:
        """
        Use the selected skill to execute the task and return the time and energy cost as well as information
        regarding whether the task was a success or a failure.
        :param selected_skill: Skill to complete (Skill)
        :return: task result (TaskResult)
        """
        pass


class ProcessingTask(Task):
    """
    Abstract class for processing tasks
    """

    def __init__(self, unique_id: str, possible_skill_types: list[Type[Skill]]) -> None:

        super().__init__(unique_id=unique_id)

        self._possible_skill_types = possible_skill_types

    @property
    def possible_skill_types(self) -> list[Type[Skill]]:
        return self._possible_skill_types

    def get_params_dict(self) -> dict:
        return {}

    def execute(self, selected_skill: Skill) -> TaskResult:
        """
        Use the selected skill to execute the task and return the time and energy cost as well as information
        regarding whether the task was a success or a failure.
        :param selected_skill: Skill to complete (Skill)
        :return: task result (TaskResult)
        """
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

    def execute(self, selected_skill: Skill) -> tuple[float, float, bool]:

        if type(selected_skill) not in self._possible_skill_types:
            raise ValueError(f"Skill {selected_skill.unique_id} can't be used for processing task {self._unique_id}.")

        # TODO Calculate total time and energy using the processing task in a way that makes sense
        time = selected_skill.time_factor * self._radius
        energy = selected_skill.energy_factor * self._radius

        noise = random.uniform(0.95, 1.1)
        total_time = round(time * noise, 3)
        total_energy = round(energy * noise, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return total_time, total_energy, success_bool


class MillingTask(ProcessingTask):
    """
    A processing task that makes use of a milling skill
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "MT"

    def __init__(self, total_area: float) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = MillingTask._generate_unique_id()

        super().__init__(unique_id=unique_id, possible_skill_types=[MillingSkill])

        self._total_area = total_area

    @property
    def total_area(self) -> float:
        return self._total_area

    def get_params_dict(self) -> dict:
        return {"total_area": self._total_area}

    def execute(self, selected_skill: Skill) -> tuple[float, float, bool]:

        if type(selected_skill) not in self._possible_skill_types:
            raise ValueError(f"Skill {selected_skill.unique_id} can't be used for processing task {self._unique_id}.")

        # TODO Calculate total time and energy using the processing task in a way that makes sense
        time = selected_skill.time_factor * self._total_area
        energy = selected_skill.energy_factor * self._total_area

        # Introduce noise
        noise = random.uniform(0.95, 1.1)
        total_time = round(time * noise, 3)
        total_energy = round(energy * noise, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return total_time, total_energy, success_bool


class CuttingTask(ProcessingTask):
    """
    A processing task that makes use of a cutting skill
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "CT"

    def __init__(self, total_length: float) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = CuttingTask._generate_unique_id()

        super().__init__(unique_id=unique_id, possible_skill_types=[CuttingSkill])

        self._total_length = total_length

    @property
    def total_length(self) -> float:
        return self._total_length

    def get_params_dict(self) -> dict:
        return {"total length": self._total_length}

    def execute(self, selected_skill: Skill) -> tuple[float, float, bool]:

        if type(selected_skill) not in self._possible_skill_types:
            raise ValueError(f"Skill {selected_skill.unique_id} can't be used for processing task {self._unique_id}.")

        # TODO Calculate total time and energy using the processing task in a way that makes sense
        time = selected_skill.time_factor * self._total_length
        energy = selected_skill.energy_factor * self._total_length

        # Introduce noise
        noise = random.uniform(0.95, 1.1)
        total_time = round(time * noise, 3)
        total_energy = round(energy * noise, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return total_time, total_energy, success_bool


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

    def execute(self, selected_skill: Skill) -> tuple[float, float, bool]:

        if not isinstance(selected_skill, TransportSkill):
            raise ValueError(f"Skill can't be used for transport task {self._unique_id}.")

        # TODO Calculate total time and energy using the transport task in a way that makes sense
        time = selected_skill.time_factor * self._distance
        energy = selected_skill.energy_factor * self._distance

        # Introduce noise
        noise = random.uniform(0.95, 1.1)
        total_time = round(time * noise, 3)
        total_energy = round(energy * noise, 3)

        # TODO Decide if this needs to be deactivated when reliability is considered during optimization
        # Decide randomly if execution of processing task is successful based on reliability
        success_bool = random.random() < selected_skill.reliability

        return total_time, total_energy, success_bool
