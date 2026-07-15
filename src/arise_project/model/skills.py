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

Module defining the various skill classes

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from abc import ABC

from src.arise_project.model.variability import ProcessVariability


class Skill(ABC):
    """
    Abstract class defining an abstract skill.

    execution_speed: rate at which the skill advances through its task's geometric quantity, in
    units-of-that-quantity per second (e.g. mm/s for a processing task's radius/length, m/s for a
    transport task's distance). Task time is computed as geometry / execution_speed.
    nominal_power_draw: average electrical power drawn while actively executing the task, in Watts.
    Energy is derived as nominal_power_draw * time, not tracked as an independent quantity.
    """

    # Class instantiation counter for unique id generation (not used in abstract class)
    _unique_id_ctr: int = 0
    _ABBREVIATION = "XX"

    def __init__(self, unique_id: str, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        self._unique_id = unique_id

        self._execution_speed = execution_speed
        self._nominal_power_draw = nominal_power_draw
        self._reliability = reliability

        if process_variability is not None:
            self._process_variability = process_variability
        else:
            self._process_variability = ProcessVariability()

    def __eq__(self, other) -> bool:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Comparing objects by unique identifier.
        :param other: Object to compare
        :return: True if unique identifier matches, False otherwise
        """
        return isinstance(other, Skill) and self._unique_id == other.unique_id

    def __hash__(self) -> int:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Hashing the unique identifier to make sure hash is unique as well.
        :return: hash value of unique identifier (int)
        """
        return hash(self._unique_id)

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

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def execution_speed(self) -> float:
        return self._execution_speed

    @execution_speed.setter
    def execution_speed(self, value: float) -> None:

        if value <= 0:
            raise ValueError("Execution speed must be positive.")

        self._execution_speed = value

    @property
    def nominal_power_draw(self) -> float:
        return self._nominal_power_draw

    @nominal_power_draw.setter
    def nominal_power_draw(self, value: float) -> None:

        if value <= 0:
            raise ValueError("Nominal power draw must be positive.")

        self._nominal_power_draw = value

    @property
    def reliability(self) -> float:
        return self._reliability

    @reliability.setter
    def reliability(self, value: float) -> None:

        if value < 0 or value > 1:
            raise ValueError("Reliability must be between 0.0 and 1.0")

        self._reliability = value

    @property
    def process_variability(self) -> ProcessVariability:
        return self._process_variability

    def type_name(self) -> str:
        return self.__class__.__name__


class ProcessingSkill(Skill):

    def __init__(self, unique_id: str, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)


class DrillingSkill(ProcessingSkill):
    """
    Drills a circular hole into a Plate at (x, y) with a given diameter.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "DS"

    def __init__(self, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = DrillingSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)


class MillingSkill(ProcessingSkill):
    """
    Mills a linear path between (x1, y1) and (x2, y2) with a given diameter.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "MS"

    def __init__(self, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = MillingSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)


class CuttingSkill(ProcessingSkill):
    """
    Cuts a straight line between two points by zeroing the structure.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "CS"

    def __init__(self, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = CuttingSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)


class TransportSkill(Skill):
    """
    Transfers a product from a source machine to a target machine.
    Measures time and energy based on Euclidean distance between machine coordinates.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "TS"

    def __init__(self, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = TransportSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)


class StoreSkill(Skill):
    """
    Stores a product into the internal storage of a Storage machine.
    Clears the occupied slot. Returns fixed time and energy with random noise.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "SS"

    def __init__(self, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = StoreSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)


class RetrieveSkill(Skill):
    """
    Retrieves the oldest product from the storage queue of a Storage machine.
    Puts it into the occupied slot and returns fixed time and energy with random noise.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "RS"

    def __init__(self, execution_speed: float = 1.0, nominal_power_draw: float = 1.0,
                 reliability: float = 1.0, process_variability: ProcessVariability = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = RetrieveSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, execution_speed=execution_speed, nominal_power_draw=nominal_power_draw,
                         reliability=reliability, process_variability=process_variability)
