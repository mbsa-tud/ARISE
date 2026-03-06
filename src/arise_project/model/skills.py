# -*- coding: utf-8 -*-

"""
Module defining the various skill classes

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from abc import ABC, abstractmethod


class Skill(ABC):
    """
    Abstract class defining an abstract skill
    """

    # Class instantiation counter for unique id generation (not used in abstract class)
    _unique_id_ctr: int = 0
    _ABBREVIATION = "XX"

    def __init__(self, unique_id: str, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        self._unique_id = unique_id

        self._time_factor = time_factor
        self._energy_factor = energy_factor
        self._reliability = reliability

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
    def time_factor(self) -> float:
        return self._time_factor

    @time_factor.setter
    def time_factor(self, value: float) -> None:

        if value <= 0:
            raise ValueError("Time factor must be positive.")

        self._time_factor = value

    @property
    def energy_factor(self) -> float:
        return self._energy_factor

    @energy_factor.setter
    def energy_factor(self, value: float) -> None:

        if value <= 0:
            raise ValueError("Energy factor must be positive.")

        self._energy_factor = value

    @property
    def reliability(self) -> float:
        return self._reliability

    @reliability.setter
    def reliability(self, value: float) -> None:

        if value < 0 or value > 1:
            raise ValueError("Reliability must be between 0.0 and 1.0")

        self._reliability = value

    def type_name(self) -> str:
        return self.__class__.__name__


class ProcessingSkill(Skill):

    def __init__(self, unique_id: str, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:
        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)


class DrillingSkill(ProcessingSkill):
    """
    Drills a circular hole into a Plate at (x, y) with a given diameter.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "DS"

    def __init__(self, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = DrillingSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)


class MillingSkill(ProcessingSkill):
    """
    Mills a linear path between (x1, y1) and (x2, y2) with a given diameter.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "MS"

    def __init__(self, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = MillingSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)


class CuttingSkill(ProcessingSkill):
    """
    Cuts a straight line between two points by zeroing the structure.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "CS"

    def __init__(self, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = CuttingSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)


class TransportSkill(Skill):
    """
    Transfers a product from a source machine to a target machine.
    Measures time and energy based on Euclidean distance between machine coordinates.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "TS"

    def __init__(self, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = TransportSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)


class StoreSkill(Skill):
    """
    Stores a product into the internal storage of a Storage machine.
    Clears the occupied slot. Returns fixed time and energy with random noise.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "SS"

    def __init__(self, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = StoreSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)


class RetrieveSkill(Skill):
    """
    Retrieves the oldest product from the storage queue of a Storage machine.
    Puts it into the occupied slot and returns fixed time and energy with random noise.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "RS"

    def __init__(self, time_factor: float = 1.0, energy_factor: float = 1.0, reliability: float = 1.0) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = RetrieveSkill._generate_unique_id()

        super().__init__(unique_id=unique_id, time_factor=time_factor, energy_factor=energy_factor, reliability=reliability)
