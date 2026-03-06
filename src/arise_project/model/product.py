# -*- coding: utf-8 -*-

"""
Module defining classes of products

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from abc import ABC, abstractmethod

from src.arise_project.model.tasks import ProcessingTask
from src.arise_project.model.task_results import TaskResult
from src.arise_project.model.product_state import ProductState


class Product(ABC):
    """
    Abstract class defining an abstract product
    """

    # Class instantiation counter for unique id generation (not used in abstract class)
    _unique_id_ctr: int = 0
    _ABBREVIATION = "XX"

    def __init__(self,
                 processing_tasks: list[ProcessingTask],
                 starting_location_id: str,
                 target_location_id: str) -> None:

        if len(processing_tasks) < 1:
            raise ValueError(f"A product requires at least one processing task.")

        if (len(starting_location_id) < 1) or (len(target_location_id) < 1):
            raise ValueError(f"A product requires a starting and target location.")

        self._unique_id = ""

        self._current_state = ProductState(location_machine_id=starting_location_id, processing_tasks=None)
        self._target_state = ProductState(location_machine_id=target_location_id, processing_tasks=processing_tasks)

        self._state_history_list = [(self._current_state, '')]
        self._processing_history_list = []

    def __eq__(self, other) -> bool:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Comparing objects by unique identifier.
        :param other: Object to compare
        :return: True if unique identifier matches, False otherwise
        """
        return isinstance(other, Product) and self._unique_id == other.unique_id

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

    @property
    def current_state(self) -> ProductState:
        return self._current_state

    @property
    def target_state(self) -> ProductState:
        return self._target_state

    @property
    def state_history_list(self) -> list[tuple[ProductState, str]]:
        return self._state_history_list

    def update_state_by_task_result(self, task_result: TaskResult) -> None:

        # Update current product state & state history with new product state
        new_product_state = self._current_state.get_next_state(task_result.task)

        self._current_state = new_product_state
        self._state_history_list.append((new_product_state, task_result.skill.unique_id))
        self._processing_history_list.append(task_result)

    def undo_last_state_change(self) -> None:

        # Make sure there is a previous state to go back to
        if len(self._state_history_list) > 0 and len(self._processing_history_list) > 0:

            # Remove last entry from state & processing history
            self._state_history_list.pop()
            self._processing_history_list.pop()

            # After removing current state from history list, get previous state (last entry)
            last_product_state = self._state_history_list[-1][0]

            # Update current product state with last product state
            self._current_state = last_product_state

    def get_remaining_processing_tasks(self) -> set[ProcessingTask]:
        """
        The set of remaining processing tasks is the difference of the target state and the current state, as these are
        # both sets of completed processing tasks. The difference in the sets is therefore the set of processing tasks
        # that need to be completed to achieve the target state.
        :return: Set of remaining processing tasks
        """

        return self._target_state.processing_tasks.difference(self._current_state.processing_tasks)

    def print_processing_history(self) -> None:

        result_str = ""

        for idx, task_result in enumerate(self._processing_history_list):

            result_str += (f"{idx + 1}. ({task_result.task.unique_id} | {task_result.skill.unique_id} | "
                           f"T: {task_result.total_time:.2f} | E: {task_result.total_energy:.2f} | "
                           f"S: {task_result.success_bool})")

            if idx < len(self._processing_history_list) - 1:
                result_str += "\n"

        print(result_str)

    def is_done(self) -> bool:
        return self._current_state == self._target_state


class Plate(Product):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "PL"

    def __init__(self, width: int, height: int, processing_tasks: list[ProcessingTask], starting_location_id: str, target_location_id: str):
        super().__init__(processing_tasks=processing_tasks, starting_location_id=starting_location_id, target_location_id=target_location_id)

        self._width = width
        self._height = height

        self._unique_id = Plate._generate_unique_id()

    def get_params_dict(self) -> dict:
        return {"width": self.width,
                "height": self.height}

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value
