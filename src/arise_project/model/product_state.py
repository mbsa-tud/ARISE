# -*- coding: utf-8 -*-

"""
Module defining the product state class

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from typing import List, Set, Self

from src.arise_project.model.tasks import ProcessingTask, Task, TransportTask


class ProductState:
    """
    A product state consists of a set of completed processing tasks as well as the location (specified by machine id)
    """

    def __init__(self, location_machine_id: str,
                 processing_tasks: list[ProcessingTask] | set[ProcessingTask] | ProcessingTask | None = None) -> None:

        self._location_machine_id = location_machine_id

        # Initialize product state using many, one, or without any processing tasks
        if processing_tasks is None:
            self._processing_tasks = set()
        elif isinstance(processing_tasks, ProcessingTask):
            self._processing_tasks = {processing_tasks}
        elif isinstance(processing_tasks, List):
            self._processing_tasks = set(processing_tasks)
        elif isinstance(processing_tasks, Set):
            self._processing_tasks = processing_tasks
        else:
            raise ValueError("Invalid type for processing task parameter")

    def __eq__(self, other) -> bool:
        """
        Allows direct comparison between two states, f.e. to check if the current state matches the target state.
        :param other: Object to compare
        :return: True if same class, same location and same completed processing tasks, False otherwise
        """

        same_class_bool = isinstance(other, ProductState)
        same_location = self.location_machine_id == other.location_machine_id
        same_processing_tasks = len(self._processing_tasks.symmetric_difference(other.processing_tasks)) == 0

        return same_class_bool and same_location and same_processing_tasks

    def __hash__(self) -> int:
        """
        Create a hash from the state's string representation, which consists of unique ids of processing tasks sorted
        in alphabetical order as well as the location (unique machine id). This makes identifying by hash value possible.
        :return: hash value of string representation (int)
        """

        return hash(str(self))

    @property
    def location_machine_id(self) -> str:
        return self._location_machine_id

    @property
    def processing_tasks(self) -> set[ProcessingTask]:
        """
        The set of completed processing tasks that make up the product state.
        :return: Completed processing tasks (set of ProcessingTask)
        """
        return self._processing_tasks

    def get_ordered_processing_task_list(self) -> list[ProcessingTask]:

        result_list = list(self._processing_tasks)
        result_list.sort()

        return result_list

    def get_next_state(self, task: Task) -> Self:
        """
        Creates a product state based on the current state expanded upon by the newly completed task.
        :param task: Completed task (Task)
        :return: Product state (ProductState)
        """

        # Copy current set of processing tasks and add the newly completed task
        completed_tasks = self._processing_tasks.copy()

        if isinstance(task, ProcessingTask):
            completed_tasks.add(task)

        new_location_machine_id = self._location_machine_id

        if isinstance(task, TransportTask):
            new_location_machine_id = task.target_machine_id

        return ProductState(location_machine_id=new_location_machine_id, processing_tasks=completed_tasks)

    def get_task_by_id(self, unique_id: str) -> ProcessingTask:

        result_task = None

        for task in list(self._processing_tasks):
            if task.unique_id == unique_id:
                result_task = task
                break

        if result_task is None:
            raise ValueError(f"Product state does not contain processing task {unique_id}.")

        return result_task

    def contains_task_with_id(self, unique_id: str) -> bool:

        for task in self._processing_tasks:
            if task.unique_id == unique_id:
                return True

        return False

    def __repr__(self):

        # Special case for the initial (empty) product state
        if len(self._processing_tasks) == 0:
            return f"Initial [{self._location_machine_id}]"

        # Convert the set to a list and sort by the unique identifier
        sorted_tasks = sorted(list(self._processing_tasks), key=lambda task: task.unique_id)

        result_str = ""

        for idx, proc_task in enumerate(sorted_tasks):

            result_str = result_str + f"{proc_task.unique_id}"

            if idx != len(sorted_tasks) - 1:
                result_str = result_str + ", "

        result_str += f" [{self._location_machine_id}]"

        return result_str
