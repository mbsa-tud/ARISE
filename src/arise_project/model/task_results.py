# -*- coding: utf-8 -*-

"""
Module defining the class of task results

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from src.arise_project.model.skills import Skill


class TaskResult:
    """
    Container / data class containing the results of executing a task with a specific skill.
    """

    def __init__(self, product: "Product", machine: "Machine", task: "Task",
                 skill: Skill, total_time: float, total_energy: float,
                 success_bool: bool):

        self._product = product
        self._machine = machine
        self._task = task
        self._skill = skill
        self._total_time = total_time
        self._total_energy = total_energy
        self._success_bool = success_bool

    def __repr__(self) -> str:
        return (f"{self._product.unique_id} | {self._task.unique_id} | {self._skill.unique_id} "
                f"({self._total_time:.2f}/{self._total_energy:.2f}/{str(self._success_bool)[:1]})")

    @property
    def product(self) -> "Product":
        return self._product

    @property
    def machine(self) -> "Machine":
        return self._machine

    @property
    def task(self) -> "Task":
        return self._task

    @property
    def skill(self) -> Skill:
        return self._skill

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def total_energy(self) -> float:
        return self._total_energy

    @property
    def success_bool(self) -> bool:
        return self._success_bool

    def get_short_name(self, with_product: bool = False, with_machine: bool = False) -> str:

        return_str = ""

        if with_product:
            return_str += f"{self._product.unique_id} | "

        return_str += f"{self._task.unique_id} | "

        if with_machine:
            return_str += f"{self._machine.unique_id} | "

        return_str += f"{self._skill.unique_id}"

        return return_str
