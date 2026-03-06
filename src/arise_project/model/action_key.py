# -*- coding: utf-8 -*-

"""
Module containing the action key class definition for reinforcement learning.

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from src.arise_project.model.product import Product
from src.arise_project.model.tasks import Task
from src.arise_project.model.skills import Skill


class ActionKey:

    def __init__(self, product: Product, task: Task, skill: Skill) -> None:
        self._product_id = product.unique_id
        self._task_id = task.unique_id
        self._skill_id = skill.unique_id

    def __repr__(self) -> str:
        return f"({self._product_id}, {self._task_id}, {self._skill_id})"

    def __hash__(self):
        return hash((self.product_id, self.task_id, self.skill_id))

    def __eq__(self, other):
        return (self.product_id, self.task_id, self.skill_id) == (other.product_id, other.task_id, other.skill_id)

    def __lt__(self, other):
        return (self.product_id, self.task_id, self.skill_id) < (other.product_id, other.task_id, other.skill_id)

    @property
    def product_id(self) -> str:
        return self._product_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def skill_id(self) -> str:
        return self._skill_id
