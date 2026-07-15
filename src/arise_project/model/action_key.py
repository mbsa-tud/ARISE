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
