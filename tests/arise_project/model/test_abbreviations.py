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
"""

import pytest

from src.arise_project.model.machines import DrillingMachine, CuttingMachine, MillingMachine, ConveyorBelt, \
    AutomatedGuidedVehicle, ThreeAxisRobot, StorageMachine
from src.arise_project.model.tasks import DrillingTask, MillingTask, CuttingTask
from src.arise_project.model.product import Plate
from src.arise_project.model.skills import DrillingSkill, MillingSkill, CuttingSkill, TransportSkill, StoreSkill, \
    RetrieveSkill


def test_abbreviations():
    """
    Test: Makes sure that each abbreviation is unique.
    :return: None
    """

    class_list = [DrillingMachine, MillingMachine, CuttingMachine,
                  ConveyorBelt, AutomatedGuidedVehicle, ThreeAxisRobot, StorageMachine,
                  DrillingSkill, MillingSkill, CuttingSkill, TransportSkill, StoreSkill, RetrieveSkill,
                  DrillingTask, MillingTask, CuttingTask,
                  Plate]

    abbreviations_set = set()

    for class_type in class_list:

        assert hasattr(class_type, "_ABBREVIATION")
        assert class_type._ABBREVIATION not in abbreviations_set
        abbreviations_set.add(class_type._ABBREVIATION)

    print(f"\n{list(abbreviations_set)}")
