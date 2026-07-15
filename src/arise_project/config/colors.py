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

Module containing color constants for consistent visualization

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from src.arise_project.model.machines import DrillingMachine, CuttingMachine, MillingMachine, TransporterMachine, \
    StorageMachine

from src.arise_project.model.skills import DrillingSkill, CuttingSkill, MillingSkill, TransportSkill, StoreSkill

COLOR_DRILLING = "darkorange"
COLOR_CUTTING = "forestgreen"
COLOR_MILLING = "dodgerblue"
COLOR_STORAGE = "dimgrey"
COLOR_TRANSPORT = "purple"


COLOR_BY_SKILL_DICT = {DrillingSkill.__name__: COLOR_DRILLING,
                       CuttingSkill.__name__: COLOR_CUTTING,
                       MillingSkill.__name__: COLOR_MILLING,
                       StoreSkill.__name__: COLOR_STORAGE,
                       TransportSkill.__name__: COLOR_TRANSPORT}


COLOR_BY_STATIONARY_MACHINE_DICT = {DrillingMachine.__name__: COLOR_DRILLING,
                                    CuttingMachine.__name__: COLOR_CUTTING,
                                    MillingMachine.__name__: COLOR_MILLING,
                                    StorageMachine.__name__: COLOR_STORAGE,
                                    TransporterMachine.__name__: COLOR_TRANSPORT}
