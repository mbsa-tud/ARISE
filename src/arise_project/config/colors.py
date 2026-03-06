# -*- coding: utf-8 -*-

"""
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
