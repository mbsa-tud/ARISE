# -*- coding: utf-8 -*-

"""
Module defining the ExecutionMode enums for the task execution.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from enum import Enum, auto


class ExecutionMode(Enum):
    WORST_CASE = auto()
    BEST_CASE = auto()
    RANDOM = auto()
