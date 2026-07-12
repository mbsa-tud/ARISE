# -*- coding: utf-8 -*-

"""
Module defining energy unit conversion constants shared across the model and GUI. Energy is always
computed internally in Joules (nominal_power_draw is in Watts, time in seconds) - these constants
only exist to convert to human-readable units at the point of display/reporting.

Author: Patrick Fischer
Version: 0.0.3
"""

JOULES_PER_WH = 3_600.0
JOULES_PER_KWH = 3_600_000.0


def joules_to_wh(joules: float) -> float:
    return joules / JOULES_PER_WH


def joules_to_kwh(joules: float) -> float:
    return joules / JOULES_PER_KWH
