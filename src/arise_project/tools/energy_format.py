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
