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

Module defining the class for modeling process variability

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import random


class ProcessVariability:

    def __init__(self,
                 use_normal_distribution: bool = False,
                 uniform_time_variability: float = 0.0,
                 normal_dist_sigma_factor: float = 0.0) -> None:

        self._use_normal_distribution = use_normal_distribution
        self._time_variability = uniform_time_variability
        self._sigma_factor = normal_dist_sigma_factor

    def time_with_variability(self, base_time: float) -> float:

        if self._use_normal_distribution:

            return random.gauss(mu=base_time, sigma=(base_time * self._sigma_factor))

        else:

            return base_time * random.uniform(a=(1 - self._time_variability), b=(1 + self._time_variability))

    def time_best_case(self, base_time: float) -> float:
        return base_time * (1 - self._time_variability)

    def time_worst_case(self, base_time: float) -> float:
        return base_time * (1 + self._time_variability)