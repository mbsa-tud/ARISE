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

Module defining the objective function class.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"


class ObjectiveFunction:

    def __init__(self, time_weight: float = 0.5, energy_weight: float = 0.5, reliability_weight: float = 0.0,
                 time_scale: float = 1.0, energy_scale: float = 1.0, reliability_scale: float = 1.0):
        """
        :param time_scale: Reference magnitude that time_cost is divided by before weighting, so that
        time_weight/energy_weight/reliability_weight express relative importance rather than being at
        the mercy of whatever raw units/typical magnitudes each term happens to have. Must be derived
        from scenario data alone (e.g. via cost_normalization.compute_cost_scales), never from an
        optimization result, so that it stays a fixed constant and every scheduling method can be
        compared fairly against the same scale. Defaults to 1.0 (no normalization).
        :param energy_scale: Same as time_scale, but for energy_cost.
        :param reliability_scale: Same idea, but for (1 - reliability). Real per-skill reliabilities
        cluster near 1.0, so raw (1 - reliability) is typically tiny - without this scale,
        reliability_weight ends up contributing almost nothing regardless of how it's set.
        """

        if time_weight + energy_weight + reliability_weight > 1.0:
            raise ValueError(f"ERROR: Sum of weights in objective function may not exceed 1.0 "
                             f"(sum: {time_weight + energy_weight + reliability_weight:.3f})")

        if time_scale <= 0 or energy_scale <= 0 or reliability_scale <= 0:
            raise ValueError("time_scale, energy_scale and reliability_scale must be positive.")

        self._time_weight = time_weight
        self._energy_weight = energy_weight
        self._reliability_weight = reliability_weight
        self._time_scale = time_scale
        self._energy_scale = energy_scale
        self._reliability_scale = reliability_scale

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @property
    def energy_scale(self) -> float:
        return self._energy_scale

    @property
    def reliability_scale(self) -> float:
        return self._reliability_scale

    def __call__(self, time_cost: float, energy_cost: float, reliability: float) -> float:

        normalized_time_cost = time_cost / self._time_scale
        normalized_energy_cost = energy_cost / self._energy_scale
        normalized_unreliability = (1.0 - reliability) / self._reliability_scale

        return (self._time_weight * normalized_time_cost
                + self._energy_weight * normalized_energy_cost
                + self._reliability_weight * normalized_unreliability)
