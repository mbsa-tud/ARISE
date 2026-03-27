# -*- coding: utf-8 -*-

"""
Module defining the objective function class.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"


class ObjectiveFunction:

    def __init__(self, time_weight: float = 0.5, energy_weight: float = 0.5, reliability_weight: float = 0.0):

        if time_weight + energy_weight + reliability_weight > 1.0:
            raise ValueError(f"ERROR: Sum of weights in objective function may not exceed 1.0 "
                             f"(sum: {time_weight + energy_weight + reliability_weight:.3f})")

        self._time_weight = time_weight
        self._energy_weight = energy_weight
        self._reliability_weight = reliability_weight

    def __call__(self, time_cost: float, energy_cost: float, reliability: float) -> float:
        return self._time_weight * time_cost + self._energy_weight * energy_cost + self._reliability_weight * (1.0 - reliability)
