# -*- coding: utf-8 -*-

"""
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