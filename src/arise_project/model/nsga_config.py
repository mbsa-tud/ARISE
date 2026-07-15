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

Module defining the NSGA configuration data class.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"


class NSGAConfig:

    def __init__(self,
                 max_sequence_length: int = 100,
                 population_size: int = 80,
                 number_generations: int = 100,
                 mutation_probability: float = 0.15,
                 penalty: float = 1e4) -> None:

        self._max_sequence_length = max_sequence_length
        self._population_size = population_size
        self._number_generations = number_generations
        self._mutation_probability = mutation_probability
        self._penalty = penalty

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def number_generations(self) -> int:
        return self._number_generations

    @property
    def mutation_probability(self) -> float:
        return self._mutation_probability

    @property
    def penalty(self) -> float:
        return self._penalty

    def print_str(self) -> str:
        return (f"Max. Sequence Length: {self._max_sequence_length}, "
                f"Population: {self._population_size}, "
                f"Generations: {self._number_generations}, "
                f"Mutation Probability: {self._mutation_probability:.3%}")
