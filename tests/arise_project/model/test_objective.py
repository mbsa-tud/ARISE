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
"""

import pytest

from src.arise_project.model.objective import ObjectiveFunction


def test_objective_function_weighted_combination():

    objective_function = ObjectiveFunction(time_weight=0.3, energy_weight=0.3, reliability_weight=0.4)

    cost = objective_function(time_cost=10.0, energy_cost=5.0, reliability=0.9)

    expected = 0.3 * 10.0 + 0.3 * 5.0 + 0.4 * (1.0 - 0.9)
    assert cost == pytest.approx(expected)


def test_objective_function_zero_weights_gives_zero_cost():

    objective_function = ObjectiveFunction(time_weight=0.0, energy_weight=0.0, reliability_weight=0.0)

    assert objective_function(time_cost=1000.0, energy_cost=1000.0, reliability=0.0) == pytest.approx(0.0)


def test_objective_function_reliability_only():

    objective_function = ObjectiveFunction(time_weight=0.0, energy_weight=0.0, reliability_weight=1.0)

    # Perfect reliability contributes no cost
    assert objective_function(time_cost=50.0, energy_cost=50.0, reliability=1.0) == pytest.approx(0.0)

    # Worst-case reliability contributes the full weight as cost
    assert objective_function(time_cost=50.0, energy_cost=50.0, reliability=0.0) == pytest.approx(1.0)


def test_objective_function_rejects_weights_summing_above_one():

    with pytest.raises(ValueError):
        ObjectiveFunction(time_weight=0.6, energy_weight=0.6, reliability_weight=0.0)
