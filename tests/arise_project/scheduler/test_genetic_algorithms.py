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

from pathlib import Path

import pytest

from src.arise_project.model.nsga_config import NSGAConfig
from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.scheduler.genetic_algorithms import (
    OPT_RES_PARAM_BEST_TRIAL_INDEX,
    OPT_RES_PARAM_NUM_TRIALS,
    run_nsga,
)

TINY_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scenario.json"

EXPECTED_TOTAL_COST = 8.0 / 3.0


def _objective_function() -> ObjectiveFunction:
    return ObjectiveFunction(time_weight=1 / 3, energy_weight=1 / 3, reliability_weight=1 / 3)


def _tiny_nsga_config() -> NSGAConfig:
    # Deliberately small population/generations so this stays fast in CI; the tiny scenario's
    # optimum is trivial enough that even a small population finds it.
    return NSGAConfig(max_sequence_length=6, population_size=8, number_generations=3,
                      mutation_probability=0.2, penalty=1e4)


def test_run_nsga_multi_trial_finds_known_optimum():

    result = run_nsga(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                      opt_method=OptimizationMethod.OPT_NSGA2, nsga_config=_tiny_nsga_config(),
                      num_trials=3, base_seed=7)

    assert result is not None
    assert result.total_cost == pytest.approx(EXPECTED_TOTAL_COST)


def test_run_nsga_reports_trial_bookkeeping():

    num_trials = 3

    result = run_nsga(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                      opt_method=OptimizationMethod.OPT_NSGA2, nsga_config=_tiny_nsga_config(),
                      num_trials=num_trials, base_seed=7)

    assert result is not None
    assert result.other_params_dict[OPT_RES_PARAM_NUM_TRIALS] == num_trials
    assert 0 <= result.other_params_dict[OPT_RES_PARAM_BEST_TRIAL_INDEX] < num_trials


def test_run_nsga_single_trial_still_works():
    """num_trials=1 must behave like the old single-run behavior (backward compatible)."""

    result = run_nsga(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                      opt_method=OptimizationMethod.OPT_NSGA2, nsga_config=_tiny_nsga_config(),
                      num_trials=1, base_seed=7)

    assert result is not None
    assert result.other_params_dict[OPT_RES_PARAM_NUM_TRIALS] == 1
    assert result.other_params_dict[OPT_RES_PARAM_BEST_TRIAL_INDEX] == 0