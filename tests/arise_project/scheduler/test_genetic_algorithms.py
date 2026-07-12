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