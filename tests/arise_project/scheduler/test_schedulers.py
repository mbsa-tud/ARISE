from pathlib import Path

import pytest

from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.scheduler.a_star_search import astar_search
from src.arise_project.scheduler.depth_first_search import run_iddfs

# Deliberately tiny (1 storage + 1 drilling machine + 1 AGV, 1 task) so exhaustive search stays fast in CI
TINY_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scenario.json"

# Known-optimal path for the tiny scenario: transport to DM1 -> drill -> transport back to ST1
EXPECTED_ACTION_SEQUENCE = [1, 0, 2]
EXPECTED_TOTAL_TIME = 4.0
EXPECTED_TOTAL_ENERGY = 4.0
EXPECTED_TOTAL_COST = 8.0 / 3.0


def _objective_function() -> ObjectiveFunction:
    return ObjectiveFunction(time_weight=1 / 3, energy_weight=1 / 3, reliability_weight=1 / 3)


def test_astar_finds_optimal_golden_path():

    result = astar_search(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                          use_heuristic=True, verbose=False)

    assert result is not None
    assert result.action_idx_sequence == EXPECTED_ACTION_SEQUENCE
    assert result.total_time == pytest.approx(EXPECTED_TOTAL_TIME)
    assert result.total_energy == pytest.approx(EXPECTED_TOTAL_ENERGY)
    assert result.sequence_reliability == pytest.approx(1.0)
    assert result.total_cost == pytest.approx(EXPECTED_TOTAL_COST)


def test_astar_without_heuristic_matches_dijkstra_behavior():

    # use_heuristic=False degrades A* to Dijkstra; must still find the same optimum
    result = astar_search(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                          use_heuristic=False, verbose=False)

    assert result is not None
    assert result.action_idx_sequence == EXPECTED_ACTION_SEQUENCE
    assert result.total_cost == pytest.approx(EXPECTED_TOTAL_COST)


def test_dfs_finds_optimal_golden_path():

    result = run_iddfs(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                       opt_method=OptimizationMethod.OPT_DFS, max_steps=6, verbose=False)

    assert result.action_idx_sequence == EXPECTED_ACTION_SEQUENCE
    assert result.total_cost == pytest.approx(EXPECTED_TOTAL_COST)


def test_iddfs_finds_optimal_golden_path():

    result = run_iddfs(scenario_file_path=TINY_SCENARIO_PATH, objective_function=_objective_function(),
                       opt_method=OptimizationMethod.OPT_IDDFS, max_steps=6, verbose=False)

    assert result.action_idx_sequence == EXPECTED_ACTION_SEQUENCE
    assert result.total_cost == pytest.approx(EXPECTED_TOTAL_COST)


def test_all_schedulers_agree_on_the_same_optimum():
    """Regression guard: independent algorithms must converge on the same cost for a deterministic scenario."""

    objective_function = _objective_function()

    astar_result = astar_search(scenario_file_path=TINY_SCENARIO_PATH, objective_function=objective_function,
                                use_heuristic=True, verbose=False)
    dfs_result = run_iddfs(scenario_file_path=TINY_SCENARIO_PATH, objective_function=objective_function,
                           opt_method=OptimizationMethod.OPT_DFS, max_steps=6, verbose=False)
    iddfs_result = run_iddfs(scenario_file_path=TINY_SCENARIO_PATH, objective_function=objective_function,
                             opt_method=OptimizationMethod.OPT_IDDFS, max_steps=6, verbose=False)

    assert astar_result.total_cost == pytest.approx(dfs_result.total_cost)
    assert astar_result.total_cost == pytest.approx(iddfs_result.total_cost)