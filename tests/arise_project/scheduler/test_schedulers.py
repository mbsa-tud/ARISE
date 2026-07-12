import json
import random
from pathlib import Path

import pytest

from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.model.skills import TransportSkill
from src.arise_project.model.task_results import TaskResult
from src.arise_project.scheduler.a_star_search import accumulate_edge_cost, astar_search, product_completion_heuristic
from src.arise_project.scheduler.depth_first_search import run_iddfs

# Deliberately tiny (1 storage + 1 drilling machine + 1 AGV, 1 task) so exhaustive search stays fast in CI
TINY_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scenario.json"

# 3 drilling tasks x 3 candidate machines: enough branching for the heuristic to actually prune
# search, while still resolving in well under a second (no exhaustive DFS/IDDFS run against this one).
SMALL_BRANCHING_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "small_branching_scenario.json"

# Two transporters with very different speed/power tradeoffs, used to check that the transport
# term of the A* heuristic stays admissible (see test_astar_heuristic_stays_admissible_with_heterogeneous_transporters)
HETEROGENEOUS_TRANSPORTERS_SCENARIO_PATH = (Path(__file__).resolve().parents[2] / "fixtures"
                                            / "heterogeneous_transporters_scenario.json")

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


def test_heuristic_reflects_current_state_not_initial_state():
    """
    Regression test for a bug where the A* heuristic was always evaluated against the initial
    (zero-progress) scenario state instead of the current search node's state, making it a
    constant across every node and silently degrading A* into plain Dijkstra.
    """

    objective_function = _objective_function()

    scn = ScenarioCore(file_path=TINY_SCENARIO_PATH, reset_class=True)

    h_before = product_completion_heuristic(current_scn=scn, objective_function=objective_function)
    assert h_before > 0.0

    # Transport to DM1, then drill: completes the scenario's only processing task
    scn.step_by_action_idx(1)
    scn.step_by_action_idx(0)

    # Processing is done, but the product must still transport back to its target location (ST1)
    # - the heuristic must reflect that remaining cost, not report 0 just because processing is done.
    h_after_drilling = product_completion_heuristic(current_scn=scn, objective_function=objective_function)

    assert h_after_drilling > 0.0
    assert h_after_drilling < h_before

    # Transport back to ST1: now nothing remains, heuristic must hit exactly 0
    scn.step_by_action_idx(2)

    h_after_full_sequence = product_completion_heuristic(current_scn=scn, objective_function=objective_function)

    assert h_after_full_sequence == pytest.approx(0.0)


def test_heuristic_reduces_expansions_versus_dijkstra():
    """
    With a working (state-dependent) heuristic, A* must expand fewer *genuine* nodes than
    plain Dijkstra (use_heuristic=False) on a scenario with real branching, while still finding
    the same optimum. Before the heuristic fix this held only as an equality; before the
    stale-pop fix, the reported 'expansions' counter also included wasted re-pops of queue
    entries superseded by a cheaper path (heapq has no decrease-key) - and that waste is
    distributed differently between the two modes, so it could make A* look worse than
    Dijkstra even though its real expansion count was lower. Both fixes are needed for
    'expansions' to be a fair number to compare across modes/algorithms.
    """

    objective_function = _objective_function()

    with_heuristic = astar_search(scenario_file_path=SMALL_BRANCHING_SCENARIO_PATH,
                                  objective_function=objective_function, use_heuristic=True, verbose=False)
    without_heuristic = astar_search(scenario_file_path=SMALL_BRANCHING_SCENARIO_PATH,
                                     objective_function=objective_function, use_heuristic=False, verbose=False)

    assert with_heuristic is not None
    assert without_heuristic is not None
    assert with_heuristic.total_cost == pytest.approx(without_heuristic.total_cost)

    assert (with_heuristic.other_params_dict["expansions"]
            < without_heuristic.other_params_dict["expansions"])

    # Diagnostic counter must be tracked (not folded into the headline 'expansions' number)
    assert with_heuristic.other_params_dict["stale_pops"] >= 0
    assert without_heuristic.other_params_dict["stale_pops"] >= 0


def test_accumulate_edge_cost_combines_reliability_multiplicatively():
    """
    Guards against a correctness trap: the objective's reliability term is a nonlinear function
    of the PRODUCT of per-skill reliabilities along a path, not a sum. A naive
    g_child = g_parent + objective_function(delta) implementation would silently return
    sub-optimal paths as soon as any skill's reliability drops below 1.0, even though it passes
    every other test while all reliabilities are 1.0 (as in every checked-in scenario today).
    """

    skill_1 = TransportSkill(execution_speed=1.0, nominal_power_draw=1.0, reliability=0.9)
    skill_2 = TransportSkill(execution_speed=1.0, nominal_power_draw=1.0, reliability=0.8)

    result_1 = TaskResult(product=None, machine=None, task=None, skill=skill_1,
                          total_time=2.0, total_energy=3.0, success_bool=True)
    result_2 = TaskResult(product=None, machine=None, task=None, skill=skill_2,
                          total_time=1.0, total_energy=1.0, success_bool=True)

    time_1, energy_1, reliability_1 = accumulate_edge_cost(0.0, 0.0, 1.0, result_1)
    time_2, energy_2, reliability_2 = accumulate_edge_cost(time_1, energy_1, reliability_1, result_2)

    assert time_2 == pytest.approx(3.0)
    assert energy_2 == pytest.approx(4.0)

    # Correct: cumulative reliability is the PRODUCT of the per-edge reliabilities.
    assert reliability_2 == pytest.approx(0.9 * 0.8)

    # The bug this guards against: summing "1 - reliability" penalties per edge instead of
    # multiplying reliabilities gives a different result once more than one edge is unreliable.
    naive_additive_reliability = 1.0 - ((1.0 - 0.9) + (1.0 - 0.8))
    assert reliability_2 != pytest.approx(naive_additive_reliability)


def test_astar_stays_internally_consistent_under_process_variability(tmp_path):
    """
    A* must still return internally consistent results when process variability is enabled:
    each edge's cost is sampled exactly once (never resampled by a replay), so the reported
    totals equal the sum of the per-action task results that were actually used to select the
    returned path.
    """

    scenario_data = json.loads(TINY_SCENARIO_PATH.read_text())
    scenario_data["variability_configurations"][0]["uniform_time_variability"] = 0.5

    noisy_scenario_path = tmp_path / "tiny_scenario_with_variability.json"
    noisy_scenario_path.write_text(json.dumps(scenario_data))

    random.seed(1234)

    result = astar_search(scenario_file_path=noisy_scenario_path, objective_function=_objective_function(),
                          use_heuristic=True, verbose=False)

    assert result is not None
    assert result.action_idx_sequence == EXPECTED_ACTION_SEQUENCE

    summed_time = sum(task_result.total_time for task_result in result.task_result_list)
    summed_energy = sum(task_result.total_energy for task_result in result.task_result_list)

    assert result.total_time == pytest.approx(summed_time)
    assert result.total_energy == pytest.approx(summed_energy)


def test_astar_heuristic_stays_admissible_with_heterogeneous_transporters():
    """
    Regression test for a correctness trap in the heuristic's transport-cost term: converting
    the transport-distance lower bound to cost using the FASTEST transporter's time paired with
    that SAME transporter's energy is not admissible when transporters differ in speed/power - a
    slower, lower-power transporter can be cheaper overall, and the heuristic would then
    overestimate true remaining cost, breaking A*'s optimality guarantee. It must instead use
    whichever transporter minimizes the combined objective cost.

    Uses heavily energy-weighted objective weights so the true optimum clearly prefers the slow,
    low-power transporter (confirmed below by inspecting which skill was actually used), then
    checks A* (heuristic-guided), A* as Dijkstra, and IDDFS all agree on the same optimal cost.
    """

    objective_function = ObjectiveFunction(time_weight=0.01, energy_weight=0.98, reliability_weight=0.01)

    astar_heuristic_result = astar_search(scenario_file_path=HETEROGENEOUS_TRANSPORTERS_SCENARIO_PATH,
                                          objective_function=objective_function, use_heuristic=True, verbose=False)
    astar_dijkstra_result = astar_search(scenario_file_path=HETEROGENEOUS_TRANSPORTERS_SCENARIO_PATH,
                                         objective_function=objective_function, use_heuristic=False, verbose=False)
    iddfs_result = run_iddfs(scenario_file_path=HETEROGENEOUS_TRANSPORTERS_SCENARIO_PATH,
                             objective_function=objective_function, opt_method=OptimizationMethod.OPT_IDDFS,
                             max_steps=6, verbose=False)

    assert astar_heuristic_result is not None
    assert astar_dijkstra_result is not None

    assert astar_heuristic_result.total_cost == pytest.approx(astar_dijkstra_result.total_cost)
    assert astar_heuristic_result.total_cost == pytest.approx(iddfs_result.total_cost)

    # Sanity check that this scenario is actually testing what it claims: the true optimum must
    # use the slow transporter (AGV_SLOW), not the fast one, for every transport leg - otherwise
    # the two transporters aren't meaningfully different for this test.
    transporter_names_used = {task_result.machine.name for task_result in astar_heuristic_result.task_result_list
                              if task_result.machine.name in ("AGV_FAST", "AGV_SLOW")}
    assert transporter_names_used == {"AGV_SLOW"}