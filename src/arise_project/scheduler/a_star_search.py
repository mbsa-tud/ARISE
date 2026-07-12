# -*- coding: utf-8 -*-

"""
Module defining an algorithm to traverse the state graph using A* algorithm.

Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import heapq
import time
from pathlib import Path

from typing import Any, Callable

from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.gui.custom.pyqt_progress_updater import DummyProgressUpdater, PyQtProgressUpdater
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.optimization_result import OptimizationResult
from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.cost_normalization import compute_cost_scales
from src.arise_project.model.product import Product
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.model.skills import TransportSkill
from src.arise_project.model.task_results import TaskResult
from src.arise_project.model.tasks import ExecutionMode
from src.arise_project.tools.output_timestamp import print_with_timestamp

OPT_RES_PARAM_EXPANSIONS = "expansions"
OPT_RES_PARAM_STALE_POPS = "stale_pops"

HeuristicFunction = Callable[[Any], float]


class PriorityQueueItem:

    def __init__(self, f: float, g: float, counter: int, path: tuple[int, ...], state_key: int,
                 time_sum: float, energy_sum: float, sequence_reliability: float,
                 task_result_history: list[TaskResult]):
        self._f = f
        self._g = g
        self._counter = counter
        self._path = path
        self._state_key = state_key

        # Cumulative cost, sampled exactly once per edge (never resampled by replay), so that
        # the g used to select this node and the totals reported for the final result agree.
        self._time_sum = time_sum
        self._energy_sum = energy_sum
        self._sequence_reliability = sequence_reliability
        self._task_result_history = task_result_history

    def __lt__(self, other):

        # First compare f(n)
        if self._f != other.f:
            return self._f < other.f

        # If equal, compare g(n)
        if self._g != other.g:
            return self._g < other.g

        # Tie-breaker
        return self._counter < other.counter

    @property
    def f(self) -> float:
        return self._f

    @property
    def g(self) -> float:
        return self._g

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def path(self) -> tuple[int, ...]:
        return self._path

    @property
    def state_key(self) -> int:
        return self._state_key

    @property
    def time_sum(self) -> float:
        return self._time_sum

    @property
    def energy_sum(self) -> float:
        return self._energy_sum

    @property
    def sequence_reliability(self) -> float:
        return self._sequence_reliability

    @property
    def task_result_history(self) -> list[TaskResult]:
        return self._task_result_history

    def __repr__(self):
        return f"PriorityQueueItem(f={self._f}, g={self._g}, counter={self._counter})"


def accumulate_edge_cost(parent_time_sum: float, parent_energy_sum: float,
                         parent_sequence_reliability: float,
                         task_result: TaskResult) -> tuple[float, float, float]:
    """
    Fold one sampled TaskResult into a path's running (time, energy, reliability) totals.

    time/energy are additive across edges, but sequence_reliability is the PRODUCT of
    per-skill reliabilities (matching ScenarioCore.execute_action), not a sum - so this must
    not be approximated as summing per-edge objective_function() values along the path.
    """

    return (parent_time_sum + task_result.total_time,
            parent_energy_sum + task_result.total_energy,
            parent_sequence_reliability * task_result.skill.reliability)


def _minimum_required_transport_cost(current_scn: ScenarioCore, product: Product,
                                     objective_function: ObjectiveFunction) -> float:
    """
    Admissible lower bound on the transport cost still required for one product.

    A product's remaining tour must visit at least one machine capable of each remaining
    processing task, plus its target location. For any one of these "required stops", the
    cumulative tour length up to visiting it is >= the straight-line distance to the nearest
    valid member of that stop (triangle inequality), and total tour length >= any such prefix.
    So `max over required stops of (min distance to that stop)` is a valid lower bound on the
    total remaining transport distance - unlike summing per-stop distances (double-counts /
    ignores that a smart route reuses travel) or building an MST/tour over each stop's
    independently-nearest machine (the specific machine chosen to minimize processing cost or
    distance need not be the one the true optimal route actually visits).

    The distance is converted to cost using whichever available transporter skill minimizes the
    COMBINED (time + energy) objective cost per unit distance - not the fastest transporter's
    time paired with its own energy, which would overestimate (and break admissibility) whenever
    a slower, lower-power transporter is cheaper overall for the real optimal solution.
    """

    current_location = product.current_state.location_machine_id
    target_location = product.target_state.location_machine_id

    # Every remaining processing task is a required stop: the product must reach some machine
    # capable of it. Only the nearest such machine matters for this stop's contribution.
    required_stop_distances = []

    for processing_task in product.get_remaining_processing_tasks():

        nearest_distance = float("inf")

        for skill_type in processing_task.possible_skill_types:
            for machine in current_scn.factory.get_machines_by_skill_type(skill_type):

                distance = current_scn.factory.get_transport_distance(current_location, machine.unique_id)

                if distance < nearest_distance:
                    nearest_distance = distance

        if nearest_distance < float("inf"):
            required_stop_distances.append(nearest_distance)

    # The product must also end up at its target location - a required stop with one member
    required_stop_distances.append(current_scn.factory.get_transport_distance(current_location, target_location))

    max_required_distance = max(required_stop_distances, default=0.0)

    if max_required_distance <= 0.0:
        return 0.0

    try:
        transporter_machines = current_scn.factory.get_machines_by_skill_type(TransportSkill)
    except ValueError:
        return 0.0

    lowest_transport_cost = float("inf")

    for machine in transporter_machines:

        transport_skill = machine.get_skill_by_type(TransportSkill)

        base_time = max_required_distance / transport_skill.execution_speed
        best_case_time = transport_skill.process_variability.time_best_case(base_time=base_time)
        best_case_energy = transport_skill.nominal_power_draw * best_case_time

        transport_cost = objective_function(time_cost=best_case_time, energy_cost=best_case_energy, reliability=1.0)

        if transport_cost < lowest_transport_cost:
            lowest_transport_cost = transport_cost

    return lowest_transport_cost if lowest_transport_cost < float("inf") else 0.0


def product_completion_heuristic(current_scn: ScenarioCore, objective_function: ObjectiveFunction) -> float:

    minimum_cost = 0.0

    # For each product
    for product in current_scn.get_sorted_product_list():

        # And for each remaining processing task of a product
        for processing_task in product.get_remaining_processing_tasks():

            lowest_processing_cost = float("inf")

            # Get all possible skill types that can be used to execute a processing task
            for skill_type in processing_task.possible_skill_types:

                # For each skill type find every machine that has a skill of this type
                for machine in current_scn.factory.get_machines_by_skill_type(skill_type):

                    processing_skill = machine.get_skill_by_type(skill_type)

                    # Considering possible process variability (noise), get the best case time and energy cost
                    total_time, total_energy, _ = processing_task.execute(processing_skill, mode=ExecutionMode.BEST_CASE)

                    # A sequence reliability of 100% is assumed as calculation of the total cost
                    # may be calculated like this: time_cost + energy_cost + (1.0 - reliability)
                    # The cost therefore only depends on the time and energy cost.
                    # Reason for this is that the sequence reliability is calculated by multiplying
                    # the reliabilities of all previous skills which are not known at this point.
                    total_cost = objective_function(time_cost=total_time, energy_cost=total_energy, reliability=1.0)

                    if total_cost < lowest_processing_cost:
                        lowest_processing_cost = total_cost

            minimum_cost += lowest_processing_cost

        minimum_cost += _minimum_required_transport_cost(current_scn, product, objective_function)

    return minimum_cost


def astar_search(
        scenario_file_path: Path,
        objective_function: ObjectiveFunction,
        use_heuristic: bool = True,
        time_limit_s: float | None = None,
        max_expansions: int | None = None,
        verbose: bool = True,
        progress_updater: PyQtProgressUpdater = DummyProgressUpdater()) -> OptimizationResult | None:

    progress_updater.text = "Start A* search"
    progress_updater.percentage = 0

    start_time = time.time()

    # Initial node (empty path at initial state)
    # Compute g(start)=0 and h(start)
    root_scn = ScenarioCore(file_path=scenario_file_path, reset_class=True)

    # Calculate h(start) or use 0.0 (Dijkstra)
    if use_heuristic:
        h0 = product_completion_heuristic(current_scn=root_scn, objective_function=objective_function)
    else:
        h0 = 0.0

    # Items contained within dict are hashable (sorting to keep hash stable and deterministic)
    start_key = hash(tuple(sorted(root_scn.get_product_states().items())))

    # Build the priority queue based on a binary heap data structure
    priority_queue: list[PriorityQueueItem] = []
    counter = 0

    first_pq_item = PriorityQueueItem(f=h0, g=0.0, counter=counter, path=tuple(), state_key=start_key,
                                      time_sum=0.0, energy_sum=0.0, sequence_reliability=1.0,
                                      task_result_history=[])
    heapq.heappush(priority_queue, first_pq_item)

    # Best g found for each state_key
    best_g: dict[int, float] = {start_key: 0.0}

    expansions = 0
    stale_pops = 0

    progress_updater.text = "Running A* ..."
    progress_updater.percentage = 25

    while priority_queue:

        if time_limit_s is not None and (time.time() - start_time) > time_limit_s:
            if verbose:
                print("[A*] Time limit reached; terminating.")
            break

        smallest_pq_item = heapq.heappop(priority_queue)

        # Skip stale entries: heapq has no decrease-key, so a state can sit in the queue under
        # multiple entries once a cheaper path to it is found. Re-expanding a superseded entry
        # would only pay for a full path replay whose children get pruned by best_g anyway -
        # skip it before that replay, not after, to actually reclaim the wasted time.
        if smallest_pq_item.g > best_g.get(smallest_pq_item.state_key, float("inf")):
            stale_pops += 1
            continue

        path = smallest_pq_item.path

        # Reconstruct scenario STATE at this node (build fresh ScenarioCore and replay "parent path").
        # This replay is only used to derive product/machine state (locations, completed tasks,
        # feasibility) which is deterministic at reliability=1.0. The reported time/energy/reliability
        # totals below intentionally come from smallest_pq_item, not from this replay: replaying
        # under process variability would resample every edge's noise and no longer match the g-value
        # that was used to select this node as best.
        scn = ScenarioCore(file_path=scenario_file_path, reset_class=True)

        # Fast-forward to node state
        for a in path:
            scn.step_by_action_idx(int(a))

        # Generate feasible actions at this node
        feasible_idx = scn.get_feasible_actions_idx_list()

        if scn.is_done():

            progress_updater.text = "Done."
            progress_updater.percentage = 100

            return OptimizationResult(action_idx_sequence=list(path),
                                      task_result_list=smallest_pq_item.task_result_history,
                                      total_time=smallest_pq_item.time_sum,
                                      total_energy=smallest_pq_item.energy_sum,
                                      sequence_reliability=smallest_pq_item.sequence_reliability,
                                      objective_function=objective_function,
                                      other_params_dict={OPT_RES_PARAM_EXPANSIONS: expansions,
                                                         OPT_RES_PARAM_STALE_POPS: stale_pops},
                                      total_duration_seconds=(time.time() - start_time),
                                      opt_method=OptimizationMethod.OPT_A_STAR)

        if expansions % 10 == 0:
            progress_updater.text = f"Expansions: {expansions}"

        # Expand
        expansions += 1

        if max_expansions is not None and expansions > max_expansions:
            if verbose:
                print("[A*] Expansion limit reached; terminating.")
            break

        # For each feasible action, compute child path and costs
        for action_idx in feasible_idx:

            child_path = path + (int(action_idx),)

            # Sample this edge's cost exactly once, by stepping 'scn' (already at 'path') forward.
            # This is the ONLY place this edge's time/energy noise is drawn for this node - it is
            # never resampled later, so the g used to rank/prune this child stays consistent with
            # whatever ends up in the returned result if this path is chosen.
            task_result, _, _ = scn.step_by_action_idx(int(action_idx))

            # Accumulate cost as a raw (time, energy, reliability) triple, not as g_parent + edge_cost:
            # the objective combines reliability multiplicatively/nonlinearly, so it is not additive
            # across edges. Recomputing the objective on the cumulative triple reproduces
            # ScenarioCore.calculate_total_cost() exactly.
            child_time_sum, child_energy_sum, child_sequence_reliability = accumulate_edge_cost(
                smallest_pq_item.time_sum, smallest_pq_item.energy_sum,
                smallest_pq_item.sequence_reliability, task_result)
            child_task_result_history = smallest_pq_item.task_result_history + [task_result]

            g_child = objective_function(time_cost=child_time_sum, energy_cost=child_energy_sum,
                                         reliability=child_sequence_reliability)

            # Items contained within dict are hashable (sorting to keep hash stable and deterministic)
            child_key = hash(tuple(sorted(scn.get_product_states().items())))

            # Prune by best g
            prev_best = best_g.get(child_key, float("inf"))

            if g_child >= prev_best:
                scn.undo_last_action()
                continue

            # Calculate h child (scn is now AT the child's state) or use 0.0 (Dijkstra)
            if use_heuristic:
                h_child = product_completion_heuristic(current_scn=scn, objective_function=objective_function)
            else:
                h_child = 0.0

            # Undo to restore 'scn' to 'path' state for the next sibling action
            scn.undo_last_action()

            f_child = g_child + h_child

            counter += 1
            new_pq_item = PriorityQueueItem(f=f_child, g=g_child, counter=counter, path=child_path,
                                            state_key=child_key, time_sum=child_time_sum,
                                            energy_sum=child_energy_sum,
                                            sequence_reliability=child_sequence_reliability,
                                            task_result_history=child_task_result_history)

            heapq.heappush(priority_queue, new_pq_item)
            best_g[child_key] = g_child

    progress_updater.text = "Done."
    progress_updater.percentage = 100

    # If we exit the loop without returning, no solution was found within limits
    print_with_timestamp(f"No solution found after {expansions} expansions... ")
    return None


if __name__ == '__main__':

    time_scale, energy_scale, reliability_scale = compute_cost_scales(
        ScenarioCore(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, reset_class=True))

    opt_result = astar_search(
        scenario_file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH,
        objective_function=ObjectiveFunction(time_weight=1/3, energy_weight=1/3, reliability_weight=1/3,
                                             time_scale=time_scale, energy_scale=energy_scale,
                                             reliability_scale=reliability_scale),
        use_heuristic=True,
        time_limit_s=None,
        max_expansions=None,
        verbose=True
    )

    if opt_result is not None:

        print("\n[A*] Optimal path found:", opt_result.action_idx_sequence)

        print(f"[A*] cost={opt_result.total_cost:.4f} | "
              f"time={opt_result.total_time:.4f} | "
              f"energy={opt_result.total_energy:.4f} | "
              f"reliability={opt_result.sequence_reliability:.6f} | "
              f"expansions={opt_result.other_params_dict['expansions']} | "
              f"elapsed={opt_result.total_duration_seconds:.2f}s\n")

        opt_result.print_task_result_history(show_numerical_index=True, show_action_index=True)

    else:
        print("\n[A*] No solution within limits.")