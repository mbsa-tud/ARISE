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

import numpy as np

from arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from arise_project.gui.custom.pyqt_progress_updater import DummyProgressUpdater, PyQtProgressUpdater
from arise_project.model.optimization_method import OptimizationMethod
from arise_project.model.optimization_result import OptimizationResult
from arise_project.model.objective import ObjectiveFunction
from arise_project.model.scenario import Scenario
from arise_project.model.tasks import ExecutionMode
from arise_project.tools.output_timestamp import print_with_timestamp

HeuristicFunction = Callable[[Any], float]


class PriorityQueueItem:

    def __init__(self, f: float, g: float, counter: int, path: tuple[int, ...], state_key: int):
        self._f = f
        self._g = g
        self._counter = counter
        self._path = path
        self._state_key = state_key

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

    def __repr__(self):
        return f"PriorityQueueItem(f={self._f}, g={self._g}, counter={self._counter})"


def product_completion_heuristic(current_scn: Scenario, objective_function: ObjectiveFunction) -> float:

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

    # TODO Improve heuristic by adding minimum transport costs (f.e. from current position to target location?)

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
    root_scn = Scenario(file_path=scenario_file_path, reset_class=True)

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

    first_pq_item = PriorityQueueItem(f=h0, g=0.0, counter=counter, path=tuple(), state_key=start_key)
    heapq.heappush(priority_queue, first_pq_item)

    # Best g found for each state_key
    best_g: dict[int, float] = {start_key: 0.0}

    expansions = 0

    progress_updater.text = "Running A* ..."
    progress_updater.percentage = 25

    while priority_queue:

        if time_limit_s is not None and (time.time() - start_time) > time_limit_s:
            if verbose:
                print("[A*] Time limit reached; terminating.")
            break

        smallest_pq_item = heapq.heappop(priority_queue)
        path = smallest_pq_item.path

        # Reconstruct scenario at this node (build fresh Scenario and replay "parent path")
        scn = Scenario(file_path=scenario_file_path, reset_class=True)

        # Fast-forward to node state
        for a in path:
            scn.step_by_action_idx(int(a))

        # Generate feasible actions at this node
        feasible_idx = scn.get_feasible_actions_idx_list()

        if scn.is_done():

            progress_updater.text = "Done."
            progress_updater.percentage = 100

            return OptimizationResult(action_idx_sequence=list(path),
                                      task_result_list=scn.task_result_history,
                                      total_time=scn.time_sum,
                                      total_energy=scn.energy_sum,
                                      sequence_reliability=scn.sequence_reliability,
                                      objective_function=objective_function,
                                      other_params_dict={"expansions": expansions},
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

            # Compute g(child) by evaluating the partial path with the objective
            scn_child = Scenario(file_path=scenario_file_path, reset_class=True)

            # Execute the sequence (partial or complete)
            done, steps_used, _ = scn_child.execute_action_idx_sequence(
                np.array(list(child_path)),
                check_validity=False,
                random_seed=None
            )

            g_child = scn_child.calculate_total_cost(objective_function=objective_function)

            del scn_child

            # Get the child's state key (by stepping once on a temporary scenario)
            # Reuse scn: it's already at 'path'. Step and undo around it for local key.
            scn.step_by_action_idx(int(action_idx))

            # Items contained within dict are hashable (sorting to keep hash stable and deterministic)
            child_key = hash(tuple(sorted(scn.get_product_states().items())))

            scn.undo_last_action()

            # Prune by best g
            prev_best = best_g.get(child_key, float("inf"))

            if g_child >= prev_best:
                continue

            # --- Compute heuristic at child ---

            # Build a child scenario to pass into heuristic (cheap one-step replay from 'scn'):
            scn.step_by_action_idx(int(action_idx))

            # Calculate h child or use 0.0 (Dijkstra)
            if use_heuristic:
                h_child = product_completion_heuristic(current_scn=root_scn, objective_function=objective_function)
            else:
                h_child = 0.0

            # Undo to avoid new instance
            scn.undo_last_action()

            f_child = g_child + h_child

            counter += 1
            new_pq_item = PriorityQueueItem(f=f_child, g=g_child, counter=counter, path=child_path, state_key=child_key)

            heapq.heappush(priority_queue, new_pq_item)
            best_g[child_key] = g_child

    progress_updater.text = "Done."
    progress_updater.percentage = 100

    # If we exit the loop without returning, no solution was found within limits
    print_with_timestamp(f"No solution found after {expansions} expansions... ")
    return None


if __name__ == '__main__':

    opt_result = astar_search(
        scenario_file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH,
        objective_function=ObjectiveFunction(time_weight=1/3, energy_weight=1/3, reliability_weight=1/3),
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