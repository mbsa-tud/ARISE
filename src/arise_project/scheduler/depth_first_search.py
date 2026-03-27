# -*- coding: utf-8 -*-

"""
Module defining an algorithm to traverse the state graph using depth first search (DFS).
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from datetime import datetime
import time
from pathlib import Path

import numpy as np
from typing import Iterable

from arise_project.gui.custom.pyqt_progress_updater import DummyProgressUpdater
from arise_project.model.objective import ObjectiveFunction
from arise_project.model.optimization_method import OptimizationMethod
from arise_project.model.optimization_result import OptimizationResult
from arise_project.tools.output_timestamp import print_with_timestamp
from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.model.scenario import Scenario


def dfs_enumerate(scenario: Scenario, max_depth: int | None = None, avoid_cycles: bool = True) -> Iterable[list[int]]:
    """
    Depth-first enumeration of feasible action sequences via backtracking.

    Yields sequences (list of action indices).
    """

    path: list[int] = []

    # Track states along the current path to avoid cycles on the branch
    branch_seen = set() if avoid_cycles else None

    def _dfs() -> Iterable[list[int]]:

        # Goal check (before expanding)
        if scenario.is_done():

            yield path.copy()
            return

        # Depth bound
        if max_depth is not None and len(path) >= max_depth:

            yield path.copy()
            return

        # Cycle detection (branch-local)
        if avoid_cycles:

            # Items contained within dict are hashable (sorting to keep hash stable and deterministic)
            key = hash(tuple(sorted(scenario.get_product_states().items())))

            if key in branch_seen:
                return

            branch_seen.add(key)

        action_mask = scenario.generate_feasible_action_mask()
        feasible_action_idx_array = np.flatnonzero(action_mask)

        if feasible_action_idx_array.size > 0:

            # For each feasible action in the current state
            for action_idx in feasible_action_idx_array:

                # Execute action and update path
                scenario.step_by_action_idx(action_idx)
                path.append(int(action_idx))

                # Recursive call
                yield from _dfs()

                # Backtrack (undo last action)
                path.pop()
                scenario.undo_last_action()

        else:

            # Terminal: no more actions (dead end)
            yield path.copy()

        if avoid_cycles:

            # Remove current state when backtracking
            branch_seen.remove(key)

    yield from _dfs()


def run_iddfs(scenario_file_path: Path, objective_function: ObjectiveFunction, opt_method: OptimizationMethod,
              max_steps: int = 20, verbose: bool = False,
              progress_updater=DummyProgressUpdater()) -> OptimizationResult:

    progress_updater.text = f"Starting {opt_method}"
    progress_updater.percentage = 0

    # Load a scenario (product and factory)
    example_scenario = Scenario(file_path=scenario_file_path, reset_class=True)

    # Multi-criteria objective, can be split into individual objectives
    min_total_cost = float("inf")
    min_total_cost_sequence = []
    min_solution_depth = 0

    if opt_method is OptimizationMethod.OPT_DFS:
        print(f"Starting Depth-First Search (DFS) with a maximum of {max_steps} steps... ")
        max_depth_list = [max_steps]

    elif opt_method is OptimizationMethod.OPT_IDDFS:
        print(f"Starting Iterative Deepening Depth-First Search (IDDFS) with a maximum of {max_steps} steps... ")
        max_depth_list = range(max_steps + 1)

    else:
        raise ValueError(f"Error: Algorithm {opt_method} must be DFS or IDDFS.")

    solution_found = False

    start_time = time.time()

    for maximum_depth in max_depth_list:

        seq_counter = 0

        progress_updater.text = f"Current maximum depth: {maximum_depth}"
        progress_updater.percentage = int(round(maximum_depth / max(max_depth_list), 0))

        if verbose:
            print_with_timestamp(f"--- MAXIMUM DEPTH: {maximum_depth} ---")

        for seq in dfs_enumerate(scenario=example_scenario, max_depth=maximum_depth, avoid_cycles=False):

            seq_counter += 1

            if seq_counter % 1000 == 0:
                print_with_timestamp(f"Progress: {seq_counter} sequences (max depth of {maximum_depth})")

            # print(f"Found: {seq}")

            done, steps_used, _ = example_scenario.execute_action_idx_sequence(np.array(seq),
                                                                               check_validity=False,
                                                                               random_seed=None)

            total_time = example_scenario.time_sum
            total_energy = example_scenario.energy_sum
            sequence_reliability = example_scenario.sequence_reliability

            if done:



                total_cost = objective_function(time_cost=example_scenario.time_sum,
                                                energy_cost=example_scenario.energy_sum,
                                                reliability=example_scenario.sequence_reliability)

                output_str = f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Successful path found [{steps_used}] "\
                             f"-> time: {total_time:.2f} & energy: {total_energy:.2f} & "\
                             f"reliability: {sequence_reliability:.2f}"

                if total_cost < min_total_cost:

                    min_total_cost = total_cost
                    min_total_cost_sequence = seq

                    output_str += f"-> New minimum total cost: {min_total_cost:.2f}"

                print_with_timestamp(output_str)

                # Keep track to stop at this depth
                solution_found = True

        if solution_found:
            min_solution_depth = maximum_depth
            break

    print("\n")
    print_with_timestamp(f"DFS done after {time.time() - start_time:.2f} seconds -> minimum cost: {min_total_cost:.2f} -> {min_total_cost_sequence}")

    progress_updater.text = f"Done."
    progress_updater.percentage = 100

    example_scenario.reset()

    # Re-simulate to get actual actions taken until done
    _, _, actions_taken = example_scenario.execute_action_idx_sequence(np.array(min_total_cost_sequence))

    return OptimizationResult(action_idx_sequence=list(min_total_cost_sequence),
                              task_result_list=example_scenario.task_result_history,
                              total_time=example_scenario.time_sum,
                              total_energy=example_scenario.energy_sum,
                              sequence_reliability=example_scenario.sequence_reliability,
                              objective_function=objective_function,
                              other_params_dict={"min_solution_depth": min_solution_depth},
                              total_duration_seconds=(time.time() - start_time),
                              opt_method=opt_method)


if __name__ == '__main__':

    objective_function = ObjectiveFunction(time_weight=1/3,
                                           energy_weight=1/3,
                                           reliability_weight=1/3)

    run_iddfs(scenario_file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH,
              objective_function=objective_function,
              opt_method=OptimizationMethod.OPT_IDDFS,
              max_steps=20,
              verbose=True)

