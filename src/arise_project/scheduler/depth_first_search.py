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

from src.arise_project.gui.custom.pyqt_progress_updater import DummyProgressUpdater
from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.cost_normalization import compute_cost_scales
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.optimization_result import OptimizationResult
from src.arise_project.tools.output_timestamp import print_with_timestamp
from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.model.scenario import ScenarioCore

OPT_RES_PARAM_MIN_SOLUTION_DEPTH = "min_solution_depth"


def dfs_enumerate(scenario: ScenarioCore, max_depth: int | None = None, avoid_cycles: bool = True) -> Iterable[list[int]]:
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
    example_scenario = ScenarioCore(file_path=scenario_file_path, reset_class=True)

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
                              other_params_dict={OPT_RES_PARAM_MIN_SOLUTION_DEPTH: min_solution_depth},
                              total_duration_seconds=(time.time() - start_time),
                              opt_method=opt_method)


if __name__ == '__main__':

    time_scale, energy_scale, reliability_scale = compute_cost_scales(
        ScenarioCore(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, reset_class=True))

    objective_function = ObjectiveFunction(time_weight=1/3,
                                           energy_weight=1/3,
                                           reliability_weight=1/3,
                                           time_scale=time_scale,
                                           energy_scale=energy_scale,
                                           reliability_scale=reliability_scale)

    run_iddfs(scenario_file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH,
              objective_function=objective_function,
              opt_method=OptimizationMethod.OPT_IDDFS,
              max_steps=20,
              verbose=True)

