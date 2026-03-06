# -*- coding: utf-8 -*-

"""
Module defining an algorithm to traverse the state graph using depth first search (DFS).
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from datetime import datetime
import time

import numpy as np
from typing import List, Optional, Iterable

from src.arise_project.config.paths import DIR_DATA_INPUT_SCENARIOS_JSON_PATH
from src.arise_project.model.scenario import Scenario


def dfs_enumerate(scenario: Scenario, max_depth: Optional[int] = None, avoid_cycles: bool = True) -> Iterable[List[int]]:
    """
    Depth-first enumeration of feasible action sequences via backtracking.

    Yields sequences (list of action indices).
    """

    path: List[int] = []

    # Track states along the current path to avoid cycles on the branch
    branch_seen = set() if avoid_cycles else None

    def _dfs() -> Iterable[List[int]]:

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

            key = hash(scenario.get_current_state())

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


if __name__ == '__main__':

    # Load a scenario (product and factory)
    example_scenario = Scenario(file_path=DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_plate_factory.json")

    max_steps = 20
    algorithm = "iddfs"
    use_reliability = True

    # Multi-criteria objective, can be split into individual objectives
    min_time_energy_cost = 999_999_999
    min_time_energy_cost_sequence = []

    if algorithm.lower() == "dfs":
        print(f"Starting Depth-First Search (DFS) with a maximum of {max_steps} steps... ")
        max_depth_list = [max_steps]

    elif algorithm.lower() == "iddfs":
        print(f"Starting Iterative Deepening Depth-First Search (IDDFS) with a maximum of {max_steps} steps... ")
        max_depth_list = range(max_steps + 1)

    else:
        raise ValueError(f"Error: Algorithm {algorithm} must be 'dfs' or 'iddfs'.")

    solution_found = False

    start_time = time.time()

    for maximum_depth in max_depth_list:

        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} --- MAXIMUM DEPTH: {maximum_depth} ---")

        for seq in dfs_enumerate(scenario=example_scenario, max_depth=maximum_depth, avoid_cycles=False):

            # print(f"Found: {seq}")

            total_time, total_energy, sequence_reliability, done, steps_used, _ = example_scenario.execute_action_idx_sequence(np.array(seq), check_validity=False, random_seed=None)

            if done:

                print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Successful path found [{steps_used}] -> time: {total_time:.2f} & energy: {total_energy:.2f} & reliability: {sequence_reliability:.2f}", end="")

                if (total_time + total_energy) < min_time_energy_cost:

                    if use_reliability:

                        min_time_energy_cost = total_time + total_energy + (1 - sequence_reliability)
                        min_time_energy_cost_sequence = seq
                        print(f"-> New minimum time & energy cost plus (1 - reliability): {min_time_energy_cost:.2f}")

                    else:

                        min_time_energy_cost = total_time + total_energy
                        min_time_energy_cost_sequence = seq
                        print(f"-> New minimum time & energy cost: {min_time_energy_cost:.2f}")

                else:
                    print("")

                # Keep track to stop at this depth
                solution_found = True

        if solution_found:
            break

    print(f"\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} DFS done after {time.time() - start_time:.2f} seconds")

    if use_reliability:

        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Minimum time & energy cost plus (1 - reliability): {min_time_energy_cost:.2f} -> {min_time_energy_cost_sequence}")

    else:

        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Minimum time & energy cost: {min_time_energy_cost:.2f} -> {min_time_energy_cost_sequence}")
