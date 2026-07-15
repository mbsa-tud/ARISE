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

Module defining a brute force algorithm for finding the optimal solution (effectively impossible).
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import numpy as np
from tqdm import tqdm

from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.model.scenario import ScenarioCore


def next_seq(seq: np.ndarray, base: int) -> bool:
    """
    Increment sequence in place treating seq[0] as the least significant 'digit'.
    Returns False if overflowed past the last position (i.e., finished all combos),
    otherwise True.
    """

    i = 0

    while i < len(seq):

        if seq[i] + 1 < base:
            seq[i] += 1
            return True

        else:
            seq[i] = 0
            i += 1

    # Overflowed the highest index
    return False


def main():

    # Load a scenario (product and factory)
    example_scenario = ScenarioCore(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)

    sequence_length = 10
    base = len(example_scenario.sorted_action_catalog)
    total_combinations = base ** sequence_length

    seq = np.zeros(sequence_length, dtype=int)

    # Debugging / Testing
    actions_done = []
    counter = 0

    # Iterate with tqdm using for loop
    for _ in tqdm(range(total_combinations)):

        # print(f"{counter} - {seq}")

        done, steps_used, actions_taken = example_scenario.execute_action_idx_sequence(
            seq=seq, check_validity=True, random_seed=None)

        total_time = example_scenario.time_sum
        total_energy = example_scenario.energy_sum
        sequence_reliability = example_scenario.sequence_reliability

        if done:
            actions_done.append(actions_taken)

        counter += 1

        if not next_seq(seq, base):
            break

        if counter > 100000:
            break

    print(f"Actions done: {actions_done}")


if __name__ == '__main__':
    main()
