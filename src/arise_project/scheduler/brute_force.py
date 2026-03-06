# -*- coding: utf-8 -*-

"""
Module defining a brute force algorithm for finding the optimal solution (effectively impossible).
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

import numpy as np
from tqdm import tqdm

from src.arise_project.config.paths import DIR_DATA_INPUT_SCENARIOS_JSON_PATH
from src.arise_project.model.scenario import Scenario


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
    example_scenario = Scenario(file_path=DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_plate_factory.json")

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

        total_time, total_energy, sequence_reliability, done, steps_used, actions_taken = example_scenario.execute_action_idx_sequence(
            seq=seq, check_validity=True, random_seed=None)

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
