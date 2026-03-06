# -*- coding: utf-8 -*-

"""
Module defining the genetic algorithms, specifically Non-dominated Sorting Genetic Algorithms (NSGA-II / NSGA-III).
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV

from src.arise_project.config.paths import DIR_DATA_INPUT_SCENARIOS_JSON_PATH
from src.arise_project.model.scenario import Scenario


class FactorySequenceProblem(Problem):
    """
    Each individual is a fixed-length sequence of action indices.

    Objectives:
        F1: total_time (+ penalty if not finished)
        F2: total_energy (+ penalty if not finished)
        F3: sequence_reliability (optional)

    Constraint:
        G[0] = 0 if finished, else 1  (feasibility: must complete all products)
    """

    def __init__(self, scenario: Scenario, seq_len=200, penalty=1e4, use_reliability: bool = False):

        self._scenario = scenario
        num_actions = len(self._scenario.sorted_action_catalog)

        self._use_reliability = use_reliability

        if use_reliability:
            self._n_objectives = 3
        else:
            self._n_objectives = 2

        super().__init__(n_var=seq_len, n_obj=self._n_objectives, n_constr=1, xl=np.zeros(seq_len, dtype=int),
                         xu=np.full(seq_len, num_actions - 1, dtype=int), type_var=int)

        self._penalty = float(penalty)

    def _evaluate(self, X, out, *args, **kwargs):

        F = np.zeros((len(X), self._n_objectives), dtype=float)
        G = np.zeros((len(X), 1), dtype=float)

        # Evaluate individuals sequentially to avoid scenario state conflicts
        for i, seq in enumerate(X):

            total_time, total_energy, sequence_reliability, done, steps_used, _ = self._scenario.execute_action_idx_sequence(seq)

            if not done:

                # TODO Check if penalty is needed, if not remove
                total_time += self._penalty
                total_energy += self._penalty
                G[i, 0] = 1.0  # violated: not finished

            else:
                G[i, 0] = 0.0

            F[i, 0] = total_time
            F[i, 1] = total_energy

            if self._use_reliability:

                # Minimize (1 - reliability) -> alternative, maximize reliability
                F[i, 2] = 1.0 - sequence_reliability

        out["F"] = F
        out["G"] = G


class FeasibleSequenceSampling(Sampling):
    """
    Create initial sequences by following only feasible actions (via action_mask).
    After termination, pad sequence to fixed length with random integers (unused once done).
    """

    def __init__(self, scenario: Scenario, seq_len: int):
        super().__init__()
        self._scenario = scenario
        self._seq_len = seq_len

    def _do(self, problem, n_samples, **kwargs):

        samples = np.zeros((n_samples, self._seq_len), dtype=int)

        for i in range(n_samples):

            self._scenario.reset()
            seq = []

            for _ in range(self._seq_len):

                # Create the action mask of all feasible actions in the current state
                action_mask = self._scenario.generate_feasible_action_mask()
                feasible_action_idx_array = np.flatnonzero(action_mask)

                # Stop in case there are no more feasible actions in current state
                if feasible_action_idx_array.size == 0:
                    break

                action_idx = int(np.random.choice(feasible_action_idx_array))
                seq.append(action_idx)

                # Execute the action by its index in the action catalog
                task_result, product_done, all_products_done = self._scenario.step_by_action_idx(action_idx)

                # This shouldn't happen due to masking, but nevertheless handle it
                if task_result is None:
                    break

                if all_products_done:
                    break

            # Pad with random actions (not executed after done)
            if len(seq) < self._seq_len:
                pad = np.random.randint(0, len(self._scenario.sorted_action_catalog), size=self._seq_len - len(seq))
                seq = np.concatenate([np.array(seq, dtype=int), pad])

            samples[i, :] = seq

        return samples


class FeasibilityRepair(Repair):
    """
    Simulate sequence; whenever an action is infeasible at a step,
    replace it with a random feasible action. Stop repairing once episode done.
    """

    def __init__(self, scenario: Scenario):
        super().__init__()
        self._scenario = scenario

    def _do(self, problem, X, **kwargs):

        Y = np.copy(X)

        for i in range(Y.shape[0]):

            self._scenario.reset()

            for j in range(Y.shape[1]):

                # Create the action mask of all feasible actions in the current state
                action_mask = self._scenario.generate_feasible_action_mask()
                feasible_action_idx_array = np.flatnonzero(action_mask)

                # Stop in case there are no more feasible actions in current state (leave remainder as it is)
                if feasible_action_idx_array.size == 0:
                    break

                action_idx = int(Y[i, j])

                if action_mask[action_idx] == 0:

                    # Repair: choose a random feasible action for this step
                    action_idx = int(np.random.choice(feasible_action_idx_array))
                    Y[i, j] = action_idx

                # Execute the action by its index in the action catalog
                task_result, product_done, all_products_done = self._scenario.step_by_action_idx(action_idx)

                # This shouldn't happen due to masking, but nevertheless handle it
                if task_result is None:
                    break

                if all_products_done:
                    break

        return Y


class RandomResetMutation(Mutation):
    """
    Discrete mutation: with probability `prob` for each gene, reset to a random valid action index.
    """

    def __init__(self, prob=0.1):
        super().__init__()
        self._prob = float(prob)

    def _do(self, problem, X, **kwargs):

        Y = np.copy(X)

        # Ensure per-variable bounds are arrays
        xl = np.array(problem.xl).astype(int)
        xu = np.array(problem.xu).astype(int)

        for i in range(Y.shape[0]):

            for j in range(Y.shape[1]):

                if np.random.rand() < self._prob:
                    Y[i, j] = np.random.randint(xl[j], xu[j] + 1)

        return Y


def run_nsga(scenario: Scenario, algorithm: str = "nsga2", seq_len: int = 200, pop_size: int = 80, n_gen: int = 150,
             penalty: float = 1e4, mut_prob: float = 0.15, seed: int = 42, use_reliability: bool = False, verbose: bool = True):
    """
    Run NSGA-II or NSGA-III on the FactorySequenceProblem.
    Returns pymoo result object.
    """

    # Define problem to minimize
    problem = FactorySequenceProblem(scenario=scenario, seq_len=seq_len, penalty=penalty, use_reliability=use_reliability)

    # Create initial population (random sequence of feasible actions)
    sampling = FeasibleSequenceSampling(scenario=scenario, seq_len=seq_len)

    # Define repair method
    repair = FeasibilityRepair(scenario=scenario)

    # Define crossover method
    crossover = TwoPointCrossover()
    crossover.repair = repair

    # Define mutation method
    mutation = RandomResetMutation(prob=mut_prob)
    mutation.repair = repair

    if algorithm.lower() == "nsga2":

        algo = NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                     mutation=mutation, eliminate_duplicates=True)

    elif algorithm.lower() == "nsga3":

        # NSGA-III with reference directions (2 or 3 objectives, though NSGA-II is typically preferred for 2D)

        if use_reliability:
            # <!> Caution, using three dimensions changes the population to 78 to avoid exceptions
            ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=11)
            pop_size = len(ref_dirs)
        else:
            ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_points=pop_size)

        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, crossover=crossover,
                     mutation=mutation, eliminate_duplicates=True)

    else:
        raise ValueError("algorithm must be 'nsga2' or 'nsga3'")

    termination = get_termination("n_gen", n_gen)

    result = minimize(problem=problem, algorithm=algo, termination=termination, seed=seed,
                      save_history=False, verbose=verbose)

    return result


def plot_pareto(results: Dict[str, Any], title: str = "Pareto Fronts"):
    """
    Plot Pareto fronts for multiple algorithms.
    results: dict like {"nsga2": res2, "nsga3": res3}
    """

    plt.figure(figsize=(8, 6))
    colors = {"nsga2": "#1f77b4", "nsga3": "#ff7f0e"}

    for name, res in results.items():

        if res is None:
            continue

        F = res.F

        plt.scatter(F[:, 0], F[:, 1], s=40, alpha=0.7, label=f"{name.upper()} ({len(F)} pts)", c=colors.get(name, None))

    plt.title(title)
    plt.xlabel("Total Time")
    plt.ylabel("Total Energy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_hypervolume(res, reference_point: Optional[Tuple[float, float]] = None) -> float:
    """
    Compute hypervolume for a 2D Pareto front.
    If reference_point is None, use slightly worse than max objectives in res.F.
    """

    F = res.F

    # F is the objective matrix
    if F.shape[1] > 2:
        F2 = F[:, :2]  # take only first two columns (time and energy)
    else:
        F2 = F  # keep as is

    if reference_point is None:
        ref_time = np.max(F[:, 0]) * 1.05
        ref_energy = np.max(F[:, 1]) * 1.05
        reference_point = (ref_time, ref_energy)

    hv = HV(ref_point=np.array(reference_point))

    return float(hv(F2))


def print_best_sequence(scenario: Scenario, res, prefer: str = "sum"):
    """
    Print action keys for one representative solution.
    prefer: 'sum' (min sum of objectives) or 'time' or 'energy'
    """

    F = res.F

    if prefer == "time":
        best_idx = int(np.argmin(F[:, 0]))
    elif prefer == "energy":
        best_idx = int(np.argmin(F[:, 1]))
    else:
        best_idx = int(np.argmin(np.sum(F, axis=1)))

    best_seq = res.X[best_idx]

    # Re-simulate to get actual actions taken until done
    _, _, _, _, _, actions_taken = scenario.execute_action_idx_sequence(best_seq)

    print(f"\nChosen solution index: {best_idx}")
    print(f"Objectives (time, energy): {F[best_idx]}")
    print(f"Actions used until completion ({len(actions_taken)} steps):")

    for action_idx in actions_taken:
        print(scenario.sorted_action_catalog[action_idx])


if __name__ == "__main__":

    # Load a scenario (product and factory)
    example_scenario = Scenario(file_path=DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_plate_factory_b.json")

    # Fixed-length chromosome
    max_steps = 200
    population_size = 80
    number_generations = 150

    # Start timer for NSGA-II
    nsga2_start_time = time.time()

    # Run NSGA-II
    res_nsga2 = run_nsga(
        scenario=example_scenario,
        algorithm="nsga2",
        seq_len=max_steps,
        pop_size=population_size,
        n_gen=number_generations,
        mut_prob=0.15,
        penalty=1e4,
        seed=42,
        use_reliability=True,
        verbose=True
    )

    # Calculate total time for NSGA-II
    nsga2_total_time = time.time() - nsga2_start_time

    # Start timer for NSGA-III
    nsga3_start_time = time.time()

    # Run NSGA-III - optional / for comparison (useful even in 2D)
    res_nsga3 = run_nsga(
        scenario=example_scenario,
        algorithm="nsga3",
        seq_len=max_steps,
        pop_size=population_size,
        n_gen=number_generations,
        mut_prob=0.15,
        penalty=1e4,
        seed=42,
        use_reliability=True,
        verbose=True
    )

    # Calculate total time for NSGA-III
    nsga3_total_time = time.time() - nsga3_start_time

    # Visualize both fronts
    results = {"nsga2": res_nsga2, "nsga3": res_nsga3}
    plot_pareto(results, title="Factory Scheduling: Time vs Energy Pareto Fronts")

    # Hypervolume comparison (bigger is better)
    hv_nsga2 = compute_hypervolume(res_nsga2)
    hv_nsga3 = compute_hypervolume(res_nsga3)

    print(f"Hypervolume NSGA-II: {hv_nsga2:.2f} in {nsga2_total_time:.2f} seconds")
    print(f"Hypervolume NSGA-III: {hv_nsga3:.2f} in {nsga3_total_time:.2f} seconds")

    # Print best sequence of actions
    print_best_sequence(example_scenario, res_nsga2, prefer="sum")
