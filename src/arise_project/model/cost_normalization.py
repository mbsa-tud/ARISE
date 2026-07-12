# -*- coding: utf-8 -*-

"""
Module computing fixed, scenario-data-only reference scales for time and energy cost.

These scales exist to bring time_cost and energy_cost onto comparable orders of magnitude before
ObjectiveFunction applies its weights - without that, time_weight/energy_weight silently stop meaning
"relative importance" the moment machines have different execution speeds/power draws (e.g. energy in
Watt-seconds can be orders of magnitude larger than time in seconds).

Deliberately NOT derived from running any optimizer: that would be circular (the objective function
is needed to run the optimizer) and, since A*/Dijkstra accumulate cost additively while searching,
only a fixed constant divisor keeps the search's g(n) valid and admissible. Instead, this reuses the
same kind of cheap, admissible "best case, cheapest skill" estimate already used as the A* heuristic
(see scheduler/a_star_search.py:product_completion_heuristic) plus a rough transport allowance -
computed once from the scenario's machines/tasks/layout, before any search starts, so every scheduling
method (A*, DFS, GA, RL, LLM) can be compared against the same fixed scale.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from src.arise_project.model.execution_mode import ExecutionMode
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.model.skills import TransportSkill

_EPSILON = 1e-9


def compute_cost_scales(scenario_core: ScenarioCore) -> tuple[float, float, float]:
    """
    Compute (time_scale, energy_scale, reliability_scale) reference magnitudes for the given scenario,
    at its current (start) state.

    time_scale/energy_scale are strictly positive lower-bound-style estimates (cheapest skill per
    task, best case): the true cost of any schedule will typically land somewhere above them, since
    they ignore machine contention, retries from unreliable skills, and non-optimal routing - that's
    fine, they only need to serve as a consistent order-of-magnitude reference, not a strict [0, 1]
    ceiling.

    reliability_scale mirrors the same idea but the other way round: it's 1.0 minus a *worst*-case
    sequence reliability (worst applicable skill per task/hop), i.e. a realistic-bad-case magnitude for
    (1 - sequence_reliability). Real per-skill reliabilities cluster near 1.0, so raw (1 - reliability)
    is typically tiny (e.g. 0.005) - without this scale, reliability_weight would end up contributing
    almost nothing to the objective/reward compared to the normalized time/energy terms, regardless of
    how it's weighted.

    :param scenario_core: scenario to compute reference scales for (ScenarioCore)
    :return: (time_scale, energy_scale, reliability_scale) (tuple[float, float, float])
    """

    time_lower_bound = 0.0
    energy_lower_bound = 0.0
    worst_case_sequence_reliability = 1.0
    n_processing_tasks = 0

    # Cheapest skill/machine combination per remaining processing task, best case (no variability),
    # and separately the worst applicable skill's reliability for that same task
    for product in scenario_core.get_sorted_product_list():

        for processing_task in product.get_remaining_processing_tasks():

            n_processing_tasks += 1

            best_time = float("inf")
            best_energy = float("inf")
            worst_reliability = 1.0

            for skill_type in processing_task.possible_skill_types:

                for machine in scenario_core.factory.get_machines_by_skill_type(skill_type):

                    skill = machine.get_skill_by_type(skill_type)

                    task_time, task_energy, _ = processing_task.execute(skill, mode=ExecutionMode.BEST_CASE)

                    if task_time < best_time:
                        best_time = task_time
                        best_energy = task_energy

                    if skill.reliability < worst_reliability:
                        worst_reliability = skill.reliability

            if best_time < float("inf"):
                time_lower_bound += best_time
                energy_lower_bound += best_energy
                worst_case_sequence_reliability *= worst_reliability

    # Rough transport allowance: average best-case cost of the factory's actual transport edges,
    # times roughly one hop per processing task (plus one for the initial/final trip), and the least
    # reliable transport skill applied over that same number of hops
    transport_tasks = list(scenario_core.factory.transport_task_by_id_dict.values())
    transport_machines = scenario_core.factory.get_machines_by_skill_type(TransportSkill)

    if transport_tasks and transport_machines:

        total_transport_time = 0.0
        total_transport_energy = 0.0
        worst_transport_reliability = 1.0

        for transport_task in transport_tasks:

            best_time = float("inf")
            best_energy = float("inf")

            for machine in transport_machines:

                skill = machine.get_skill_by_type(TransportSkill)
                task_time, task_energy, _ = transport_task.execute(skill, mode=ExecutionMode.BEST_CASE)

                if task_time < best_time:
                    best_time = task_time
                    best_energy = task_energy

                if skill.reliability < worst_transport_reliability:
                    worst_transport_reliability = skill.reliability

            total_transport_time += best_time
            total_transport_energy += best_energy

        avg_transport_time = total_transport_time / len(transport_tasks)
        avg_transport_energy = total_transport_energy / len(transport_tasks)

        n_hops = n_processing_tasks + 1

        time_lower_bound += n_hops * avg_transport_time
        energy_lower_bound += n_hops * avg_transport_energy
        worst_case_sequence_reliability *= worst_transport_reliability ** n_hops

    reliability_scale = 1.0 - worst_case_sequence_reliability

    return (max(time_lower_bound, _EPSILON),
            max(energy_lower_bound, _EPSILON),
            max(reliability_scale, _EPSILON))
