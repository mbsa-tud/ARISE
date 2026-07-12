# -*- coding: utf-8 -*-

"""
Module containing the factory environment for reinforcement learning.
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import gymnasium as gym
import numpy as np
from typing import Any

from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.model.tasks import ProcessingTask

# Reward constants, rescaled to match the normalized (roughly O(1) per completed schedule) objective
# function cost instead of the raw, unnormalized time/energy costs they were originally tuned against
INVALID_ACTION_PENALTY = -1.0
PROCESSING_TASK_BONUS = 2.5
PRODUCT_DONE_BONUS = 6.0
ALL_PRODUCTS_DONE_BONUS = 12.0


class FactoryEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, scenario: ScenarioCore, objective_function: ObjectiveFunction, max_steps: int = 200,
                 seed: int = 0, output_action_state: bool = False):

        super().__init__()

        self.scenario = scenario
        self.objective_function = objective_function
        self.rng = np.random.RandomState(seed)
        self._max_steps = max_steps
        self.output_action_state = output_action_state

        self._step_count = 0
        self._prev_cost = 0.0

        # Catalogs and fixed sizes
        self.products = self.scenario.get_sorted_product_list()
        self.processing_tasks = self.scenario.get_sorted_processing_tasks_list()
        self.stationary_machines = self.scenario.factory.get_sorted_all_stationary_machines_list()

        self.prod_index = {p: i for i, p in enumerate(self.products)}
        self.task_index = {t: i for i, t in enumerate(self.processing_tasks)}
        self.stat_mach_index = {m: i for i, m in enumerate(self.stationary_machines)}

        # Observation: features (float vector) + action_mask (binary)
        self.per_product_dim = len(self.processing_tasks) + len(self.stationary_machines)
        self.obs_dim = len(self.products) * self.per_product_dim

        self.observation_space = gym.spaces.Dict({
            "features": gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32),
            "action_mask": gym.spaces.MultiBinary(len(self.scenario.sorted_action_catalog))
        })

        self.action_space = gym.spaces.Discrete(len(self.scenario.sorted_action_catalog))

    def _encode_state(self) -> dict[str, np.ndarray]:
        """
        Build normalized features and binary action mask from the scenario's current state.
        """

        # Initialize features array
        features = np.zeros((len(self.products), self.per_product_dim), dtype=np.float32)

        # # Get state of products
        # product_states_dict = self.scenario.get_product_states()
        #
        # for product, p_idx in list(self.prod_index.items()):
        #
        #     if product.unique_id not in product_states_dict:
        #         continue
        #
        #     product_state = product_states_dict[product.unique_id]
        #
        #     # Get completed tasks and mark them in features array
        #     for t in product_state.get_ordered_processing_task_list():
        #         if t in self.task_index:
        #             features[p_idx, self.task_index[t]] = 1.0
        #
        #     # Get current location of product
        #     stationary_machine_loc = self.scenario.factory.stationary_machine_by_id_dict[product_state.location_machine_id]
        #
        #     # Mark machine in features array
        #     if stationary_machine_loc in self.stat_mach_index:
        #
        #         loc_idx = self.stat_mach_index[stationary_machine_loc]
        #         features[p_idx, len(self.task_index) + loc_idx] = 1.0
        #
        # # Flatten array
        features = features.reshape(-1)

        # Create the action mask of all feasible actions in the current state
        action_mask = self.scenario.generate_feasible_action_mask()

        return {"features": features, "action_mask": action_mask}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):

        if seed is not None:
            self.rng.seed(seed)

        self._step_count = 0
        self._prev_cost = 0.0
        self.scenario.reset()

        obs = self._encode_state()
        info = {}

        return obs, info

    def step(self, action_idx: int):

        self._step_count += 1

        truncated = False

        if self._step_count >= self._max_steps:
            truncated = True

        if self.output_action_state:
            print(f"{self._step_count} -> {self.scenario.sorted_action_catalog[action_idx]} -> {self.scenario.get_product_states()["PL01"]}")

        # Validate action via mask
        obs = self._encode_state()
        action_mask = obs["action_mask"]

        if action_mask[action_idx] == 0:

            return obs, INVALID_ACTION_PENALTY, False, truncated, {"invalid_action": True}

        # Execute the action by its index in the action catalog
        task_result, product_done, all_products_done = self.scenario.step_by_action_idx(action_idx)

        # This shouldn't happen due to masking, but nevertheless handle it
        if task_result is None:

            return obs, INVALID_ACTION_PENALTY, False, truncated, {"invalid_action": True}

        # Reward is the negative *change* in the same (normalized) objective cost used by every other
        # scheduling method, so summing rewards over an episode telescopes to -total_cost of the final
        # schedule. Uses the scenario's running cumulative sums, not this single action's raw values,
        # so reliability correctly reflects the compounding sequence reliability, not just this skill's.
        current_cost = self.objective_function(time_cost=self.scenario.time_sum,
                                               energy_cost=self.scenario.energy_sum,
                                               reliability=self.scenario.sequence_reliability)

        reward = -(current_cost - self._prev_cost)
        self._prev_cost = current_cost

        if isinstance(task_result.task, ProcessingTask):
            reward += PROCESSING_TASK_BONUS

        # Add completion bonus for one product
        if product_done:
            reward += PRODUCT_DONE_BONUS

        # Add completion bonus for all products
        if all_products_done:
            reward += ALL_PRODUCTS_DONE_BONUS

        next_obs = self._encode_state()

        info = {"time_cost": task_result.total_time, "energy_cost": task_result.total_energy,
               "reliability": task_result.skill.reliability}

        return next_obs, reward, all_products_done, truncated, info

    def render(self):
        pass
