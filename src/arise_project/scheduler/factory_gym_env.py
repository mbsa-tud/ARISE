# -*- coding: utf-8 -*-

"""
Module containing the factory environment for reinforcement learning.
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

import gymnasium as gym
import numpy as np
from typing import Dict, Any

from src.arise_project.model.scenario import Scenario
from src.arise_project.model.tasks import ProcessingTask


class FactoryEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, scenario: Scenario, alpha: float = 1.0, beta: float = 1.0, max_steps: int = 200,
                 seed: int = 0, use_reliability: bool = False, output_action_state: bool = False):

        super().__init__()

        self.scenario = scenario
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.RandomState(seed)
        self._max_steps = max_steps
        self._use_reliability = use_reliability
        self.output_action_state = output_action_state

        self._step_count = 0

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

    def _encode_state(self) -> Dict[str, np.ndarray]:
        """
        Build normalized features and binary action mask from the scenario's current state.
        """

        # Initialize features array
        features = np.zeros((len(self.products), self.per_product_dim), dtype=np.float32)

        # Get state of products
        product_states_dict = self.scenario.get_product_states()

        for product, p_idx in list(self.prod_index.items()):

            if product.unique_id not in product_states_dict:
                continue

            product_state = product_states_dict[product.unique_id]

            # Get completed tasks and mark them in features array
            for t in product_state.get_ordered_processing_task_list():
                if t in self.task_index:
                    features[p_idx, self.task_index[t]] = 1.0

            # Get current location of product
            stationary_machine_loc = self.scenario.factory.stationary_machine_by_id_dict[product_state.location_machine_id]

            # Mark machine in features array
            if stationary_machine_loc in self.stat_mach_index:

                loc_idx = self.stat_mach_index[stationary_machine_loc]
                features[p_idx, len(self.task_index) + loc_idx] = 1.0

        # Flatten array
        features = features.reshape(-1)

        # Create the action mask of all feasible actions in the current state
        action_mask = self.scenario.generate_feasible_action_mask()

        return {"features": features, "action_mask": action_mask}

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):

        if seed is not None:
            self.rng.seed(seed)

        self._step_count = 0
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

            # Strong penalty for invalid action
            penalty = -15.0

            return obs, penalty, False, truncated, {"invalid_action": True}

        # Execute the action by its index in the action catalog
        task_result, product_done, all_products_done = self.scenario.step_by_action_idx(action_idx)

        # This shouldn't happen due to masking, but nevertheless handle it
        if task_result is None:

            # Strong penalty for invalid action
            penalty = -15.0

            return obs, penalty, False, truncated, {"invalid_action": True}

        # Reward is based on the negative weighted cost
        time_cost = task_result.total_time
        energy_cost = task_result.total_energy
        reliability = task_result.skill.reliability

        reward = - ((self.alpha * time_cost) + (self.beta * energy_cost))

        if self._use_reliability:
            reward -= 100.0 * (1 - reliability)

        if isinstance(task_result.task, ProcessingTask):
            reward += 75.0

        # Add completion bonus for one product
        if product_done:
            reward += 300.0

        # Add completion bonus for all products
        if all_products_done:
            reward += 1000.0

        next_obs = self._encode_state()

        info = {"time_cost": time_cost, "energy_cost": energy_cost, "reliability": reliability}

        return next_obs, reward, all_products_done, truncated, info

    def render(self):
        pass
