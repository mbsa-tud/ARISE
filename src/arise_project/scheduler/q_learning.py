# -*- coding: utf-8 -*-

"""
Module containing the Q-learning (reinforcement learning) algorithm.
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from typing import Dict, List
from collections import defaultdict
import numpy as np

from src.arise_project.model.action_key import ActionKey
from src.arise_project.model.product_state import ProductState
from src.arise_project.model.scenario import Scenario


class QLearningAlgorithm:

    def __init__(self,
                 scenario: Scenario,
                 num_episodes: int = 5000,
                 gamma_discount_factor: float = 0.99,
                 learning_rate: float = 0.2,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay_episodes: int = 3000,
                 alpha_weight: float = 1.0,
                 beta_weight: float = 1.0,
                 base_seed: int = 123) -> None:

        self._scenario = scenario
        self._num_episodes = num_episodes
        self._gamma_discount_factor = gamma_discount_factor
        self._learning_rate = learning_rate
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_episodes = epsilon_decay_episodes
        self._alpha_weight = alpha_weight
        self._beta_weight = beta_weight
        self._base_seed = base_seed

        self._rng = np.random.default_rng(self._base_seed)
        self._Q: Dict[tuple, float] = defaultdict(float)
        self._returns_hist: List[float] = []

    def train(self):
        """
        Trains a tabular Q-learning agent.
        """

        self._rng = np.random.default_rng(self._base_seed)
        self._Q: Dict[tuple, float] = defaultdict(float)
        self._returns_hist: List[float] = []

        def choose_action_epsilon_greedy(s_repr: ProductState, action_key_list: List[ActionKey], epsilon: float) -> int:

            if not action_key_list:
                return -1

            if self._rng.random() < epsilon:
                return int(self._rng.integers(0, len(action_key_list)))

            # Greedy: pick action with largest Q[(state, action)]
            best_i, best_q = 0, -float("inf")

            for i, ak in enumerate(action_key_list):

                q = self._Q[(s_repr, ak)]

                if q > best_q:
                    best_q, best_i = q, i

            return best_i

        epsilon = self._epsilon_start

        # Repeat for each episode
        for episode_idx in range(self._num_episodes):

            self._scenario.reset(random_seed=self._base_seed + episode_idx)
            is_done_bool = self._scenario.is_done()
            state_current = self._scenario.get_current_state()

            total_return = 0.0

            while not is_done_bool:

                actions = self._scenario.get_action_identities()

                # Dead-end: terminate this episode
                if not actions:
                    break

                action_idx = choose_action_epsilon_greedy(state_current, actions, epsilon)

                if action_idx < 0:
                    break

                action = actions[action_idx]

                # Execute and get outcome (time, energy)
                outcome = self._scenario.execute_action_key(action)
                step_cost = (self._alpha_weight * float(outcome.total_time)
                             + self._beta_weight * float(outcome.total_energy))

                r = -step_cost

                is_done_bool = self._scenario.is_done()
                state_next = self._scenario.get_current_state()

                # Bootstrap only over next feasible actions
                if is_done_bool:
                    best_next = 0.0

                else:
                    next_actions = self._scenario.get_action_identities()

                    if next_actions:
                        best_next = max(self._Q[(state_next, ak_next)] for ak_next in next_actions)
                    else:
                        best_next = 0.0

                # Q update
                key = (state_current, action)
                self._Q[key] = ((1 - self._learning_rate) * self._Q[key] +
                                self._learning_rate * (r + self._gamma_discount_factor * best_next))

                state_current = state_next
                total_return += r

            # Epsilon linear schedule
            frac = min(1.0, (episode_idx + 1) / max(1, self._epsilon_decay_episodes))
            epsilon = self._epsilon_start + frac * (self._epsilon_end - self._epsilon_start)

            self._returns_hist.append(total_return)

            if (episode_idx + 1) % 500 == 0:
                print(f"[Q] Episode {episode_idx+1}/{self._num_episodes} | avg return(100)={np.mean(self._returns_hist[-100:]):.3f}")

    def greedy_action_from_current_state(self) -> ActionKey | None:
        """
        Select the greedy action from the current state.
        :return: greedy action (ActionKey or None)
        """

        state_current = self._scenario.get_current_state()
        actions = self._scenario.get_action_identities()

        if not actions:
            return None

        return max(actions, key=lambda ak: self._Q[(state_current, ak)])
