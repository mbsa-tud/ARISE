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
"""

from pathlib import Path

import numpy as np
import pytest

from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.scheduler.factory_gym_env import FactoryEnv

TINY_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scenario.json"

# Note: gymnasium's strict check_env() fails on an unrelated pre-existing quirk (reset() does not
# call super().reset(seed=...)), so this file checks the properties that matter for RL training instead.


def _make_env() -> FactoryEnv:
    scenario = ScenarioCore(file_path=TINY_SCENARIO_PATH, reset_class=True)
    objective_function = ObjectiveFunction(time_weight=1 / 3, energy_weight=1 / 3, reliability_weight=1 / 3)
    return FactoryEnv(scenario=scenario, objective_function=objective_function)


def test_spaces_match_action_catalog_size():

    env = _make_env()

    assert env.action_space.n == len(env.scenario.sorted_action_catalog) == 3
    assert env.observation_space["action_mask"].n == 3


def test_reset_action_mask_matches_scenario_feasibility():

    env = _make_env()
    obs, info = env.reset()

    assert obs["action_mask"].tolist() == [0, 1, 0]
    assert (obs["action_mask"] == env.scenario.generate_feasible_action_mask()).all()


def test_invalid_action_is_penalized_without_changing_scenario_state():

    env = _make_env()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert reward == pytest.approx(-1.0)
    assert terminated is False
    assert info["invalid_action"] is True
    assert env.scenario.step_count == 0


def test_golden_path_reaches_termination_with_expected_rewards():
    """
    Reward is -(change in normalized objective cost) plus fixed completion bonuses, so summing the
    cost component alone over an episode telescopes to -total_cost of the final schedule.
    """

    env = _make_env()
    env.reset()

    _, reward, terminated, _, _ = env.step(1)
    assert reward == pytest.approx(-2 / 3)
    assert terminated is False

    _, reward, terminated, _, _ = env.step(0)
    assert reward == pytest.approx(7 / 6)
    assert terminated is False

    _, reward, terminated, _, _ = env.step(2)
    assert reward == pytest.approx(52 / 3)
    assert terminated is True