# -*- coding: utf-8 -*-

"""
Module containing the functions used to train DQN and for inference.
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from src.arise_project.config.paths import DIR_DATA_INPUT_SCENARIOS_JSON_PATH, DIR_DATA_OUTPUT_DQN_MODELS_PATH
from src.arise_project.model.scenario import Scenario
from src.arise_project.scheduler.factory_gym_env import FactoryEnv


def run_train():

    # Load a scenario (product and factory)
    scenario = Scenario(file_path=DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_plate_factory_b.json")
    train_env = FactoryEnv(scenario, alpha=1.0, beta=1.0, max_steps=200, seed=0, use_reliability=True)

    check_env(train_env, warn=True)
    train_env = Monitor(train_env)

    # Reset unique ID counters
    Scenario.reset_all()

    # Evaluation environment (new scenario clone)
    eval_scenario = Scenario(file_path=DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_plate_factory_b.json")
    eval_env = FactoryEnv(eval_scenario, alpha=1.0, beta=1.0, max_steps=200, seed=123, use_reliability=True)

    check_env(eval_env, warn=True)
    eval_env = Monitor(eval_env)

    # Define log output (for tensorboard)
    logs_path = DIR_DATA_OUTPUT_DQN_MODELS_PATH / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    # For dict observation: MultiInputPolicy
    model = DQN(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        target_update_interval=1_000,
        train_freq=16,
        gradient_steps=8,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
        tensorboard_log=logs_path,
        verbose=1,
    )

    # Periodic evaluation
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=logs_path,
        log_path=logs_path,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    start_time = time.time()

    # Start training the model
    model.learn(total_timesteps=250_000, callback=eval_cb)

    print(f"Training time -> {time.time() - start_time:.5f} seconds")

    model.save(DIR_DATA_OUTPUT_DQN_MODELS_PATH / "factory_dqn_final")

    print("Done.")


def run_inference(count: int = 1, quick_eval: bool = False):

    scenario = Scenario(file_path=DIR_DATA_INPUT_SCENARIOS_JSON_PATH / "scenario_plate_factory_b.json")

    original_env = FactoryEnv(scenario, alpha=1.0, beta=1.0, max_steps=200, seed=999, use_reliability=True, output_action_state=False)
    env = Monitor(original_env)

    input_final_filepath = DIR_DATA_OUTPUT_DQN_MODELS_PATH / "dqn_factory_final"  # "factory_dqn_final_DQN_2_20251111_perfect"
    input_best_filepath = DIR_DATA_OUTPUT_DQN_MODELS_PATH / "logs" / "factory_dqn_best"  # "factory_dqn_best_DQN_2_20251111_great.zip"

    model = DQN.load(input_final_filepath, env=env)

    if quick_eval:

        # Quick evaluation
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
        print(f"\n--> DQN evaluation - Mean reward: {mean_r:.3f} ± {std_r:.3f}\n")

    shortest_done_length = 999

    for i in range(count):

        print(f"--- Round {i+1} ---")

        # Manual rollout (deterministic)
        obs, info = env.reset(seed=2025)
        done = False
        truncated = False

        counter = 0

        start_time = time.time()

        while not (done or truncated):

            counter += 1

            # Obs is a dictionary, action is an integer (index of action catalog)
            action_idx, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action_idx)

            print(f"{counter}. {original_env.scenario.sorted_action_catalog[action_idx]} -> {original_env.scenario.get_product_states()}")

        print(f"Time {original_env.scenario.time_sum:.2f} & Energy {original_env.scenario.energy_sum:.2f} -> {time.time() - start_time:.5f} seconds")
        print("")

        shortest_done_length = min(shortest_done_length, counter)

    print(f"Shortest sequence until done: {shortest_done_length}")
