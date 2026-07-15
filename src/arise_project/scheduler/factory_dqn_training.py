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

Module containing the functions used to train DQN and for inference.
Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import json
import time
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from jsonschema import validate, ValidationError

from src.arise_project.gui.custom.pyqt_progress_updater import DummyProgressUpdater
from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.cost_normalization import compute_cost_scales
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.optimization_result import OptimizationResult
from src.arise_project.tools.duration_format import duration_formatting
from src.arise_project.tools.hash_generation import get_scenario_data_dir_path
from src.arise_project.config.paths import DIR_NAME_LOGS, FILE_NAME_DQN_CONFIG, FILE_NAME_BEST_MODEL, \
    FILE_NAME_FINAL_MODEL, DIR_NAME_DQN, DIR_NAME_MODELS, DIR_NAME_BACKUP

from src.arise_project.tools.hash_generation import get_canonical_hash_dqn_config_json
from src.arise_project.config.paths import FILE_DQN_CONFIG_JSON_PATH
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.scheduler.factory_gym_env import FactoryEnv


OPT_RES_PARAM_REWARD = "reward"


def get_dqn_model_dir_path(scenario_file_path: Path, output_dir_path: Path) -> Path:

    dqn_file_path = scenario_file_path.parent / FILE_NAME_DQN_CONFIG
    hash_str = get_canonical_hash_dqn_config_json(dqn_file_path)

    dqn_model_dir_path = output_dir_path / DIR_NAME_DQN / f"{dqn_file_path.stem}_{hash_str}"

    return dqn_model_dir_path


def load_dqn_config(scenario_file_path: Path) -> dict:
    """
    Loads and validates the DQN configuration JSON file that is expected to exist within the same directory
    as the scenario file.
    """

    dqn_config_file_path = scenario_file_path.parent / FILE_NAME_DQN_CONFIG

    if not dqn_config_file_path.exists():
        raise FileNotFoundError(f"The file '{dqn_config_file_path.name}' needs to be in the same directory as the scenario file.")

    with open(dqn_config_file_path, 'r') as json_file:
        dqn_config_dict = json.load(json_file)

    with open(FILE_DQN_CONFIG_JSON_PATH, 'r') as schema_file:
        schema = json.load(schema_file)

    # Validate the JSON data against the schema
    try:
        validate(instance=dqn_config_dict, schema=schema)

    except ValidationError as e:
        print("Unable to load DQN configuration due to JSON validation error:", e.message)

    return dqn_config_dict


def run_training(scenario_file_path: Path) -> float:

    # Load the DQN training configuration
    dqn_config_dict = load_dqn_config(scenario_file_path=scenario_file_path)

    # Load a scenario (product and factory)
    scenario = ScenarioCore(file_path=scenario_file_path)

    # Time step budget: scaled linearly with problem size via "timesteps_per_processing_task"
    # (matching the linear growth of the action catalog), or fixed via "total_timesteps"
    timesteps_per_task = dqn_config_dict["training"].get("timesteps_per_processing_task")

    if timesteps_per_task is not None:
        total_timesteps = timesteps_per_task * len(scenario.get_sorted_processing_tasks_list())
        print(f"Time step budget: {timesteps_per_task} x "
              f"{len(scenario.get_sorted_processing_tasks_list())} processing tasks = {total_timesteps}")
    else:
        total_timesteps = dqn_config_dict["training"]["total_timesteps"]

    time_scale, energy_scale, reliability_scale = compute_cost_scales(scenario)
    objective_function = ObjectiveFunction(time_weight=dqn_config_dict["environment"]["time_weight"],
                                           energy_weight=dqn_config_dict["environment"]["energy_weight"],
                                           reliability_weight=dqn_config_dict["environment"]["reliability_weight"],
                                           time_scale=time_scale, energy_scale=energy_scale,
                                           reliability_scale=reliability_scale)

    # Build the training environment based on the DQN configuration file
    train_env = FactoryEnv(scenario,
                           objective_function=objective_function,
                           max_steps=dqn_config_dict["environment"]["max_steps"],
                           seed=dqn_config_dict["environment"]["training_seed"])

    check_env(train_env, warn=True)
    train_env = Monitor(train_env)

    # Create a new scenario clone for evaluation
    eval_scenario = ScenarioCore(file_path=scenario_file_path, reset_class=True)

    # Build the evaluation environment based on the DQN configuration file
    eval_env = FactoryEnv(eval_scenario,
                          objective_function=objective_function,
                          max_steps=dqn_config_dict["environment"]["max_steps"],
                          seed=dqn_config_dict["environment"]["evaluation_seed"])

    check_env(eval_env, warn=True)
    eval_env = Monitor(eval_env)

    # Get scenario output directory path (based on hash value of scenario JSON file)
    output_dir_path = get_scenario_data_dir_path(scenario_file_path=scenario_file_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Get dqn model output directory path (based on hash value of dqn config JSON file)
    output_dqn_dir_path = get_dqn_model_dir_path(scenario_file_path=scenario_file_path, output_dir_path=output_dir_path)
    output_dqn_dir_path.mkdir(parents=True, exist_ok=True)

    # Define log output (for tensorboard)
    logs_path = output_dqn_dir_path / DIR_NAME_LOGS
    logs_path.mkdir(parents=True, exist_ok=True)

    # Define model output (best & final models)
    models_path = output_dqn_dir_path / DIR_NAME_MODELS
    models_path.mkdir(parents=True, exist_ok=True)

    # For dict observation: MultiInputPolicy
    model = DQN(
        policy=dqn_config_dict["training"]["policy"],
        env=train_env,
        learning_rate=dqn_config_dict["training"]["learning_rate"],
        buffer_size=dqn_config_dict["training"]["buffer_size"],
        learning_starts=dqn_config_dict["training"]["learning_starts"],
        batch_size=dqn_config_dict["training"]["batch_size"],
        gamma=dqn_config_dict["training"]["gamma"],
        target_update_interval=dqn_config_dict["training"]["target_update_interval"],
        train_freq=dqn_config_dict["training"]["train_freq"],
        gradient_steps=dqn_config_dict["training"]["gradient_steps"],
        exploration_fraction=dqn_config_dict["training"]["exploration_fraction"],
        exploration_final_eps=dqn_config_dict["training"]["exploration_final_eps"],
        tensorboard_log=str(logs_path),
        verbose=1,
    )

    # Periodic evaluation
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(models_path),
        log_path=str(logs_path),
        eval_freq=dqn_config_dict["evaluation"]["eval_freq"],
        deterministic=dqn_config_dict["evaluation"]["deterministic"],
        render=dqn_config_dict["evaluation"]["render"],
    )

    start_time = time.time()

    # Start training the model
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    training_duration_seconds = time.time() - start_time
    training_duration_str = duration_formatting(training_duration_seconds)
    print(f"Training time -> {training_duration_str}")

    timestamp_training_str = f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {training_duration_str}"

    with open(models_path / "last_training_duration.txt", "w") as f:
        f.write(timestamp_training_str)

    model.save(path=models_path / FILE_NAME_FINAL_MODEL)

    # Copy the scenario and configuration used for training for reference later
    backup_scenario_file_path = output_dir_path / f"{DIR_NAME_BACKUP}_{scenario_file_path.name}"
    backup_dqn_config_file_path = output_dqn_dir_path / f"{DIR_NAME_BACKUP}_{FILE_NAME_DQN_CONFIG}"

    shutil.copy(scenario_file_path, backup_scenario_file_path)
    shutil.copy(scenario_file_path.parent / FILE_NAME_DQN_CONFIG, backup_dqn_config_file_path)

    print("Done.")

    return training_duration_seconds


def run_inference(scenario_file_path: Path, objective_function: ObjectiveFunction,
                  count: int = 1, quick_eval: bool = False, use_best_model: bool = False,
                  progress_updater = DummyProgressUpdater()) -> OptimizationResult:

    progress_updater.text = "Preparing DQN inference..."
    progress_updater.percentage = 0

    start_time = time.time()

    # Get output directory path (based on hash value of JSON file)
    output_dir_path = get_scenario_data_dir_path(scenario_file_path=scenario_file_path)

    if not output_dir_path.exists():
        raise FileNotFoundError(f"Directory '{output_dir_path.name}' doesn't exist. Please train a model first.")

    dqn_model_dir_path = get_dqn_model_dir_path(scenario_file_path=scenario_file_path, output_dir_path=output_dir_path)

    if not output_dir_path.exists():
        raise FileNotFoundError(f"Directory '{dqn_model_dir_path.name}' doesn't exist. Please train a model first.")

    # Load the DQN training configuration
    dqn_config_dict = load_dqn_config(scenario_file_path=scenario_file_path)

    scenario = ScenarioCore(file_path=scenario_file_path, reset_class=True)

    time_scale, energy_scale, reliability_scale = compute_cost_scales(scenario)
    objective_function = ObjectiveFunction(time_weight=dqn_config_dict["environment"]["time_weight"],
                                           energy_weight=dqn_config_dict["environment"]["energy_weight"],
                                           reliability_weight=dqn_config_dict["environment"]["reliability_weight"],
                                           time_scale=time_scale, energy_scale=energy_scale,
                                           reliability_scale=reliability_scale)

    original_env = FactoryEnv(scenario=scenario,
                              objective_function=objective_function,
                              max_steps=dqn_config_dict["environment"]["max_steps"],
                              seed=dqn_config_dict["environment"]["inference_seed"],
                              output_action_state=dqn_config_dict["environment"]["output_action_state"])

    env = Monitor(original_env)

    if use_best_model:
        model_filepath = dqn_model_dir_path / DIR_NAME_MODELS / FILE_NAME_BEST_MODEL
    else:
        model_filepath = dqn_model_dir_path / DIR_NAME_MODELS / FILE_NAME_FINAL_MODEL

    if not model_filepath.exists():
        raise FileNotFoundError(f"The '{model_filepath.name}' model could not be found.")

    model = DQN.load(path=model_filepath, env=env)

    if quick_eval:

        progress_updater.text = "Quick evaluation..."
        progress_updater.percentage = 10

        # Quick evaluation
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
        print(f"\n--> DQN evaluation - Mean reward: {mean_r:.3f} ± {std_r:.3f}\n")

    progress_updater.text = "Running DQN inference..."
    progress_updater.percentage = 25

    shortest_done_length = 999

    for i in range(count):

        print(f"--- Round {i+1} ---")

        # Manual rollout (deterministic)
        obs, info = env.reset(seed=2025)
        done = False
        truncated = False
        action_idx_sequence = []

        counter = 0

        start_time = time.time()

        while not (done or truncated):

            counter += 1

            # Obs is a dictionary, action is an integer (index of action catalog)
            action_idx, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action_idx)

            action_idx_sequence.append(int(action_idx))

            print(f"{counter}. {original_env.scenario.sorted_action_catalog[action_idx]} -> {original_env.scenario.get_product_states()}")

        print(f"Time {original_env.scenario.time_sum:.2f} & Energy {original_env.scenario.energy_sum:.2f} -> {time.time() - start_time:.5f} seconds")
        print("")

        shortest_done_length = min(shortest_done_length, counter)

    print(f"Shortest sequence until done: {shortest_done_length}")

    progress_updater.text = "Done."
    progress_updater.percentage = 100

    scenario.reset()

    # Re-simulate to get actual actions taken until done
    _, _, actions_taken = scenario.execute_action_idx_sequence(np.array(action_idx_sequence))


    return OptimizationResult(action_idx_sequence=list(action_idx_sequence),
                              task_result_list=scenario.task_result_history,
                              total_time=scenario.time_sum,
                              total_energy=scenario.energy_sum,
                              sequence_reliability=scenario.sequence_reliability,
                              objective_function=objective_function,
                              other_params_dict={OPT_RES_PARAM_REWARD: reward},
                              total_duration_seconds=(time.time() - start_time),
                              opt_method=OptimizationMethod.OPT_RL_DQN)
