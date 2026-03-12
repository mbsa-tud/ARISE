# -*- coding: utf-8 -*-

"""
Main module of the ICM ARISE Project

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import random

from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.model.machines import Machine
from src.arise_project.model.tasks import ProcessingTask
from src.arise_project.model.scenario import Scenario
from src.arise_project.scheduler.scheduler import Scheduler
from src.arise_project.scheduler.q_learning import QLearningAlgorithm

from src.arise_project.scheduler.factory_dqn_training import run_training, run_inference


def main():

    # Load a scenario (product and factory) and create a simulation interface
    scenario = Scenario(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)

    scheduler = Scheduler(scenario=scenario)
    scheduler.optimize()

    print("Completed")


def main_random_processing():

    # Load a scenario (product and factory)
    scenario = Scenario(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)
    factory = scenario.factory
    product = scenario.get_sorted_product_list()[0]

    remaining_tasks = product.get_remaining_processing_tasks()
    loop_counter = 0

    print(f"Processing {product} -> Target: {product.target_state}\n")

    while len(remaining_tasks) > 0:

        loop_counter += 1
        print(f"Round {loop_counter}")

        random_task: ProcessingTask = random.choice(list(remaining_tasks))
        print(f"Task: {random_task}")

        skill_type = random_task.possible_skill_types[0]

        possible_machines = factory.get_machines_by_skill_type(skill_type)

        random_possible_machine: Machine = random.choice(list(possible_machines))
        print(f"Selected machine: {random_possible_machine}")

        task_result = random_possible_machine.process(product, random_task)

        print(f"Used skill: {task_result.skill}, "
              f"Total time: {task_result.total_time}, "
              f"Total energy: {task_result.total_energy}, "
              f"Success: {task_result.success_bool}")

        remaining_tasks = product.get_remaining_processing_tasks()

        print(f"Current state: {product.current_state} (remaining tasks: {len(remaining_tasks)})\n")

    print(f"Completed processing {product}\n")
    print(f"{product.state_history_list}\n")

    product.print_processing_history()


def main_sim():

    # Load a scenario (product and factory)
    scenario = Scenario(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)

    possible_actions = scenario.get_specific_actions()
    loop_counter = 0

    while len(possible_actions) > 0:

        loop_counter += 1
        print(f"Round {loop_counter}")

        print(f"Possible actions ({len(possible_actions)}): ")

        for idx, task_result in enumerate(possible_actions):
            print(f"{idx + 1}. ({task_result})")

        # Randomly choose and action to execute
        random_action = random.choice(possible_actions)
        scenario.execute_action(random_action)

        print(f"--> Executed action: < {random_action.task.unique_id} | {random_action.skill.unique_id} >\n")

        possible_actions = scenario.get_specific_actions()

    print(f"Completed processing")

    scenario.get_sorted_product_list()[0].print_processing_history()


def main_q_learning():

    # Load a scenario (product and factory)
    scenario = Scenario(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)

    q_learning_alg = QLearningAlgorithm(scenario=scenario, num_episodes=3000)
    q_learning_alg.train()

    scenario.reset()

    loop_counter = 0

    while not scenario.is_done():

        loop_counter += 1

        action_key = q_learning_alg.greedy_action_from_current_state()
        scenario.execute_action_key(action_key=action_key)

        print(f"{loop_counter}. {action_key}")

        if loop_counter > 15:
            break


if __name__ == "__main__":

    # main()
    # main_random_processing()
    # main_sim()
    # main_q_learning()

    training_bool = False

    scenario_file_path = FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH

    if training_bool:
        run_training(scenario_file_path=scenario_file_path)
    else:
        run_inference(scenario_file_path=scenario_file_path, count=5, quick_eval=True)

    pass
