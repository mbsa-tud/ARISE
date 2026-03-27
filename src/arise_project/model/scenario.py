# -*- coding: utf-8 -*-

"""
Module containing the scenario class definition

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import json
import random

import numpy as np
from jsonschema import validate, ValidationError

from pathlib import Path

from arise_project.model.objective import ObjectiveFunction
from arise_project.model.variability import ProcessVariability
from src.arise_project.config.paths import FILE_SCENARIO_JSON_SCHEMA_PATH
from src.arise_project.model.action_key import ActionKey
from src.arise_project.model.factory import Factory
from src.arise_project.model.machines import StorageMachine, DrillingMachine, CuttingMachine, MillingMachine, \
    AutomatedGuidedVehicle, ThreeAxesRobot, ConveyorBelt
from src.arise_project.model.skills import MillingSkill, DrillingSkill, CuttingSkill, TransportSkill, StoreSkill, \
    RetrieveSkill

from src.arise_project.model.tasks import DrillingTask, CuttingTask, MillingTask, ProcessingTask, Task
from src.arise_project.model.task_results import TaskResult
from src.arise_project.model.product import Plate, Product
from src.arise_project.model.product_state import ProductState
from src.arise_project.model.tasks import TransportTask


class Scenario:

    def __init__(self, file_path: Path, random_seed: int | None = None, reset_class: bool = False) -> None:

        if reset_class:
            Scenario.reset_all()

        if random_seed is not None:
            random.seed(random_seed)

        self._factory = Factory()
        self._product_by_id_dict = {}
        self._task_result_history = []

        self._sorted_action_catalog: list[ActionKey] = []

        self._step_count = 0
        self._time_sum = 0.0
        self._energy_sum = 0.0
        self._sequence_reliability = 1.0
        self._done_products = 0

        self._file_path = file_path
        self._load_from_json(self._file_path)

    @property
    def factory(self) -> Factory:
        return self._factory

    @property
    def product_by_id_dict(self) -> dict[str, Product]:
        return self._product_by_id_dict

    def get_sorted_product_list(self) -> list[Product]:
        """
        Make sure list of products is always sorted alphabetically by unique id to ensure consistency
        :return: Sorted list of products in the factory (list)
        """

        result_list = list(self._product_by_id_dict.values())
        result_list.sort()

        return result_list

    @property
    def task_result_history(self) -> list[TaskResult]:
        return self._task_result_history

    @property
    def sorted_action_catalog(self) -> list[ActionKey]:
        return self._sorted_action_catalog

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def time_sum(self) -> float:
        return self._time_sum

    @property
    def energy_sum(self) -> float:
        return self._energy_sum

    @property
    def sequence_reliability(self) -> float:
        return self._sequence_reliability

    def _load_from_json(self, file_path: Path) -> None:

        with open(file_path, 'r') as json_file:
            data_dict = json.load(json_file)

        with open(FILE_SCENARIO_JSON_SCHEMA_PATH, 'r') as schema_file:
            schema = json.load(schema_file)

        # Validate the JSON data against the schema
        try:
            validate(instance=data_dict, schema=schema)

        except ValidationError as e:
            print("Unable to load scenario due to JSON validation error:", e.message)

        # --- Create the factory ---

        self._factory = Factory()
        factory_dict = data_dict["factory"]
        stationary_machine_obj_by_name_dict = {}

        # Go through all storage machines
        for stm in factory_dict["storage_machines"]:

            if stm["class"] == "StorageMachine":

                # Create skill with the machine's skill-specific parameters
                store_skill = StoreSkill(time_factor=stm["skill_params"]["store_skill"]["time_factor"],
                                         energy_factor=stm["skill_params"]["store_skill"]["energy_factor"],
                                         reliability=stm["skill_params"]["store_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=stm["skill_params"]["store_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=stm["skill_params"]["store_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=stm["skill_params"]["store_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=stm["skill_params"]["store_skill"]["variability"]["normal_dist_sigma_factor"]))

                # Create skill with the machine's skill-specific parameters
                retrieve_skill = RetrieveSkill(time_factor=stm["skill_params"]["retrieve_skill"]["time_factor"],
                                               energy_factor=stm["skill_params"]["retrieve_skill"]["energy_factor"],
                                               reliability=stm["skill_params"]["retrieve_skill"]["reliability"],
                                               process_variability=ProcessVariability(use_normal_distribution=stm["skill_params"]["retrieve_skill"]["variability"]["use_normal_distribution"],
                                                                                      uniform_time_variability=stm["skill_params"]["retrieve_skill"]["variability"]["uniform_time_variability"],
                                                                                      uniform_energy_variability=stm["skill_params"]["retrieve_skill"]["variability"]["uniform_energy_variability"],
                                                                                      normal_dist_sigma_factor=stm["skill_params"]["retrieve_skill"]["variability"]["normal_dist_sigma_factor"]))

                storage_machine = StorageMachine(name=stm["name"], x=stm["x"], y=stm["y"],
                                                 store_skill=store_skill, retrieve_skill=retrieve_skill)

                stationary_machine_obj_by_name_dict[stm["name"]] = storage_machine
                self._factory.add_machine(storage_machine)

        # Go through all processing machines
        for prm in factory_dict["processing_machines"]:

            if prm["class"] == "DrillingMachine":

                # Create skill with the machine's skill-specific parameters
                drilling_skill = DrillingSkill(time_factor=prm["skill_params"]["drilling_skill"]["time_factor"],
                                               energy_factor=prm["skill_params"]["drilling_skill"]["energy_factor"],
                                               reliability=prm["skill_params"]["drilling_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=prm["skill_params"]["drilling_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=prm["skill_params"]["drilling_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=prm["skill_params"]["drilling_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=prm["skill_params"]["drilling_skill"]["variability"]["normal_dist_sigma_factor"]))

                drilling_machine = DrillingMachine(name=prm["name"], x=prm["x"], y=prm["y"],
                                                   drilling_skill=drilling_skill)

                stationary_machine_obj_by_name_dict[prm["name"]] = drilling_machine
                self._factory.add_machine(drilling_machine)

            elif prm["class"] == "CuttingMachine":

                # Create skill with the machine's skill-specific parameters
                cutting_skill = CuttingSkill(time_factor=prm["skill_params"]["cutting_skill"]["time_factor"],
                                             energy_factor=prm["skill_params"]["cutting_skill"]["energy_factor"],
                                             reliability=prm["skill_params"]["cutting_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=prm["skill_params"]["cutting_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=prm["skill_params"]["cutting_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=prm["skill_params"]["cutting_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=prm["skill_params"]["cutting_skill"]["variability"]["normal_dist_sigma_factor"]))

                cutting_machine = CuttingMachine(name=prm["name"], x=prm["x"], y=prm["y"],
                                                 cutting_skill=cutting_skill)

                stationary_machine_obj_by_name_dict[prm["name"]] = cutting_machine
                self._factory.add_machine(cutting_machine)

            elif prm["class"] == "MillingMachine":

                # Create skill with the machine's skill-specific parameters
                milling_skill = MillingSkill(time_factor=prm["skill_params"]["milling_skill"]["time_factor"],
                                             energy_factor=prm["skill_params"]["milling_skill"]["energy_factor"],
                                             reliability=prm["skill_params"]["milling_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=prm["skill_params"]["milling_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=prm["skill_params"]["milling_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=prm["skill_params"]["milling_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=prm["skill_params"]["milling_skill"]["variability"]["normal_dist_sigma_factor"]))

                milling_machine = MillingMachine(name=prm["name"], x=prm["x"], y=prm["y"],
                                                 milling_skill=milling_skill)

                stationary_machine_obj_by_name_dict[prm["name"]] = milling_machine
                self._factory.add_machine(milling_machine)

        # Go through all transporter machines
        for tpm in factory_dict["transporter_machines"]:

            if (len(tpm["from"]) < 1) or (len(tpm["to"]) < 1):
                raise ValueError(f"The 'from' and 'to' lists of transporter machines must not be empty.")

            if tpm["from"][0] == "*":
                # Select all storage and processing machines (all stationary machines)
                from_machine_list = list(self._factory.stationary_machine_by_id_dict.values())
            else:
                # Translate list from names to unique ids generated during object instantiation
                from_machine_list = [stationary_machine_obj_by_name_dict[name] for name in tpm["from"]]

            if tpm["to"][0] == "*":
                # Select all storage and processing machines (all stationary machines)
                to_machine_list = list(self._factory.stationary_machine_by_id_dict.values())
            else:
                # Translate list from names to unique ids generated during object instantiation
                to_machine_list = [stationary_machine_obj_by_name_dict[name] for name in tpm["to"]]

            if tpm["class"] == "AutomatedGuidedVehicle":

                # Create skill with the machine's skill-specific parameters
                transport_skill = TransportSkill(time_factor=tpm["skill_params"]["transport_skill"]["time_factor"],
                                                 energy_factor=tpm["skill_params"]["transport_skill"]["energy_factor"],
                                                 reliability=tpm["skill_params"]["transport_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=tpm["skill_params"]["transport_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=tpm["skill_params"]["transport_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=tpm["skill_params"]["transport_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=tpm["skill_params"]["transport_skill"]["variability"]["normal_dist_sigma_factor"]))

                agv = AutomatedGuidedVehicle(name=tpm["name"], transport_skill=transport_skill)

                self._factory.add_machine(agv)
                self._factory.create_connections(transporter_machine=agv,
                                                 from_machine_list=from_machine_list,
                                                 to_machine_list=to_machine_list)

            elif tpm["class"] == "ThreeAxesRobot":

                # Create skill with the machine's skill-specific parameters
                transport_skill = TransportSkill(time_factor=tpm["skill_params"]["transport_skill"]["time_factor"],
                                                 energy_factor=tpm["skill_params"]["transport_skill"]["energy_factor"],
                                                 reliability=tpm["skill_params"]["transport_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=tpm["skill_params"]["transport_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=tpm["skill_params"]["transport_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=tpm["skill_params"]["transport_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=tpm["skill_params"]["transport_skill"]["variability"]["normal_dist_sigma_factor"]))

                tar = ThreeAxesRobot(name=tpm["name"], transport_skill=transport_skill)

                self._factory.add_machine(tar)
                self._factory.create_connections(transporter_machine=tar,
                                                 from_machine_list=from_machine_list,
                                                 to_machine_list=to_machine_list)

            elif tpm["class"] == "ConveyorBelt":

                # Create skill with the machine's skill-specific parameters
                transport_skill = TransportSkill(time_factor=tpm["skill_params"]["transport_skill"]["time_factor"],
                                                 energy_factor=tpm["skill_params"]["transport_skill"]["energy_factor"],
                                                 reliability=tpm["skill_params"]["transport_skill"]["reliability"],
                                         process_variability=ProcessVariability(use_normal_distribution=tpm["skill_params"]["transport_skill"]["variability"]["use_normal_distribution"],
                                                                                uniform_time_variability=tpm["skill_params"]["transport_skill"]["variability"]["uniform_time_variability"],
                                                                                uniform_energy_variability=tpm["skill_params"]["transport_skill"]["variability"]["uniform_energy_variability"],
                                                                                normal_dist_sigma_factor=tpm["skill_params"]["transport_skill"]["variability"]["normal_dist_sigma_factor"]))

                cob = ConveyorBelt(name=tpm["name"], transport_skill=transport_skill)

                self._factory.add_machine(cob)
                self._factory.create_connections(transporter_machine=cob,
                                                 from_machine_list=from_machine_list,
                                                 to_machine_list=to_machine_list)

        # --- Create the product ---
        self._product_by_id_dict = {}
        product_dict = data_dict["product"]

        if product_dict["class"] == "Plate":

            # Find the starting and target location (machines)
            starting_location_machine = stationary_machine_obj_by_name_dict[product_dict["starting_location"]]
            target_location_machine = stationary_machine_obj_by_name_dict[product_dict["target_location"]]

            if (not isinstance(starting_location_machine, StorageMachine) or
                    not isinstance(target_location_machine, StorageMachine)):

                raise ValueError(f"The 'starting_location' and 'target_location' must be of type 'StorageMachine'.")

            # Create the specified number of products
            for i in range(0, product_dict["count"]):

                processing_task_list = []

                for task in product_dict["processing_tasks"]:

                    if task["class"] == "DrillingTask":

                        drilling_task = DrillingTask(center_x=task["params"]["center_x"],
                                                     center_y=task["params"]["center_y"],
                                                     radius=task["params"]["radius"])

                        processing_task_list.append(drilling_task)

                    elif task["class"] == "CuttingTask":

                        cutting_task = CuttingTask(total_length=task["params"]["total_length"])
                        processing_task_list.append(cutting_task)

                    elif task["class"] == "MillingTask":

                        milling_task = MillingTask(total_area=task["params"]["total_area"])
                        processing_task_list.append(milling_task)

                # Generate a product with an individual processing task list
                new_product = Plate(width=product_dict["params"]["width"],
                                    height=product_dict["params"]["height"],
                                    starting_location_id=starting_location_machine.unique_id,
                                    target_location_id=target_location_machine.unique_id,
                                    processing_tasks=processing_task_list)

                self._product_by_id_dict[new_product.unique_id] = new_product

        if len(self._product_by_id_dict) < 1:
            raise ValueError(f"The number of products must be greater than 0.")

        # Once loaded, generate the sorted action catalog (regenerate if factory or product changes)
        self._generate_sorted_action_catalog()

    def get_actions(self) -> list[TaskResult]:

        all_possible_actions = []

        for product in self._product_by_id_dict.values():

            for task in product.get_remaining_processing_tasks():

                for skill_type in task.possible_skill_types:

                    for machine in self._factory.get_machines_by_skill_type(skill_type):

                        if product.current_state.location_machine_id == machine.unique_id:

                            task_result = machine.calculate(product, task)
                            all_possible_actions.append(task_result)

            for transport_machine in self._factory.transport_machine_by_id_dict.values():

                for transport_task in self._factory.transport_task_list_by_transport_dict[transport_machine.unique_id]:

                    # Only consider transportation from current machine to any machine other than the current machine
                    if ((transport_task.source_machine_id != product.current_state.location_machine_id)
                            or (transport_task.target_machine_id == product.current_state.location_machine_id)):
                        continue

                    task_result = transport_machine.calculate(product, transport_task)
                    all_possible_actions.append(task_result)

        return all_possible_actions

    def get_action_identities(self) -> list[ActionKey]:

        return [ActionKey(product=task_result.product,
                          task=task_result.task,
                          skill=task_result.skill)

                for task_result in self.get_actions()]

    def get_specific_actions(self) -> list[TaskResult]:
        """
        Returns a list of all possible actions available based on the state of the simulation (in this case product).
        Actions are identified and precalculated task results, whose tasks have not yet been executed.

        :return: list of task results (TaskResult)
        """

        all_possible_actions = []

        for task in self.get_sorted_product_list()[0].get_remaining_processing_tasks():

            for skill_type in task.possible_skill_types:

                for machine in self.factory.get_machines_by_skill_type(skill_type):

                    task_result = machine.calculate(self.get_sorted_product_list()[0], task)
                    all_possible_actions.append(task_result)

        return all_possible_actions

    def _generate_sorted_action_catalog(self):

        action_catalog: list[ActionKey] = []

        # For each product
        for product in self.get_sorted_product_list():

            # And for each task of a product
            for task in product.target_state.processing_tasks:

                # Get all possible skill types that can be used to execute a task
                for skill_type in task.possible_skill_types:

                    # For each skill type find every machine that has a skill of this type
                    for machine in self.factory.get_machines_by_skill_type(skill_type):

                        skill = machine.get_skill_by_type(skill_type)

                        new_action_key = ActionKey(product=product, task=task, skill=skill)
                        action_catalog.append(new_action_key)

            for transport_machine in self._factory.transport_machine_by_id_dict.values():

                transport_skill = transport_machine.get_skill_by_type(TransportSkill)

                for transport_task in self._factory.transport_task_list_by_transport_dict[transport_machine.unique_id]:

                    action_catalog.append(ActionKey(product=product, task=transport_task, skill=transport_skill))

        # Make sure the action catalog is always sorted the same way, the ordering is extremely important1
        action_catalog.sort()

        # Remove duplicates (Python idiom) while keeping the ordering (Python 3.7+)
        self._sorted_action_catalog = list(dict.fromkeys(action_catalog))

    def generate_feasible_action_mask(self) -> np.ndarray:

        # Get set of all feasible actions in current state
        feasible_set = set(self.get_feasible_actions())

        # Initialize the action mask as a numpy array of zeros, one for each action in the action catalog
        action_mask = np.zeros((len(self._sorted_action_catalog),), dtype=np.int8)

        # Each feasible action of the action catalog is marked with a '1'
        for idx, action_key in enumerate(self._sorted_action_catalog):
            action_mask[idx] = 1 if action_key in feasible_set else 0

        return action_mask

    def get_feasible_actions_idx_list(self) -> list[int]:

        action_mask = self.generate_feasible_action_mask()
        return np.flatnonzero(action_mask)

    def get_product_states(self) -> dict[str, ProductState]:

        result_dict = {}

        for product_id in self._product_by_id_dict.keys():

            result_dict[product_id] = self._product_by_id_dict[product_id].current_state

        return result_dict

    def get_sorted_all_tasks_list(self) -> list[Task]:
        """
        Make sure list of processing & transport tasks is always sorted alphabetically by unique id to ensure consistency
        :return: Sorted list of processing & transport tasks in the factory (list)
        """

        result_list = []

        for product in self._product_by_id_dict.values():

            result_list.extend(list(product.target_state.processing_tasks))

        for transport_machine in self._factory.transport_machine_by_id_dict.values():

            result_list.extend(self._factory.transport_task_list_by_transport_dict[transport_machine.unique_id])

        # Make sure list of tasks is always sorted alphabetically by unique id to ensure consistency
        result_list.sort()

        return result_list

    def get_sorted_processing_tasks_list(self) -> list[Task]:
        """
        Make sure list of processing tasks is always sorted alphabetically by unique id to ensure consistency
        :return: Sorted list of processing tasks in the factory (list)
        """

        result_list = []

        for product in self._product_by_id_dict.values():
            result_list.extend(list(product.target_state.processing_tasks))

        # Make sure list of tasks is always sorted alphabetically by unique id to ensure consistency
        result_list.sort()

        return result_list

    def step(self, action_key: ActionKey) -> tuple[TaskResult, bool, bool] | tuple[None, None, None]:

        task_result = self.execute_action_key(action_key=action_key)

        # Illegal action
        if task_result is None:
            return None, None, None

        completed_product_count = 0

        for product in self._product_by_id_dict.values():

            if product.is_done():
                completed_product_count = completed_product_count + 1

        product_done_this_step_bool = completed_product_count > self._done_products
        self._done_products = completed_product_count

        all_products_done_bool = self.is_done()

        return task_result, product_done_this_step_bool, all_products_done_bool

    def step_by_action_idx(self, action_idx: int) -> tuple[TaskResult, bool, bool] | tuple[None, None, None]:

        action_key = self._sorted_action_catalog[action_idx]

        return self.step(action_key=action_key)

    def get_feasible_actions(self) -> list[ActionKey]:
        return self.get_action_identities()

    def execute_action(self, task_result: TaskResult) -> None:

        product = task_result.product

        # If successful, update product state
        if task_result.success_bool:
            product.update_state_by_task_result(task_result=task_result)

        self._step_count = self._step_count + 1
        self._time_sum = self._time_sum + task_result.total_time
        self._energy_sum = self._energy_sum + task_result.total_energy
        self._sequence_reliability = self._sequence_reliability * task_result.skill.reliability

        self._task_result_history.append(task_result)

    def undo_last_action(self) -> None:

        # Make sure there is an action to undo
        if len(self._task_result_history) > 0:

            last_executed_task_result = self._task_result_history.pop()

            product = last_executed_task_result.product

            # If successful, update product state
            if last_executed_task_result.success_bool:
                product.undo_last_state_change()

            self._step_count = self._step_count - 1
            self._time_sum = self._time_sum - last_executed_task_result.total_time
            self._energy_sum = self._energy_sum - last_executed_task_result.total_energy
            self._sequence_reliability = self._sequence_reliability / last_executed_task_result.skill.reliability

    def execute_action_key(self, action_key: ActionKey) -> TaskResult | None:

        # Find objects associated to unique ids
        selected_product = self._product_by_id_dict[action_key.product_id]

        # Feasibility check
        current_machine = self._factory.get_machine_by_id(selected_product.current_state.location_machine_id)
        processing_skill_available = action_key.skill_id in current_machine.skill_by_id_dict

        valid_transport_task = False

        # If processing skill was found, there is no need to search for a transport skill
        if not processing_skill_available:

            if action_key.task_id in self._factory.transport_task_by_id_dict:

                # Check if source machine matches current location
                if (self._factory.transport_task_by_id_dict[action_key.task_id].source_machine_id
                        == selected_product.current_state.location_machine_id):

                    valid_transport_task = True

        # The requested skill must either be available at the current machine or be a transport skill with the
        # current machine as the source machine, otherwise the action is illegal.
        if not processing_skill_available and not valid_transport_task:
            return None

        if selected_product.target_state.contains_task_with_id(action_key.task_id):
            selected_task = selected_product.target_state.get_task_by_id(action_key.task_id)

        elif action_key.task_id in self._factory.transport_task_by_id_dict:
            selected_task = self._factory.transport_task_by_id_dict[action_key.task_id]

        else:
            raise ValueError(f"Task with id {action_key.task_id} could not be found.")

        selected_machine = self._factory.get_machine_with_skill_id(action_key.skill_id)
        selected_skill = self._factory.get_skill_with_id(action_key.skill_id)

        # Calculate task result
        task_result = selected_machine.calculate(product=selected_product, task=selected_task, skill=selected_skill)
        self.execute_action(task_result)

        return task_result

    def execute_action_idx_sequence(self, seq: np.ndarray, check_validity: bool = False, random_seed: int | None = None) -> tuple[bool, int, list[int]]:
        """
        Roll out a sequence of action indices in the environment.
        Returns:
            done, steps_used, actions_taken
        """

        self.reset(random_seed=random_seed)

        actions_taken = []
        all_products_done = False

        for i, action_idx in enumerate(seq):

            if check_validity:

                # Create the action mask of all feasible actions in the current state
                action_mask = self.generate_feasible_action_mask()

                # Discontinue execution if action is not feasible
                if action_mask[int(action_idx)] == 0:
                    break

            # Execute the action by its index in the action catalog
            task_result, product_done, all_products_done = self.step_by_action_idx(int(action_idx))

            if task_result is None:
                break

            actions_taken.append(int(action_idx))

            if all_products_done:
                break

        steps_used = len(actions_taken)

        return all_products_done, steps_used, actions_taken

    def print_task_result_history(self, show_numerical_index: bool = False) -> None:

        for idx, task_result in enumerate(self._task_result_history):

            line_str = ""

            if show_numerical_index:
                 line_str += f"{idx+1}. "

            line_str += f"{task_result} "

            print(line_str)

    def calculate_total_cost(self, objective_function: ObjectiveFunction) -> float:

        return objective_function(time_cost=self._time_sum,
                                  energy_cost=self._energy_sum,
                                  reliability=self._sequence_reliability)

    def reset(self, random_seed: int = None) -> None:

        if random_seed is not None:
            random.seed(random_seed)

        Scenario.reset_all()

        self._factory = Factory()
        self._product_by_id_dict = {}
        self._task_result_history = []

        self._step_count = 0
        self._time_sum = 0.0
        self._energy_sum = 0.0
        self._sequence_reliability = 1.0
        self._done_products = 0

        self._load_from_json(self._file_path)

    def is_done(self) -> bool:

        # Not done as long as at least one product exists which is not done
        for product in self._product_by_id_dict.values():

            if not product.is_done():
                return False

        return True

    @classmethod
    def reset_all(cls):

        # TODO Find a more elegant solution than listing all classes here
        class_list = [DrillingMachine, MillingMachine, CuttingMachine,
                      ConveyorBelt, AutomatedGuidedVehicle, ThreeAxesRobot, StorageMachine,
                      DrillingSkill, MillingSkill, CuttingSkill, TransportSkill, StoreSkill, RetrieveSkill,
                      DrillingTask, MillingTask, CuttingTask, TransportTask,
                      Plate]

        for class_type in class_list:
            class_type.reset_unique_id_ctr()
