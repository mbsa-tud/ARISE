# -*- coding: utf-8 -*-

"""
Module defining the factory class

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from typing import Type, List

import numpy as np
import pandas as pd
import networkx as nx
from networkx.classes import DiGraph

from src.arise_project.config.colors import COLOR_BY_STATIONARY_MACHINE_DICT
from src.arise_project.model.machines import Machine, StationaryMachine, StorageMachine, ProcessingMachine, TransporterMachine
from src.arise_project.model.skills import Skill
from src.arise_project.model.tasks import TransportTask


class Factory:

    def __init__(self) -> None:

        # Machine objects in dictionaries by id (for fast access)
        self._machine_by_id_dict = {}
        self._storage_machine_by_id_dict = {}
        self._processing_machine_by_id_dict = {}
        self._transport_machine_by_id_dict = {}
        self._stationary_machine_by_id_dict = {}

        # Machine objects in dictionary by offered skills (for fast access)
        self._machines_by_skill_dict = {}

        # List of edges: (source_machine_id, target_machine_id, transport_machine)
        self._transport_connections = []
        self._transport_task_list_by_transport_dict = {}

        self._stationary_machine_distance_df = pd.DataFrame()

    @property
    def machine_by_id_dict(self) -> dict[str, Machine]:
        return self._machine_by_id_dict

    @property
    def storage_machine_by_id_dict(self) -> dict[str, StorageMachine]:
        return self._storage_machine_by_id_dict

    @property
    def processing_machine_by_id_dict(self) -> dict[str, ProcessingMachine]:
        return self._processing_machine_by_id_dict

    @property
    def transport_machine_by_id_dict(self) -> dict[str, TransporterMachine]:
        return self._transport_machine_by_id_dict

    @property
    def stationary_machine_by_id_dict(self) -> dict[str, StationaryMachine]:
        return self._stationary_machine_by_id_dict

    @property
    def machines_by_skill_dict(self) -> dict[str, set[Machine]]:
        return self._machines_by_skill_dict

    @property
    def transport_connections(self) -> list[tuple[str, str, str]]:
        return self._transport_connections

    @property
    def transport_task_list_by_transport_dict(self) -> dict[str, List[TransportTask]]:
        return self._transport_task_list_by_transport_dict

    def get_sorted_all_machines_list(self) -> List[Machine]:
        """
        Make sure list of machines is always sorted alphabetically by unique id to ensure consistency
        :return: Sorted list of machines in the factory (list)
        """

        result_list = list(self._machine_by_id_dict.values())
        result_list.sort()

        return result_list

    def get_sorted_all_stationary_machines_list(self) -> List[StationaryMachine]:
        """
        Make sure list of stationary machines is always sorted alphabetically by unique id to ensure consistency
        :return: Sorted list of stationary machines in the factory (list)
        """

        result_list = list(self._stationary_machine_by_id_dict.values())
        result_list.sort()

        return result_list

    def get_machine_with_skill_id(self, unique_id: str) -> Machine | None:
        """
        Find machine with specified skill (by unique id). This is possible as each skill
        is only associated with one machine.

        :param unique_id: skill id (str)
        :return: machine with skill, if found else None (Machine | None)
        """

        for machine in self._machine_by_id_dict.values():

            if unique_id in machine.skill_by_id_dict:
                return machine

        raise ValueError(f"Skill with id {unique_id} does not exist anywhere in the factory.")

    def get_skill_with_id(self, unique_id: str) -> Skill | None:

        """
        Find skill by unique id by searching across all machines.

        :param unique_id: skill id (str)
        :return: skill with id, if found else None (Skill | None)
        """

        for machine in self._machine_by_id_dict.values():

            if unique_id in machine.skill_by_id_dict:
                return machine.skill_by_id_dict[unique_id]

        raise ValueError(f"Skill with id {unique_id} does not exist anywhere in the factory.")

    def get_transport_task_with_id(self, unique_id: str) -> TransportTask | None:

        for transport_task_list in self._transport_task_list_by_transport_dict.values():

            for transport_task in transport_task_list:

                if transport_task.unique_id == unique_id:
                    return transport_task

        return None

    def contains_transport_task_with_id(self, unique_id: str) -> bool:

        transport_task = self.get_transport_task_with_id(unique_id)

        if transport_task is None:
            return False
        else:
            return True

    def get_transport_distance(self, source_machine_id: str, target_machine_id: str) -> float:
        return self._stationary_machine_distance_df.loc[source_machine_id, target_machine_id]

    def add_machine(self, machine: Machine) -> None:
        """
        Add a machine to the factory by adding it to the dictionaries.
        """

        # Add it in the registry by its unique identifier
        self._machine_by_id_dict[machine.unique_id] = machine

        if isinstance(machine, StorageMachine):
            self._storage_machine_by_id_dict[machine.unique_id] = machine
        elif isinstance(machine, ProcessingMachine):
            self._processing_machine_by_id_dict[machine.unique_id] = machine
        elif isinstance(machine, TransporterMachine):
            self._transport_machine_by_id_dict[machine.unique_id] = machine

        if isinstance(machine, StationaryMachine):
            self._stationary_machine_by_id_dict[machine.unique_id] = machine

            # Recalculate distances each time a machine is added
            self._calculate_stationary_machine_distances()

        # Add it to the list of machines per skill type based on the skills it offers
        for skill in machine.skill_set:

            if skill.type_name() in self._machines_by_skill_dict.keys():
                self._machines_by_skill_dict[skill.type_name()].add(machine)

            else:
                self._machines_by_skill_dict[skill.type_name()] = {machine}

    def connect(self, source_machine: Machine, target_machine: Machine, transport_machine: TransporterMachine) -> None:
        """
        Connects two machines using a transporter machine. Acts like an edge of a graph.
        :param source_machine: The source machine to connect to. (Machine)
        :param target_machine: The target machine to connect to. (Machine)
        :param transport_machine: The machine used for transporting a product (TransporterMachine)
        :return: None
        """

        if (source_machine.unique_id not in self._machine_by_id_dict
                or target_machine.unique_id not in self._machine_by_id_dict
                or transport_machine.unique_id not in self._machine_by_id_dict):

            raise ValueError("Machines must be added to the factory first.")

        if isinstance(source_machine, TransporterMachine) or isinstance(target_machine, TransporterMachine):
            raise TypeError("Nodes must be processing/storage machines.")

        if not isinstance(transport_machine, TransporterMachine):
            raise TypeError("Edge must be a transporter machine.")

        self._transport_connections.append((source_machine.unique_id, target_machine.unique_id, transport_machine.unique_id))

        # Create a transport task per connection and per transport machine
        transport_task = TransportTask(source_machine_id=source_machine.unique_id,
                                       target_machine_id=target_machine.unique_id,
                                       distance=self.get_transport_distance(source_machine_id=source_machine.unique_id,
                                                                            target_machine_id=target_machine.unique_id))

        if transport_machine.unique_id not in self._transport_task_list_by_transport_dict:
            self._transport_task_list_by_transport_dict[transport_machine.unique_id] = [transport_task]
        else:

            self._transport_task_list_by_transport_dict[transport_machine.unique_id].append(transport_task)

    def create_connections(self, transporter_machine: TransporterMachine, from_machine_list: list[Machine], to_machine_list: list[Machine]):

        # Create connections from specified list of machines to specified list of machines
        for from_machine in from_machine_list:

            for to_machine in to_machine_list:

                # Skip connections from a machine to itself
                if from_machine.unique_id == to_machine.unique_id:
                    continue

                # Connect (with all necessary checks that go with it)
                self.connect(source_machine=from_machine, target_machine=to_machine, transport_machine=transporter_machine)

    def _calculate_stationary_machine_distances(self) -> None:

        # Get all stationary machines to calculate their distances from each other
        stationary_machine_list = list(self._stationary_machine_by_id_dict.values())

        stationary_machine_pos_list = [[machine.x, machine.y] for machine in stationary_machine_list]
        stationary_machine_unique_id_list = [machine.unique_id for machine in stationary_machine_list]

        # Create a numpy array for efficiency
        machine_coordinates = np.array(stationary_machine_pos_list)

        # Compute pairwise distances (Hint: this was suggested by AI)
        diffs = machine_coordinates[:, np.newaxis, :] - machine_coordinates[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=-1)

        # Convert to DataFrame with labels
        self._stationary_machine_distance_df = pd.DataFrame(data=distances,
                                                            index=stationary_machine_unique_id_list,
                                                            columns=stationary_machine_unique_id_list)

    def create_digraph_stationary_machines(self) -> DiGraph:

        # Build a directed graph using only stationary machine machine_nodes
        factory_graph = nx.DiGraph()

        for src_id, tgt_id, transport in self._transport_connections:

            if (src_id in self._machine_by_id_dict and tgt_id in self._machine_by_id_dict and
                    not isinstance(self._machine_by_id_dict[src_id], TransporterMachine) and
                    not isinstance(self._machine_by_id_dict[tgt_id], TransporterMachine)):

                factory_graph.add_edge(src_id, tgt_id,
                                       distance=self._stationary_machine_distance_df.loc[src_id, tgt_id])

        for machine_key in self._machine_by_id_dict.keys():

            machine = self._machine_by_id_dict[machine_key]

            if isinstance(machine, StationaryMachine):

                machine_type_name = type(machine).__name__

                if machine_type_name in COLOR_BY_STATIONARY_MACHINE_DICT:
                    color = COLOR_BY_STATIONARY_MACHINE_DICT[machine_type_name]
                else:
                    color = "lightgrey"

                factory_graph.add_node(machine.unique_id, x=machine.x, y=machine.y, color=color)

        return factory_graph

    def get_machine_by_id(self, unique_id: str) -> Machine:
        """
        Get a machine by its unique ID.
        :param unique_id: Machine's unique ID. (str)
        :return: Machine
        """

        if unique_id not in self._machine_by_id_dict.keys():
            raise ValueError(f"Machine with ID {unique_id} not found in factory")

        return self._machine_by_id_dict[unique_id]

    def get_machines_by_skill_type(self, skill_type: Type[Skill]) -> set[Machine]:

        if skill_type.__name__ not in self._machines_by_skill_dict.keys():
            raise ValueError(f"No machines with {skill_type.__name__} not found in factory")

        return self._machines_by_skill_dict[skill_type.__name__]

    def get_neighbors(self, unique_id: str) -> list[tuple[str, str, str]]:
        """
        Return a list of (neighbor_machine, transport_machine) for outgoing edges from a machines unique ID.
        """

        result_list = []

        for (src_id, target_id, transport_id) in self._transport_connections:

            if src_id == unique_id:
                result_tuple = (self._machine_by_id_dict[target_id], self._machine_by_id_dict[transport_id])
                result_list.append(result_tuple)

        return result_list

    def __repr__(self):
        """
        Return a human-readable description of the factory connections.
        """
        repr_str = "Factory:\n"

        for src_id, target_id, transport in self._transport_connections:

            src = self.machine_by_id_dict[src_id]
            tgt = self._machine_by_id_dict[target_id]

            repr_str += (f"  {src.__class__.__name__}({src.unique_id}) → "
                         f"{tgt.__class__.__name__}({tgt.unique_id}) via "
                         f"{transport.__class__.__name__}({transport.unique_id})\n")

        return repr_str
