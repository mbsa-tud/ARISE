# -*- coding: utf-8 -*-

"""
Module defining the 'traditional' scheduler algorithm

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

import time
from copy import deepcopy

import networkx as nx

from networkx import MultiDiGraph
from networkx.drawing.nx_agraph import to_agraph

from src.arise_project.config.paths import DIR_DATA_OUTPUT_PATH
from src.arise_project.model.scenario import Scenario


class Scheduler:

    def __init__(self, scenario: Scenario):

        self._scenario = scenario
        self._product_state_graph = MultiDiGraph()

    def optimize(self):

        # Remember starting state for graph search
        starting_state_str = str(self._scenario.get_sorted_product_list()[0].current_state)

        print(f"\nFinding optimal processing plan for {self._scenario.get_sorted_product_list()[0]} "
              f"-> Target: {self._scenario.get_sorted_product_list()[0].target_state}\n")

        # Keep track of time for benchmarking and comparisons
        start_time = time.time()

        self._product_state_graph.add_node(str(self._scenario.get_sorted_product_list()[0].current_state))

        possible_actions = self._scenario.get_specific_actions()

        action_array = []

        min_time = 999_999_999
        min_time_actions = ()

        min_energy = 999_999_999
        min_energy_actions = ()

        min_time_energy_sum = 999_999_999
        min_time_energy_sum_actions = ()

        # TODO Improve this to accommodate any action sequence length

        for i, action in enumerate(possible_actions):

            new_sim_interface = deepcopy(self._scenario)

            new_sim_interface.execute_action(action)

            tripple_action_list = [action]

            for j, next_action in enumerate(new_sim_interface.get_specific_actions()):

                new_new_sim_interface = deepcopy(new_sim_interface)

                new_new_sim_interface.execute_action(next_action)

                tripple_action_list.append(next_action)

                for k, next_next_action in enumerate(new_new_sim_interface.get_specific_actions()):

                    new_new_new_sim_interface = deepcopy(new_new_sim_interface)
                    new_new_new_sim_interface.execute_action(next_next_action)

                    tripple_action_list.append(next_next_action)

                    if i == len(action_array):
                        action_array.append([])

                    if j == len(action_array[i]):
                        action_array[i].append([])

                    if k == len(action_array[i][j]):
                        action_array[i][j].append([])

                    # Keep track of current sequence of actions / task results / processing tasks
                    current_action_tuple = (action, next_action, next_next_action)

                    # Calculate total time and energy of this sequence
                    total_time = action.total_time + next_action.total_time + next_next_action.total_time
                    total_energy = action.total_energy + next_action.total_energy + next_next_action.total_energy

                    # Find the action sequence with the minimum total time
                    if total_time < min_time:
                        min_time = total_time
                        min_time_actions = current_action_tuple

                    # Find the action sequence with the minimum total energy consumption
                    if total_energy < min_energy:
                        min_energy = total_energy
                        min_energy_actions = current_action_tuple

                    # Find the action sequence with the minimum sum of total time and energy consumption
                    if (total_time + total_energy) < min_time_energy_sum:
                        min_time_energy_sum = (total_time + total_energy)
                        min_time_energy_sum_actions = current_action_tuple

                    # Keep track of all action tuples and their total time and energy cost
                    action_array[i][j][k] = [current_action_tuple, total_time, total_energy]

                    previous_state = None

                    for x, state_history_entry in enumerate(new_new_new_sim_interface.get_sorted_product_list()[0].state_history_list):

                        product_state = state_history_entry[0]

                        # Skip first, empty state
                        if x > 0:

                            self._product_state_graph.add_edge(str(previous_state), str(product_state),
                                                               key=f"{current_action_tuple[x - 1].get_short_name()}",
                                                               time=current_action_tuple[x - 1].total_time,
                                                               energy=current_action_tuple[x - 1].total_energy,
                                                               name=f"{current_action_tuple[x - 1].get_short_name()}",
                                                               params=f"{current_action_tuple[x - 1].get_short_name()}\n"
                                                                      f"T: {current_action_tuple[x - 1].total_time:.2f}, "
                                                                      f"E: {current_action_tuple[x - 1].total_energy:.2f}"
                                                               )

                        previous_state = product_state

        print("All possible actions considered. Graph created. Optimal path found during graph generation: \n")

        print(f"Minimum time: {min_time:.2f} -> {min_time_actions}")
        print(f"Minimum energy: {min_energy:.2f} -> {min_energy_actions}")
        print(f"Minimum time & energy: {min_time_energy_sum:.2f} -> {min_time_energy_sum_actions}")

        print(f"\nCompleted in {time.time() - start_time:.2f} seconds")

        # --- Time - Compute the shortest path from initial to target state using Dijkstra's algorithm
        min_time_path_nodes = nx.shortest_path(self._product_state_graph,
                                               source=starting_state_str,
                                               target=str(self._scenario.get_sorted_product_list()[0].target_state),
                                               weight='time')

        min_time_length = nx.shortest_path_length(self._product_state_graph,
                                                  source=starting_state_str,
                                                  target=str(self._scenario.get_sorted_product_list()[0].target_state),
                                                  weight='time')

        min_time_path_edges = []
        min_time_path_edges_weights = []

        for u, v in zip(min_time_path_nodes, min_time_path_nodes[1:]):

            # Select edge with minimum weight if multiple exist
            edge_key, edge_attr = min(self._product_state_graph[u][v].items(), key=lambda z: z[1].get("time", float('inf')))

            min_time_path_edges.append(edge_key)
            min_time_path_edges_weights.append((edge_key, edge_attr["time"]))

        print(f"\nMinimum time: {min_time_path_nodes} [{min_time_length:.3f}]")
        print(f"Specifically: {min_time_path_edges_weights} ")

        # --- Energy - Compute the shortest path from initial to target state using Dijkstra's algorithm
        min_energy_path_nodes = nx.shortest_path(self._product_state_graph,
                                                 source=starting_state_str,
                                                 target=str(self._scenario.get_sorted_product_list()[0].target_state),
                                                 weight='energy')

        min_energy_length = nx.shortest_path_length(self._product_state_graph,
                                                    source=starting_state_str,
                                                    target=str(self._scenario.get_sorted_product_list()[0].target_state),
                                                    weight='energy')

        min_energy_path_edges = []
        min_energy_path_edges_weights = []

        for u, v in zip(min_energy_path_nodes, min_energy_path_nodes[1:]):

            # Select edge with minimum weight if multiple exist
            edge_key, edge_attr = min(self._product_state_graph[u][v].items(), key=lambda z: z[1].get("energy", float('inf')))

            min_energy_path_edges.append(edge_key)
            min_energy_path_edges_weights.append((edge_key, edge_attr["energy"]))

        print(f"\nMinimum energy: {min_energy_path_nodes} [{min_energy_length:.3f}]")
        print(f"Specifically: {min_energy_path_edges_weights} ")

        # For multicriteria optimization create a new weight from other weights
        for u, v, k, data in self._product_state_graph.edges(keys=True, data=True):
            data['time_energy_sum'] = data['time'] + data['energy']

        # --- Combined (Time + Energy) - Compute the shortest path from initial to target state using Dijkstra's algorithm
        min_time_energy_sum_path_nodes = nx.shortest_path(self._product_state_graph,
                                                          source=starting_state_str,
                                                          target=str(self._scenario.get_sorted_product_list()[0].target_state),
                                                          weight='time_energy_sum')

        min_time_energy_sum_length = nx.shortest_path_length(self._product_state_graph,
                                                             source=starting_state_str,
                                                             target=str(self._scenario.get_sorted_product_list()[0].target_state),
                                                             weight='time_energy_sum')

        min_time_energy_sum_path_edges = []
        min_time_energy_sum_path_edges_weights = []

        for u, v in zip(min_time_energy_sum_path_nodes, min_time_energy_sum_path_nodes[1:]):

            # Select edge with minimum weight if multiple exist
            edge_key, edge_attr = min(self._product_state_graph[u][v].items(), key=lambda z: z[1].get("time_energy_sum", float('inf')))

            min_time_energy_sum_path_edges.append(edge_key)
            min_time_energy_sum_path_edges_weights.append((edge_key, edge_attr["time_energy_sum"]))

        print(f"\nMinimum time & energy sum: {min_time_energy_sum_path_nodes} [{min_time_energy_sum_length:.3f}]")
        print(f"Specifically: {min_time_energy_sum_path_edges_weights} ")

        # Convert to AGraph (PyGraphviz)
        A = to_agraph(self._product_state_graph)

        # Global attributes
        A.graph_attr.update(dpi="300", size="10,10")
        A.node_attr.update(style="filled", fillcolor="lightblue", fontname="Arial")
        A.edge_attr.update(fontname="Arial")

        # Set individual node attributes (color, shape, size)
        for node in self._product_state_graph.nodes(data=True):
            n = A.get_node(node[0])
            n.attr['color'] = "black"
            n.attr['style'] = 'filled'
            n.attr['fillcolor'] = "lightblue"
            n.attr['fontsize'] = 12

        # Set individual edge labels
        for u, v, k, data in self._product_state_graph.edges(keys=True, data=True):

            e = A.get_edge(u, v, k)
            e.attr['label'] = data['params']
            e.attr['fontsize'] = 6

        idx = 0

        # Color the edges associated with the shortest path green (for one specific weight)
        for u, v in zip(min_time_energy_sum_path_nodes, min_time_energy_sum_path_nodes[1:]):

            e = A.get_edge(u, v, min_time_energy_sum_path_edges[idx])
            e.attr['color'] = "forestgreen"
            e.attr['fontcolor'] = "forestgreen"

            idx += 1

        A.layout(prog='dot')
        A.draw(DIR_DATA_OUTPUT_PATH / "scheduler_product_state_graph.svg")

        print("\nDrawing processing state graph completed. Saved image to disk.")
