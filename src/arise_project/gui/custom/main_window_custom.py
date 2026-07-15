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

Module defining the main window GUI based on a class generated from a '.ui' file

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import time
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from PyQt6 import QtGui
from PyQt6.QtGui import QBrush, QColor, QPixmap
from PyQt6.QtWidgets import QTreeWidgetItem, QGroupBox, QVBoxLayout, QTreeWidget, QTableWidget, QTableWidgetItem, \
    QHeaderView, QFileDialog, QDialog
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

from src.arise_project.model.nsga_config import NSGAConfig
from src.arise_project.scheduler.llm_scheduler import run_iterative_llm_scheduler, OPT_RES_PARAM_TOTAL_TOKEN_COUNT, \
    OPT_RES_PARAM_AVG_RESPONSE_TIME
from src.arise_project.gui.custom.plots import AnalysisPlot, COL_TASKS, COL_A_STAR_TIME, COL_NSGA2_TIME, \
    COL_RL_DQN_TIME, COL_NSGA2_RATIO, COL_RL_DQN_RATIO
from src.arise_project.model.product import Plate
from src.arise_project.config.debug import DEBUG_MODE
from src.arise_project.gui.custom.pyqt_log_stream import PyQtLogStream
from src.arise_project.gui.custom.pyqt_progress_updater import PyQtProgressUpdater, DummyProgressUpdater
from src.arise_project.gui.custom.thread_manager import ThreadManager
from src.arise_project.model.machines import AutomatedGuidedVehicle, ConveyorBelt, ThreeAxisRobot
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.optimization_result import OptimizationResult
from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.cost_normalization import compute_cost_scales
from src.arise_project.tools.energy_format import joules_to_wh, joules_to_kwh
from src.arise_project.scheduler.a_star_search import astar_search, OPT_RES_PARAM_EXPANSIONS
from src.arise_project.scheduler.depth_first_search import run_iddfs, OPT_RES_PARAM_MIN_SOLUTION_DEPTH
from src.arise_project.scheduler.genetic_algorithms import run_nsga, OPT_RES_PARAM_HYPERVOLUME, \
    OPT_RES_PARAM_NUM_TRIALS

if not DEBUG_MODE:
    from src.arise_project.scheduler.factory_dqn_training import (run_inference, run_training, OPT_RES_PARAM_REWARD)

else:
    # TODO temporary work around
    OPT_RES_PARAM_REWARD = "reward"

from src.arise_project.tools.duration_format import duration_formatting
from src.arise_project.tools.hash_generation import get_scenario_data_dir_path
from src.arise_project.tools.output_timestamp import print_with_timestamp

from src.arise_project.config.colors import COLOR_BY_SKILL_DICT
from src.arise_project.config.paths import DIR_DATA_INPUT_ALL_SCENARIO_DIRS_PATH, FILE_GUI_ICON_PATH, \
    FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, DIR_DATA_INPUT_SC_DIR_ANALYSIS_TASK_COUNT

from src.arise_project.gui.generated.main_window_generated import Ui_MainWindow
from src.arise_project.gui.generated.about_dialog_generated import Ui_Dialog

from src.arise_project.model.machines import StorageMachine, \
    ProcessingMachine, TransporterMachine, MillingMachine, CuttingMachine, DrillingMachine

from src.arise_project.model.scenario import Scenario

WINDOW_TITLE_BASIS = f"ICM ARISE - Factory Simulation"

COL_SCENARIO_OVERVIEW_NAME = "Name"
COL_SCENARIO_OVERVIEW_PRODUCTS = "Products"
COL_SCENARIO_OVERVIEW_TASKS = "PT"
COL_SCENARIO_OVERVIEW_PROC_SKILLS = "PS"
COL_SCENARIO_OVERVIEW_MACHINES = "Machines"
COL_SCENARIO_OVERVIEW_STORAGE_MACHINES = "SM"
COL_SCENARIO_OVERVIEW_PROCESSING_MACHINES = "PM"
COL_SCENARIO_OVERVIEW_TRANSPORT_MACHINES = "TM"
COL_SCENARIO_OVERVIEW_CONNECTIONS = "Connections"
COL_SCENARIO_OVERVIEW_STATE_COUNT = "States"
COL_SCENARIO_OVERVIEW_TRANSITION_COUNT = "Transitions"
COL_SCENARIO_OVERVIEW_ACTION_CATALOG_COUNT = "AC"

COL_SIM_ACTIONS_PRODUCT = "Product"
COL_SIM_ACTIONS_TASK = "Task"
COL_SIM_ACTIONS_MACHINE = "Machine"
COL_SIM_ACTIONS_SKILL = "Skill"
COL_SIM_ACTIONS_SKILL_TYPE = "Skill Type"
COL_SIM_ACTIONS_TIME = "Time (s)"
COL_SIM_ACTIONS_ENERGY = "Energy (Wh)"
COL_SIM_ACTIONS_RELIABILITY = "Reliability"
COL_SIM_ACTIONS_NOTE = "Note"

COL_OPT_RES_COMP_METHOD = "Method"
COL_OPT_RES_COMP_STEPS = "Steps"
COL_OPT_RES_COMP_TOTAL_TIME = "Total Time (s)"
COL_OPT_RES_COMP_TOTAL_ENERGY = "Total Energy (kWh)"
COL_OPT_RES_COMP_SEQUENCE_RELIABILITY = "Seq. Reliability"
COL_OPT_RES_COMP_TOTAL_COST = "Total Cost"
COL_OPT_RES_COMP_DURATION_SECONDS = "Duration"


class Ui_MainWindow_Custom(Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self._window = None
        self._pyqt_log_stream = PyQtLogStream()

        self._active_sc_directory = DIR_DATA_INPUT_SC_DIR_ANALYSIS_TASK_COUNT # DIR_DATA_INPUT_SCENARIOS_JSON_PATH

        self._sim_started = False
        self._sim_start_time = time.time()
        self._sim_action_idx_sequence = []

        # Undone actions, kept so the Edit > Redo menu can re-execute them. Each entry is the
        # (task_result, action_idx) pair that was removed by the last undo.
        self._sim_redo_stack = []

        self._factory_graph_fig = None
        self._factory_graph_canvas = None
        self._factory_graph_ax = None
        self._factory_graph_toolbar = None

        self._loaded_scenario_list: list[Scenario] = []
        self._active_scenario_idx = 0
        self._selected_scenario_idx = self._active_scenario_idx

        self._current_task_result_list = []
        self._current_action_idx_list = []
        self._selected_task_result = None
        self._selected_action_idx = 0

        # TODO Refactor / configuration
        use_reliability = True

        if use_reliability:

            self._time_weight = 1 / 3
            self._energy_weight = 1 / 3
            self._reliability_weight = 1 / 3

        else:

            self._time_weight = 1 / 2
            self._energy_weight = 1 / 2
            self._reliability_weight = 0

        # Scale is recomputed per-scenario in _initialize_scenario(); no scenario is loaded yet here
        self._objective_function = ObjectiveFunction(time_weight=self._time_weight,
                                                     energy_weight=self._energy_weight,
                                                     reliability_weight=self._reliability_weight)

        self._thread_manager = ThreadManager()


    def setupUi(self, MainWindow):

        super().setupUi(MainWindow)

        self._window = MainWindow

        # Set window icon
        MainWindow.setWindowIcon(QtGui.QIcon(str(FILE_GUI_ICON_PATH)))

        self._pyqt_log_stream.msg_signal.connect(self._on_log_stream_message)
        sys.stdout = self._pyqt_log_stream

        self.checkBox_factory_show_distances.stateChanged.connect(self.on_change_checkbox_factory_distances)

        self.tableWidget_sim_actions.cellClicked.connect(self.on_cell_selected_sim_actions)
        self.tableWidget_scenario_overview.cellClicked.connect(self.on_cell_selected_scenario_overview)

        self.pushButton_scenario_overview_reload.clicked.connect(self.on_click_scenario_overview_reload)
        self.pushButton_scenario_load.clicked.connect(self.on_click_scenario_load)

        self.pushButton_sim_execute_action.clicked.connect(self.on_click_sim_execute_action)
        self.pushButton_sim_undo_last_action.clicked.connect(self.on_click_sim_undo_last_action)
        self.pushButton_sim_reset_scenario.clicked.connect(self.on_click_sim_reset_scenario)
        self.pushButton_sim_save_results.clicked.connect(self.on_click_sim_save_results)

        self.pushButton_opt_a_star_run.clicked.connect(self.on_click_opt_a_star_run)
        self.pushButton_opt_dijkstra_run.clicked.connect(self.on_click_opt_dijkstra_run)
        self.pushButton_opt_dfs_run.clicked.connect(self.on_click_opt_dfs_run)
        self.pushButton_opt_iddfs_run.clicked.connect(self.on_click_opt_iddfs_run)
        self.pushButton_opt_nsga2_run.clicked.connect(self.on_click_opt_nsga2_run)
        self.pushButton_opt_nsga3_run.clicked.connect(self.on_click_opt_nsga3_run)
        self.pushButton_opt_rl_dqn_run_training_and_inference.clicked.connect(self.on_click_opt_rl_dqn_training_and_inference_run)
        self.pushButton_opt_llm_agent_run_prompt.clicked.connect(self.on_click_opt_llm_agent_prompt_run)
        self.pushButton_opt_human_sim.clicked.connect(self.on_click_opt_human_sim)

        self.pushButton_opt_auto_run.clicked.connect(self.on_click_opt_auto_run)

        self.action_switch_scenario_directory.triggered.connect(self.on_action_switch_scenario_directory)
        self.action_about.triggered.connect(self.on_action_show_about_dialog)
        self.action_export_analysis.triggered.connect(self.on_action_export_analysis_data)

        # Edit > Undo mirrors the simulation-tab undo button; Edit > Redo re-executes the last undone action
        self.action_undo.triggered.connect(self.on_click_sim_undo_last_action)
        self.action_redo.triggered.connect(self.on_click_sim_redo_last_action)
        self.action_undo.setEnabled(False)
        self.action_redo.setEnabled(False)

        self._analysis_plot = AnalysisPlot(self.groupBox_analysis)

        default_nsga_config = NSGAConfig()

        self.spinBox_opt_nsga2_max_sequence_length.setValue(default_nsga_config.max_sequence_length)
        self.spinBox_opt_nsga2_population.setValue(default_nsga_config.population_size)
        self.spinBox_opt_nsga2_generations.setValue(default_nsga_config.number_generations)
        self.doubleSpinBox_opt_nsga2_mutation_probability.setValue(default_nsga_config.mutation_probability * 100)

        self.spinBox_opt_nsga2_max_sequence_length.setValue(default_nsga_config.max_sequence_length)
        self.spinBox_opt_nsga2_population.setValue(default_nsga_config.population_size)
        self.spinBox_opt_nsga2_generations.setValue(default_nsga_config.number_generations)
        self.doubleSpinBox_opt_nsga2_mutation_probability.setValue(default_nsga_config.mutation_probability * 100)

        self._init_graph_in_groupbox()

        self._load_all_scenarios(progress_updater=DummyProgressUpdater())
        self._initialize_scenario()

    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)

    def _load_all_scenarios(self, progress_updater: PyQtProgressUpdater) -> None:

        progress_updater.percentage = 0
        progress_updater.text = ""

        self._loaded_scenario_list: list[Scenario] = []
        self._active_scenario_idx = 0

        found_scenario_dir_list = list(self._active_sc_directory.iterdir())
        counter = 0

        for directory_path in found_scenario_dir_list:

            counter += 1

            progress_updater.percentage = int(round(counter / len(found_scenario_dir_list) * 100, 0))
            progress_updater.text = f"Loading scenario {counter} of {len(found_scenario_dir_list)}"

            if not directory_path.is_dir():
                continue

            scenario_file_path = directory_path / f"{directory_path.name}.json"

            if scenario_file_path.exists():
                scenario = Scenario(file_path=scenario_file_path, reset_class=True)
                self._loaded_scenario_list.append(scenario)

        progress_updater.percentage = 100
        progress_updater.text = "Done."
        progress_updater.finish()

    @staticmethod
    def _directory_contains_scenarios(directory_path: Path) -> bool:
        """Return True if the directory holds at least one scenario (a subdir with a matching JSON)."""

        if not directory_path.is_dir():
            return False

        for sub_directory_path in directory_path.iterdir():

            if not sub_directory_path.is_dir():
                continue

            if (sub_directory_path / f"{sub_directory_path.name}.json").exists():
                return True

        return False

    def _initialize_scenario(self):

        self.plainTextEdit_log_output.clear()

        self._sim_started = False
        self._sim_start_time = time.time()
        self._sim_action_idx_sequence = []
        self._sim_redo_stack = []

        self._current_task_result_list = []
        self._current_action_idx_list = []
        self._selected_task_result = None
        self._selected_action_idx = 0

        self._thread_manager = ThreadManager()

        # Reset GUI
        for opt_method in OptimizationMethod:
            self._update_labels_opt_results(opt_method=opt_method, reset=True)

        self.tableWidget_sim_actions.clear()

        # Reset every optimization result table. dijkstra and llm_agent were previously missing
        # here, so their contents lingered from a different scenario after loading a new one.
        opt_result_tables = [self.tableWidget_opt_a_star_results,
                             self.tableWidget_opt_dijkstra_results,
                             self.tableWidget_opt_dfs_results,
                             self.tableWidget_opt_iddfs_results,
                             self.tableWidget_opt_nsga2_results,
                             self.tableWidget_opt_nsga3_results,
                             self.tableWidget_opt_rl_dqn_results,
                             self.tableWidget_opt_llm_agent_results,
                             self.tableWidget_opt_human_results,
                             self.tableWidget_opt_comparison]

        for opt_result_table in opt_result_tables:
            opt_result_table.clear()
            opt_result_table.setRowCount(0)
            opt_result_table.setColumnCount(0)

        self.pushButton_sim_execute_action.setEnabled(False)
        self.pushButton_sim_undo_last_action.setEnabled(False)
        self.pushButton_sim_reset_scenario.setEnabled(False)
        self.pushButton_sim_save_results.setEnabled(False)

        self.action_undo.setEnabled(False)
        self.action_redo.setEnabled(False)

        self._update_tree_widget_factory()
        self._update_tree_widget_product()
        self._update_tree_widget_state()
        self._update_table_widget_sim_actions()

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        # Cost scale is scenario-specific, so the objective function is rebuilt for whichever
        # scenario just became active (weights stay fixed, only time_scale/energy_scale change)
        time_scale, energy_scale, reliability_scale = compute_cost_scales(active_scenario)
        self._objective_function = ObjectiveFunction(time_weight=self._time_weight,
                                                     energy_weight=self._energy_weight,
                                                     reliability_weight=self._reliability_weight,
                                                     time_scale=time_scale,
                                                     energy_scale=energy_scale,
                                                     reliability_scale=reliability_scale)

        self._window.setWindowTitle(f"{WINDOW_TITLE_BASIS} - Scenario: {active_scenario.name}")

        self.label_sim_total_steps.setText(f"{active_scenario.step_count}")
        self.label_sim_total_time.setText(f"{active_scenario.time_sum:.2f} s")
        self.label_sim_total_energy.setText(f"{joules_to_kwh(active_scenario.energy_sum):.4f} kWh")
        self.label_sim_sequence_reliability.setText(f"{active_scenario.sequence_reliability:.3f}")
        self.label_sim_total_cost.setText(f"{active_scenario.calculate_total_cost(self._objective_function):.2f}")

        self._draw_graph_in_groupbox(active_scenario.factory.create_digraph_stationary_machines(),
                                     labels=True,
                                     edge_labels=self.checkBox_factory_show_distances.isChecked(),
                                     node_size=800)

        for opt_method in active_scenario.opt_result_dict:

            if isinstance(active_scenario.opt_result_dict[opt_method], OptimizationResult):

                match opt_method:

                    case OptimizationMethod.OPT_A_STAR:
                        self._on_a_star_complete()
                    case OptimizationMethod.OPT_DIJKSTRA:
                        self._on_dijkstra_complete()
                    case OptimizationMethod.OPT_DFS:
                        self._on_dfs_complete()
                    case OptimizationMethod.OPT_IDDFS:
                        self._on_iddfs_complete()
                    case OptimizationMethod.OPT_NSGA2:
                        self._on_nsga2_complete()
                    case OptimizationMethod.OPT_NSGA3:
                        self._on_nsga3_complete()
                    case OptimizationMethod.OPT_RL_DQN:
                        self._on_rl_dqn_training_and_inference_complete()
                    case OptimizationMethod.OPT_LLM_AGENT:
                        self._on_llm_agent_prompt_complete()
                    case OptimizationMethod.OPT_HUMAN:
                        self._update_opt_result(OptimizationMethod.OPT_HUMAN)

        self._update_table_widget_opt_comparison()
        self._update_product_image()
        self._update_table_widget_scenario_overview()

        self._update_analysis_plots()

        print_with_timestamp(f"Initialization of '{active_scenario.name}' complete...")

    def _on_log_stream_message(self, msg_str: str):
        self.plainTextEdit_log_output.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.plainTextEdit_log_output.insertPlainText(msg_str)

    def _init_graph_in_groupbox(self) -> None:

        layout = QVBoxLayout(self.groupBox_factory_plot)
        self.groupBox_factory_plot.setLayout(layout)

        self._factory_graph_fig = Figure(figsize=(5, 4), dpi=100)
        self._factory_graph_canvas = FigureCanvas(self._factory_graph_fig)
        self._factory_graph_ax = self._factory_graph_fig.add_subplot(111)
        self._factory_graph_toolbar = NavigationToolbar(self._factory_graph_canvas, self.groupBox_factory_plot)

        # Add toolbar and canvas to the group box
        layout.addWidget(self._factory_graph_toolbar)
        layout.addWidget(self._factory_graph_canvas)

    def _draw_graph_in_groupbox(self, graph: nx.Graph,
                                labels: bool = True,
                                edge_labels: bool = True,
                                node_size: int = 600) -> None:
        """
        Draw the provided graph in the factory graph groupbox. (Hint: Code partly AI generated)
        :param graph: networkx graph (nx.Graph)
        :param labels: True if labels should be displayed (bool)
        :param node_size: Size of nodes (int)
        :return: None
        """

        pos_dict = {}

        for node, data_dict in graph.nodes(data=True):

            if 'x' in data_dict and 'y' in data_dict:
                pos_dict[node] = (data_dict['x'], data_dict['y'])

            else:
                raise ValueError(f"Node {node} has no x or y position")

        self._factory_graph_ax.clear()

        # Edges
        nx.draw_networkx_edges(
            graph, pos_dict, ax=self._factory_graph_ax,
            arrows=graph.is_directed(),
            arrowstyle="-|>",
            arrowsize=14,
            width=1.2,
            connectionstyle="arc3,rad=0.0",
            edge_color="#888"
        )

        # Extract node colors from attributes
        node_colors = [graph.nodes[node]["color"] for node in graph.nodes]

        # Nodes
        nx.draw_networkx_nodes(
            graph, pos_dict, ax=self._factory_graph_ax,
            node_size=node_size,
            node_color=node_colors,
            edgecolors="white",
            linewidths=0.0,
        )

        if labels:

            nx.draw_networkx_labels(graph, pos_dict, ax=self._factory_graph_ax, font_size=9, font_color="white")

        if edge_labels:

            # Get labels from edge data and format the values
            edge_labels = {edge: f"{data['distance']:.3f}" for edge, data in graph.edges.items()}
            nx.draw_networkx_edge_labels(graph, pos_dict, ax=self._factory_graph_ax, edge_labels=edge_labels)

        # Keep your coordinate system
        self._factory_graph_ax.set_aspect("equal")
        self._factory_graph_ax.axis("off")
        self._factory_graph_fig.tight_layout()
        self._factory_graph_canvas.draw_idle()

    def _update_product_image(self) -> None:

        display_product = self._loaded_scenario_list[self._active_scenario_idx].get_sorted_product_list()[0]

        qimg = display_product.render_q_image()
        pix = QPixmap.fromImage(qimg)

        if isinstance(display_product, Plate):

            scaled_width = int(display_product.width * 4)
            scaled_height = int(display_product.height * 4)

            # 🔍 Scale while keeping sharp pixels
            pix_scaled = pix.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.FastTransformation)

            self.label_product_image.setPixmap(pix_scaled)

        else:
            raise NotImplementedError(f"Product image rendering for type '{type(display_product)}' is not implemented.")

    def _update_tree_widget_factory(self) -> None:
        """
        Function used to populate the tree widget containing a factory definition
        :return: None
        """

        self.treeWidget_factory.clear()

        proc_machine_child_item = QTreeWidgetItem(self.treeWidget_factory)
        proc_machine_child_item.setText(0, "Processing Machine")
        proc_machine_child_item.setExpanded(True)

        storage_machine_child_item = QTreeWidgetItem(self.treeWidget_factory)
        storage_machine_child_item.setText(0, "Storage Machine")
        storage_machine_child_item.setExpanded(True)

        transporter_machine_child_item = QTreeWidgetItem(self.treeWidget_factory)
        transporter_machine_child_item.setText(0, "Transporter Machine")
        transporter_machine_child_item.setExpanded(True)

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        for machine_key in active_scenario.factory.machine_by_id_dict.keys():

            machine = active_scenario.factory.machine_by_id_dict[machine_key]

            if isinstance(machine, ProcessingMachine):
                machine_child_item = QTreeWidgetItem(proc_machine_child_item)

            elif isinstance(machine, StorageMachine):
                machine_child_item = QTreeWidgetItem(storage_machine_child_item)

            elif isinstance(machine, TransporterMachine):
                machine_child_item = QTreeWidgetItem(transporter_machine_child_item)

            else:
                raise ValueError(
                    f"Machine {machine_key} is not a ProcessingMachine, StorageMachine, TransporterMachine")

            machine_child_item.setText(0, f"{machine.unique_id} [{type(machine).__name__}]")

            params_child_item = QTreeWidgetItem(machine_child_item)
            params_child_item.setText(0, "Params")

            if isinstance(machine, ProcessingMachine) or isinstance(machine, StorageMachine):
                position_param_item = QTreeWidgetItem(params_child_item)
                position_param_item.setText(0, f"Position: {machine.x:.2f} m, {machine.y:.2f} m")

            skills_child_item = QTreeWidgetItem(machine_child_item)
            skills_child_item.setText(0, "Skills")

            for skill_key in machine.skill_by_id_dict.keys():
                skill = machine.skill_by_id_dict[skill_key]

                skill_item = QTreeWidgetItem(skills_child_item)
                skill_item.setText(0, f"{skill.unique_id} [{type(skill).__name__}]")

                execution_speed_item = QTreeWidgetItem(skill_item)
                execution_speed_item.setText(0, f"Execution speed: {skill.execution_speed:.2f}")

                nominal_power_draw_item = QTreeWidgetItem(skill_item)
                nominal_power_draw_item.setText(0, f"Nominal power draw: {skill.nominal_power_draw:.2f} W")

                reliability_item = QTreeWidgetItem(skill_item)
                reliability_item.setText(0, f"Reliability: {skill.reliability:.3f}")

    def _update_tree_widget_product(self) -> None:
        """
        Function used to populate the tree widget containing a product definition
        :return: None
        """

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        first_product = active_scenario.get_sorted_product_list()[0]

        self.treeWidget_product.clear()

        product_child_item = QTreeWidgetItem(self.treeWidget_product)
        product_child_item.setText(0, "Product")
        product_child_item.setExpanded(True)

        specific_product_child_item = QTreeWidgetItem(product_child_item)
        specific_product_child_item.setText(0, f"{first_product.unique_id} "
                                               f"[{type(first_product).__name__}]")
        specific_product_child_item.setExpanded(True)

        count_child_item = QTreeWidgetItem(specific_product_child_item)
        count_child_item.setText(0, f"Count: {len(active_scenario.get_sorted_product_list())}")

        params_child_item = QTreeWidgetItem(specific_product_child_item)
        params_child_item.setText(0, "Params")

        product_params_dict = first_product.get_params_dict()

        for product_param_key in product_params_dict.keys():
            product_param_child_item = QTreeWidgetItem(params_child_item)
            product_param_child_item.setText(0, f"{product_param_key}: {product_params_dict[product_param_key]}")

        processing_tasks_child_item = QTreeWidgetItem(specific_product_child_item)
        processing_tasks_child_item.setText(0, "Processing")
        processing_tasks_child_item.setExpanded(True)

        for processing_task in first_product.target_state.processing_tasks:

            processing_task_child_item = QTreeWidgetItem(processing_tasks_child_item)
            processing_task_child_item.setText(0, f"{processing_task.unique_id} [{type(processing_task).__name__}]")

            processing_task_preconditions_child_item = QTreeWidgetItem(processing_task_child_item)
            processing_task_preconditions_child_item.setText(0, "Preconditions")

            for precondition_task_id in processing_task.precondition_completed_task_id_set:
                processing_task_precondition_child_item = QTreeWidgetItem(processing_task_preconditions_child_item)
                processing_task_precondition_child_item.setText(0, f"Task: {precondition_task_id}")

            processing_task_params_child_item = QTreeWidgetItem(processing_task_child_item)
            processing_task_params_child_item.setText(0, "Params")

            params_dict = processing_task.get_params_dict()

            for param_key in params_dict.keys():
                processing_task_param_child_item = QTreeWidgetItem(processing_task_params_child_item)

                param_value_str = str(params_dict[param_key])

                if isinstance(params_dict[param_key], float):
                    param_value_str = f"{params_dict[param_key]:.2f}"

                processing_task_param_child_item.setText(0, f"{param_key}: {param_value_str}")

            skills_child_item = QTreeWidgetItem(processing_task_child_item)
            skills_child_item.setText(0, "Skills")

            for skill_type in processing_task.possible_skill_types:
                skill_item = QTreeWidgetItem(skills_child_item)
                skill_item.setText(0, f"{skill_type.__name__}")

    def _update_tree_widget_state(self) -> None:

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        self.treeWidget_sim_states.clear()

        product_child_item = QTreeWidgetItem(self.treeWidget_sim_states)
        product_child_item.setText(0, "Product State")
        product_child_item.setExpanded(True)

        for product in active_scenario.get_sorted_product_list():

            specific_product_machine_child_item = QTreeWidgetItem(product_child_item)
            specific_product_machine_child_item.setText(0, f"{product.unique_id} "
                                                           f"[{type(product).__name__}]")
            specific_product_machine_child_item.setExpanded(True)

            location_child_item = QTreeWidgetItem(specific_product_machine_child_item)
            location_child_item.setText(0, f"Location: {product.current_state.location_machine_id}")
            location_child_item.setExpanded(True)

            is_done_child_item = QTreeWidgetItem(specific_product_machine_child_item)
            is_done_child_item.setText(0, f"Done: {product.is_done()}")

            if product.is_done():
                is_done_child_item.setForeground(0, QBrush(QColor("forestgreen")))
            else:
                is_done_child_item.setForeground(0, QBrush(QColor("Crimson")))

            is_done_child_item.setExpanded(True)

            completed_tasks_child_item = QTreeWidgetItem(specific_product_machine_child_item)
            completed_tasks_child_item.setText(0, "Completed Tasks")
            completed_tasks_child_item.setExpanded(True)

            if len(product.current_state.processing_tasks) > 0:

                for processing_task in product.current_state.processing_tasks:

                    processing_task_child_item = QTreeWidgetItem(completed_tasks_child_item)
                    processing_task_child_item.setText(0, f"{processing_task.unique_id} [{type(processing_task).__name__}]")

                    processing_task_params_child_item = QTreeWidgetItem(processing_task_child_item)
                    processing_task_params_child_item.setText(0, "Params")

                    params_dict = processing_task.get_params_dict()

                    for param_key in params_dict.keys():
                        processing_task_param_child_item = QTreeWidgetItem(processing_task_params_child_item)
                        processing_task_param_child_item.setText(0, f"{param_key}: {params_dict[param_key]}")

                    skills_child_item = QTreeWidgetItem(processing_task_child_item)
                    skills_child_item.setText(0, "Skills")

                    for skill_type in processing_task.possible_skill_types:
                        skill_item = QTreeWidgetItem(skills_child_item)
                        skill_item.setText(0, f"{skill_type.__name__}")

            else:

                processing_task_child_item = QTreeWidgetItem(completed_tasks_child_item)
                processing_task_child_item.setText(0, f"None")

        action_history_child_item = QTreeWidgetItem(self.treeWidget_sim_states)
        action_history_child_item.setText(0, "Action History")
        action_history_child_item.setExpanded(True)

        if len(active_scenario.task_result_history) > 0:

            for idx, task_result in enumerate(active_scenario.task_result_history):

                specific_action_child_item = QTreeWidgetItem(action_history_child_item)
                specific_action_child_item.setText(0, f"{idx + 1}. {task_result.get_short_name(with_product=True, with_machine=True)}")

                task_result_time_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_time_child_item.setText(0, f"Time: {task_result.total_time:.2f} s")

                task_result_energy_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_energy_child_item.setText(0, f"Energy: {joules_to_wh(task_result.total_energy):.3f} Wh")

                task_result_reliability_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_reliability_child_item.setText(0, f"Reliability: {task_result.skill.reliability:.2f}")

        else:

            specific_action_child_item = QTreeWidgetItem(action_history_child_item)
            specific_action_child_item.setText(0, f"None")

    def _update_table_widget_sim_actions(self) -> None:
        """
        Function used to populate the table widget with all actions available at this step
        :return: None
        """

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        self._current_task_result_list = active_scenario.get_actions()
        self._current_action_idx_list = active_scenario.get_feasible_actions_idx_list()

        action_data_dict = [{COL_SIM_ACTIONS_PRODUCT: task_result.product.unique_id,
                             COL_SIM_ACTIONS_TASK: task_result.task.unique_id,
                             COL_SIM_ACTIONS_MACHINE: task_result.machine.unique_id,
                             COL_SIM_ACTIONS_SKILL: f"{task_result.skill.unique_id}",
                             COL_SIM_ACTIONS_SKILL_TYPE: task_result.skill.type_name(),
                             COL_SIM_ACTIONS_TIME: task_result.total_time,
                             COL_SIM_ACTIONS_ENERGY: joules_to_wh(task_result.total_energy),
                             COL_SIM_ACTIONS_RELIABILITY: task_result.skill.reliability,
                             COL_SIM_ACTIONS_NOTE: task_result.task.get_description_short()}

                            for task_result in self._current_task_result_list]

        actions_df = pd.DataFrame(data=action_data_dict)
        actions_df.index = self._current_action_idx_list

        self.tableWidget_sim_actions.clear()

        # Reset sorting after update
        self.tableWidget_sim_actions.setSortingEnabled(False)
        self.tableWidget_sim_actions.setSortingEnabled(True)
        self.tableWidget_sim_actions.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

        self.tableWidget_sim_actions.setRowCount(len(actions_df))
        self.tableWidget_sim_actions.setColumnCount(len(actions_df.columns))
        self.tableWidget_sim_actions.setHorizontalHeaderLabels(actions_df.columns.tolist())
        self.tableWidget_sim_actions.setVerticalHeaderLabels([str(i) for i in actions_df.index])

        for row in range(len(actions_df)):

            for col in range(len(actions_df.columns)):

                col_name = actions_df.columns[col]

                item = QTableWidgetItem()

                # Set data for sorting
                item.setData(Qt.ItemDataRole.DisplayRole, actions_df.iat[row, col])

                if (col_name == COL_SIM_ACTIONS_TIME) or (col_name == COL_SIM_ACTIONS_ENERGY):
                    item.setText(f"{actions_df.iat[row, col]:.2f}")

                elif col_name == COL_SIM_ACTIONS_RELIABILITY:
                    item.setText(f"{actions_df.iat[row, col]:.3f}")

                else:
                    item.setText(f"{actions_df.iat[row, col]}")

                # Center text
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if str(actions_df.iat[row, col]) in COLOR_BY_SKILL_DICT:
                    item.setForeground(QBrush(QColor(COLOR_BY_SKILL_DICT[str(actions_df.iat[row, col])])))

                self.tableWidget_sim_actions.setItem(row, col, item)

        header = self.tableWidget_sim_actions.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Make text in header bold
        font = header.font()
        font.setBold(True)
        header.setFont(font)

    def _update_table_widget_opt_results(self, opt_method: OptimizationMethod) -> None:
        """
        Function used to populate the table widget with all actions available at this step
        :return: None
        """

        match opt_method:

            case OptimizationMethod.OPT_A_STAR:
                table_widget = self.tableWidget_opt_a_star_results

            case OptimizationMethod.OPT_DIJKSTRA:
                table_widget = self.tableWidget_opt_dijkstra_results

            case OptimizationMethod.OPT_DFS:
                table_widget = self.tableWidget_opt_dfs_results

            case OptimizationMethod.OPT_IDDFS:
                table_widget = self.tableWidget_opt_iddfs_results

            case OptimizationMethod.OPT_NSGA2:
                table_widget = self.tableWidget_opt_nsga2_results

            case OptimizationMethod.OPT_NSGA3:
                table_widget = self.tableWidget_opt_nsga3_results

            case OptimizationMethod.OPT_RL_DQN:
                table_widget = self.tableWidget_opt_rl_dqn_results

            case OptimizationMethod.OPT_LLM_AGENT:
                table_widget = self.tableWidget_opt_llm_agent_results

            case OptimizationMethod.OPT_HUMAN:
                table_widget = self.tableWidget_opt_human_results

            case _:
                raise NotImplementedError(f"Optimization method {opt_method} not implemented")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_res = active_scenario.opt_result_dict[opt_method]
        opt_res_df = opt_res.to_dataframe()

        table_widget.clear()
        table_widget.setRowCount(len(opt_res_df))
        table_widget.setColumnCount(len(opt_res_df.columns))
        table_widget.setHorizontalHeaderLabels(opt_res_df.columns.tolist())

        for row in range(len(opt_res_df)):

            for col in range(len(opt_res_df.columns)):

                value = str(opt_res_df.iat[row, col])
                item = QTableWidgetItem(value)

                # Center text
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if value in COLOR_BY_SKILL_DICT:
                    item.setForeground(QBrush(QColor(COLOR_BY_SKILL_DICT[value])))

                table_widget.setItem(row, col, item)

        header = table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Make text in header bold
        font = header.font()
        font.setBold(True)
        header.setFont(font)

    def _update_labels_opt_results(self, opt_method: OptimizationMethod, reset: bool = False) -> None:

        match opt_method:

            case OptimizationMethod.OPT_A_STAR:
                label_total_steps = self.label_opt_a_star_total_steps
                label_total_time = self.label_opt_a_star_total_time
                label_total_energy = self.label_opt_a_star_total_energy
                label_total_reliability = self.label_opt_a_star_sequence_reliability
                label_total_cost = self.label_opt_a_star_total_cost
                label_total_duration = self.label_opt_a_star_total_duration
                label_total_timestamp = self.label_opt_a_star_timestamp

            case OptimizationMethod.OPT_DIJKSTRA:
                label_total_steps = self.label_opt_dijkstra_total_steps
                label_total_time = self.label_opt_dijkstra_total_time
                label_total_energy = self.label_opt_dijkstra_total_energy
                label_total_reliability = self.label_opt_dijkstra_sequence_reliability
                label_total_cost = self.label_opt_dijkstra_total_cost
                label_total_duration = self.label_opt_dijkstra_total_duration
                label_total_timestamp = self.label_opt_dijkstra_timestamp

            case OptimizationMethod.OPT_DFS:
                label_total_steps = self.label_opt_dfs_total_steps
                label_total_time = self.label_opt_dfs_total_time
                label_total_energy = self.label_opt_dfs_total_energy
                label_total_reliability = self.label_opt_dfs_sequence_reliability
                label_total_cost = self.label_opt_dfs_total_cost
                label_total_duration = self.label_opt_dfs_total_duration
                label_total_timestamp = self.label_opt_dfs_timestamp

            case OptimizationMethod.OPT_IDDFS:
                label_total_steps = self.label_opt_iddfs_total_steps
                label_total_time = self.label_opt_iddfs_total_time
                label_total_energy = self.label_opt_iddfs_total_energy
                label_total_reliability = self.label_opt_iddfs_sequence_reliability
                label_total_cost = self.label_opt_iddfs_total_cost
                label_total_duration = self.label_opt_iddfs_total_duration
                label_total_timestamp = self.label_opt_iddfs_timestamp

            case OptimizationMethod.OPT_NSGA2:
                label_total_steps = self.label_opt_nsga2_total_steps
                label_total_time = self.label_opt_nsga2_total_time
                label_total_energy = self.label_opt_nsga2_total_energy
                label_total_reliability = self.label_opt_nsga2_sequence_reliability
                label_total_cost = self.label_opt_nsga2_total_cost
                label_total_duration = self.label_opt_nsga2_total_duration
                label_total_timestamp = self.label_opt_nsga2_timestamp

            case OptimizationMethod.OPT_NSGA3:
                label_total_steps = self.label_opt_nsga3_total_steps
                label_total_time = self.label_opt_nsga3_total_time
                label_total_energy = self.label_opt_nsga3_total_energy
                label_total_reliability = self.label_opt_nsga3_sequence_reliability
                label_total_cost = self.label_opt_nsga3_total_cost
                label_total_duration = self.label_opt_nsga3_total_duration
                label_total_timestamp = self.label_opt_nsga3_timestamp

            case OptimizationMethod.OPT_RL_DQN:
                label_total_steps = self.label_opt_rl_dqn_total_steps
                label_total_time = self.label_opt_rl_dqn_total_time
                label_total_energy = self.label_opt_rl_dqn_total_energy
                label_total_reliability = self.label_opt_rl_dqn_sequence_reliability
                label_total_cost = self.label_opt_rl_dqn_total_cost
                label_total_duration = self.label_opt_rl_dqn_total_duration
                label_total_timestamp = self.label_opt_rl_dqn_timestamp

            case OptimizationMethod.OPT_LLM_AGENT:
                label_total_steps = self.label_opt_llm_agent_total_steps
                label_total_time = self.label_opt_llm_agent_total_time
                label_total_energy = self.label_opt_llm_agent_total_energy
                label_total_reliability = self.label_opt_llm_agent_sequence_reliability
                label_total_cost = self.label_opt_llm_agent_total_cost
                label_total_duration = self.label_opt_llm_agent_total_duration
                label_total_timestamp = self.label_opt_llm_agent_timestamp

            case OptimizationMethod.OPT_HUMAN:
                label_total_steps = self.label_opt_human_total_steps
                label_total_time = self.label_opt_human_total_time
                label_total_energy = self.label_opt_human_total_energy
                label_total_reliability = self.label_opt_human_sequence_reliability
                label_total_cost = self.label_opt_human_total_cost
                label_total_duration = self.label_opt_human_total_duration
                label_total_timestamp = self.label_opt_human_timestamp

            case _:
                raise NotImplementedError(f"Optimization method {opt_method} not implemented")

        if reset:

            label_total_steps.setText("0")
            label_total_time.setText("0.00 s")
            label_total_energy.setText("0.0000 kWh")
            label_total_reliability.setText("1.000")
            label_total_cost.setText("0.00")
            label_total_duration.setText(f"0min 0s")
            label_total_timestamp.setText("")

            match opt_method:

                case OptimizationMethod.OPT_A_STAR:
                    self.label_opt_a_star_expansions.setText("-")

                case OptimizationMethod.OPT_DIJKSTRA:
                    self.label_opt_dijkstra_expansions.setText("-")

                case OptimizationMethod.OPT_DFS:
                    self.label_opt_dfs_min_solution_depth.setText("-")

                case OptimizationMethod.OPT_IDDFS:
                    self.label_opt_iddfs_min_solution_depth.setText("-")

                case OptimizationMethod.OPT_NSGA2:
                    self.label_opt_nsga2_hypervolume.setText("-")

                case OptimizationMethod.OPT_NSGA3:
                    self.label_opt_nsga3_hypervolume.setText("-")

                case OptimizationMethod.OPT_RL_DQN:
                    self.label_opt_rl_dqn_reward.setText("-")

                case OptimizationMethod.OPT_LLM_AGENT:
                    self.label_opt_llm_agent_tokens.setText("-")

        else:

            active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

            label_total_steps.setText(f"{active_scenario.opt_result_dict[opt_method].steps}")
            label_total_time.setText(f"{active_scenario.opt_result_dict[opt_method].total_time:.2f} s")
            label_total_energy.setText(f"{joules_to_kwh(active_scenario.opt_result_dict[opt_method].total_energy):.4f} kWh")
            label_total_reliability.setText(f"{active_scenario.opt_result_dict[opt_method].sequence_reliability:.3f}")
            label_total_cost.setText(f"{active_scenario.opt_result_dict[opt_method].total_cost:.2f}")
            label_total_duration.setText(f"{duration_formatting(active_scenario.opt_result_dict[opt_method].total_duration_seconds)}")
            label_total_timestamp.setText(active_scenario.opt_result_dict[opt_method].get_timestamp_str())

            match opt_method:

                case OptimizationMethod.OPT_A_STAR:
                    expansions_count = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_EXPANSIONS]
                    self.label_opt_a_star_expansions.setText(f"{expansions_count}")

                case OptimizationMethod.OPT_DIJKSTRA:
                    expansions_count = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_EXPANSIONS]
                    self.label_opt_dijkstra_expansions.setText(f"{expansions_count}")

                case OptimizationMethod.OPT_DFS:
                    min_solution_depth = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_MIN_SOLUTION_DEPTH]
                    self.label_opt_dfs_min_solution_depth.setText(f"{min_solution_depth}")

                case OptimizationMethod.OPT_IDDFS:
                    min_solution_depth = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_MIN_SOLUTION_DEPTH]
                    self.label_opt_iddfs_min_solution_depth.setText(f"{min_solution_depth}")

                case OptimizationMethod.OPT_NSGA2:
                    hypervolume = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_HYPERVOLUME]
                    self.label_opt_nsga2_hypervolume.setText(f"{hypervolume:.3f}")

                case OptimizationMethod.OPT_NSGA3:
                    hypervolume = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_HYPERVOLUME]
                    self.label_opt_nsga3_hypervolume.setText(f"{hypervolume:.3f}")

                case OptimizationMethod.OPT_RL_DQN:
                    reward = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_REWARD]
                    self.label_opt_rl_dqn_reward.setText(f"{reward:.3f}")

                case OptimizationMethod.OPT_LLM_AGENT:
                    tokens = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_TOTAL_TOKEN_COUNT]
                    self.label_opt_llm_agent_tokens.setText(f"{tokens}")

                    avg_response_seconds = active_scenario.opt_result_dict[opt_method].other_params_dict[OPT_RES_PARAM_AVG_RESPONSE_TIME]
                    self.label_opt_llm_agent_avg_response.setText(f"{avg_response_seconds:.2f} s")


    def _update_table_widget_opt_comparison(self) -> None:

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        if len(active_scenario.opt_result_dict) == 0:
            return

        # Generate comparison dataframe
        opt_result_comp_data_dict_list = []

        for opt_result in active_scenario.opt_result_dict.values():

            if not isinstance(opt_result, OptimizationResult):
                continue

            opt_result_comp_data_dict = {COL_OPT_RES_COMP_METHOD: opt_result.opt_method.get_short_name(),
                                         COL_OPT_RES_COMP_STEPS: opt_result.steps,
                                         COL_OPT_RES_COMP_TOTAL_TIME: opt_result.total_time,
                                         COL_OPT_RES_COMP_TOTAL_ENERGY: joules_to_kwh(opt_result.total_energy),
                                         COL_OPT_RES_COMP_SEQUENCE_RELIABILITY: opt_result.sequence_reliability,
                                         COL_OPT_RES_COMP_TOTAL_COST: opt_result.total_cost,
                                         COL_OPT_RES_COMP_DURATION_SECONDS: opt_result.total_duration_seconds}

            opt_result_comp_data_dict_list.append(opt_result_comp_data_dict)

        if len(opt_result_comp_data_dict_list) == 0:
            return

        opt_comp_df = pd.DataFrame(data=opt_result_comp_data_dict_list)

        # Find minimum values per column
        min_value_indices_per_col_dict = {}

        for col in opt_comp_df.columns:

            if (col == COL_OPT_RES_COMP_METHOD) or (col == COL_OPT_RES_COMP_SEQUENCE_RELIABILITY):
                continue

            min_value_indices_list = opt_comp_df.index[opt_comp_df[col] == opt_comp_df[col].min()].tolist()
            min_value_indices_per_col_dict[col] = min_value_indices_list

        max_value_seq_rel_indices = opt_comp_df.index[opt_comp_df[COL_OPT_RES_COMP_SEQUENCE_RELIABILITY] ==
                                                      opt_comp_df[COL_OPT_RES_COMP_SEQUENCE_RELIABILITY].max()].tolist()

        most_recent_result_opt_method = list(active_scenario.opt_result_dict.keys())[0]
        most_recent_opt_result = active_scenario.opt_result_dict[most_recent_result_opt_method]

        if most_recent_opt_result is None:
            return

        most_recent_result_datetime = most_recent_opt_result.timestamp_dt

        for opt_method in active_scenario.opt_result_dict.keys():

            opt_result = active_scenario.opt_result_dict[opt_method]

            if opt_result is None:
                continue

            if opt_result.timestamp_dt > most_recent_result_datetime:
                most_recent_result_datetime = opt_result.timestamp_dt
                most_recent_result_opt_method = opt_method

        self.label_opt_comparison_timestamp.setText(active_scenario.opt_result_dict[most_recent_result_opt_method].get_timestamp_str())

        # Fill table with content
        self.tableWidget_opt_comparison.clear()

        # Reset sorting after update
        self.tableWidget_opt_comparison.setSortingEnabled(False)
        self.tableWidget_opt_comparison.setSortingEnabled(True)
        self.tableWidget_opt_comparison.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

        self.tableWidget_opt_comparison.setRowCount(len(opt_comp_df))
        self.tableWidget_opt_comparison.setColumnCount(len(opt_comp_df.columns))
        self.tableWidget_opt_comparison.setHorizontalHeaderLabels(opt_comp_df.columns.tolist())
        self.tableWidget_opt_comparison.setVerticalHeaderLabels([str(i) for i in opt_comp_df.index])

        for row in range(len(opt_comp_df)):

            for col in range(len(opt_comp_df.columns)):

                col_name = opt_comp_df.columns[col]

                item = QTableWidgetItem()

                # Set data for sorting
                item.setData(Qt.ItemDataRole.DisplayRole, opt_comp_df.iat[row, col])

                if col_name == COL_OPT_RES_COMP_TOTAL_ENERGY:

                    item.setText(f"{opt_comp_df.iat[row, col]:.4f}")

                elif (col_name == COL_OPT_RES_COMP_TOTAL_TIME) or (col_name == COL_OPT_RES_COMP_TOTAL_COST):

                    item.setText(f"{opt_comp_df.iat[row, col]:.2f}")

                elif col_name == COL_OPT_RES_COMP_SEQUENCE_RELIABILITY:

                    item.setText(f"{opt_comp_df.iat[row, col]:.3f}")

                elif col_name == COL_OPT_RES_COMP_DURATION_SECONDS:

                    item.setText(duration_formatting(opt_comp_df.iat[row, col]))

                # Center text
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Mark minimum / maximum values
                if col_name in min_value_indices_per_col_dict.keys():

                    if row in min_value_indices_per_col_dict[col_name]:
                        item.setForeground(QBrush(QColor("forestgreen")))

                elif col_name == COL_OPT_RES_COMP_SEQUENCE_RELIABILITY:

                    if row in max_value_seq_rel_indices:
                        item.setForeground(QBrush(QColor("forestgreen")))

                self.tableWidget_opt_comparison.setItem(row, col, item)

        header = self.tableWidget_opt_comparison.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Make text in header bold
        font = header.font()
        font.setBold(True)
        header.setFont(font)

    def _update_table_widget_scenario_overview(self) -> None:

        scenario_overview_dict_list = []

        for scenario in self._loaded_scenario_list:

            product_count = len(scenario.product_by_id_dict)
            task_count = len(scenario.get_sorted_product_list()[0].target_state.processing_tasks)
            skill_count = scenario.factory.get_total_processing_skill_count()
            processing_machine_count = len(scenario.factory.processing_machine_by_id_dict)
            transport_machine_count = len(scenario.factory.transport_machine_by_id_dict)
            storage_machine_count = len(scenario.factory.storage_machine_by_id_dict)
            machine_count = len(scenario.factory.machine_by_id_dict)
            connections_count = len(scenario.factory.transport_connections)
            action_catalog_count = len(scenario.sorted_action_catalog)

            scenario_overview_dict = {COL_SCENARIO_OVERVIEW_NAME: scenario.name,
                                      COL_SCENARIO_OVERVIEW_PRODUCTS: product_count,
                                      COL_SCENARIO_OVERVIEW_TASKS: task_count,
                                      COL_SCENARIO_OVERVIEW_PROC_SKILLS: skill_count,
                                      COL_SCENARIO_OVERVIEW_PROCESSING_MACHINES: processing_machine_count,
                                      COL_SCENARIO_OVERVIEW_TRANSPORT_MACHINES: transport_machine_count,
                                      COL_SCENARIO_OVERVIEW_STORAGE_MACHINES: storage_machine_count,
                                      COL_SCENARIO_OVERVIEW_MACHINES: machine_count,
                                      COL_SCENARIO_OVERVIEW_CONNECTIONS: connections_count,
                                      COL_SCENARIO_OVERVIEW_ACTION_CATALOG_COUNT: action_catalog_count,
                                      }

            # Only show this data if specifically requested as it may take some time to calculate
            if self.checkBox_scenario_overview_show_states_transitions.isChecked():

                state_count = scenario.calculate_total_state_count()
                transition_count = scenario.calculate_total_transition_count()

                scenario_overview_dict[COL_SCENARIO_OVERVIEW_STATE_COUNT] = state_count
                scenario_overview_dict[COL_SCENARIO_OVERVIEW_TRANSITION_COUNT] = transition_count

            scenario_overview_dict_list.append(scenario_overview_dict)

        if len(scenario_overview_dict_list) == 0:
            return

        scenario_overview_df = pd.DataFrame(data=scenario_overview_dict_list)

        # Fill table with content
        self.tableWidget_scenario_overview.clear()

        # Reset sorting after update
        self.tableWidget_scenario_overview.setSortingEnabled(False)
        self.tableWidget_scenario_overview.setSortingEnabled(True)
        self.tableWidget_scenario_overview.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

        self.tableWidget_scenario_overview.setRowCount(len(scenario_overview_df))
        self.tableWidget_scenario_overview.setColumnCount(len(scenario_overview_df.columns))
        self.tableWidget_scenario_overview.setHorizontalHeaderLabels(scenario_overview_df.columns.tolist())
        self.tableWidget_scenario_overview.setVerticalHeaderLabels([str(i) for i in scenario_overview_df.index])

        for row in range(len(scenario_overview_df)):

            for col in range(len(scenario_overview_df.columns)):

                item = QTableWidgetItem()

                # Set data for sorting
                item.setData(Qt.ItemDataRole.DisplayRole, scenario_overview_df.iat[row, col])

                # Center text
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                self.tableWidget_scenario_overview.setItem(row, col, item)

        header = self.tableWidget_scenario_overview.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Make text in header bold
        font = header.font()
        font.setBold(True)
        header.setFont(font)

        self._update_labels_scenario_overview_details()

    def _update_labels_scenario_overview_details(self) -> None:

        selected_scenario = self._loaded_scenario_list[self._selected_scenario_idx]

        self.label_scenario_details_name.setText(selected_scenario.name)
        self.label_scenario_details_note.setText(selected_scenario.note)
        self.label_scenario_details_file_name.setText(selected_scenario.file_path.name)
        self.label_scenario_details_modified.setText(selected_scenario.last_modified_dt.strftime('%d.%m.%Y %H:%M:%S'))

        milling_machine_count = str(len(selected_scenario.factory.get_machines_by_machine_type(MillingMachine)))
        cutting_machine_count = str(len(selected_scenario.factory.get_machines_by_machine_type(CuttingMachine)))
        drilling_machine_count = str(len(selected_scenario.factory.get_machines_by_machine_type(DrillingMachine)))
        agv_machine_count = str(len(selected_scenario.factory.get_machines_by_machine_type(AutomatedGuidedVehicle)))
        cb_machine_count = str(len(selected_scenario.factory.get_machines_by_machine_type(ConveyorBelt)))
        tar_machine_count = str(len(selected_scenario.factory.get_machines_by_machine_type(ThreeAxisRobot)))

        self.label_scenario_details_milling_machines.setText(milling_machine_count)
        self.label_scenario_details_cutting_machines.setText(cutting_machine_count)
        self.label_scenario_details_drilling_machines.setText(drilling_machine_count)
        self.label_scenario_details_agvs.setText(agv_machine_count)
        self.label_scenario_details_conveyor_belts.setText(cb_machine_count)
        self.label_scenario_details_three_axis_robots.setText(tar_machine_count)

    def _update_opt_result(self, opt_method: OptimizationMethod):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        if active_scenario.opt_result_dict[opt_method] is None:
            return

        # Save results as pickle file
        active_scenario.opt_result_dict[opt_method].pickle_dump(output_directory=active_scenario.opt_result_dir_path)

        # Save exemplary task results as CSV
        active_scenario.opt_result_dict[opt_method].to_csv(output_directory=active_scenario.opt_result_dir_path)

        # Update table with exemplary task results
        self._update_table_widget_opt_results(opt_method)

        # Update result labels
        self._update_labels_opt_results(opt_method)

        # Update comparison table
        self._update_table_widget_opt_comparison()

    def _update_sim_all(self):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        self.label_sim_total_steps.setText(f"{active_scenario.step_count}")
        self.label_sim_total_time.setText(f"{active_scenario.time_sum:.2f} s")
        self.label_sim_total_energy.setText(f"{joules_to_kwh(active_scenario.energy_sum):.4f} kWh")
        self.label_sim_sequence_reliability.setText(f"{active_scenario.sequence_reliability:.3f}")
        self.label_sim_total_cost.setText(f"{active_scenario.calculate_total_cost(self._objective_function):.2f}")

        if active_scenario.is_done():
            self.label_sim_is_done.setText("True")
            self.label_sim_is_done.setStyleSheet("color: forestgreen;")
            self.pushButton_sim_save_results.setEnabled(True)
        else:
            self.label_sim_is_done.setText("False")
            self.label_sim_is_done.setStyleSheet("color: crimson;")
            self.pushButton_sim_save_results.setEnabled(False)

        self._update_table_widget_sim_actions()
        self._update_tree_widget_state()

    def _update_analysis_plots(self) -> None:

        analysis_dict_list = []

        for loaded_scenario in self._loaded_scenario_list:

            task_count = len(loaded_scenario.get_sorted_product_list()[0].target_state.processing_tasks)

            a_star_opt_res = loaded_scenario.opt_result_dict.get(OptimizationMethod.OPT_A_STAR)
            nsga2_opt_res = loaded_scenario.opt_result_dict.get(OptimizationMethod.OPT_NSGA2)
            rl_dqn_opt_res = loaded_scenario.opt_result_dict.get(OptimizationMethod.OPT_RL_DQN)

            analysis_dict = {
                COL_TASKS: task_count,
                COL_A_STAR_TIME: a_star_opt_res.total_duration_seconds if isinstance(a_star_opt_res, OptimizationResult) else np.nan,
                COL_NSGA2_TIME: nsga2_opt_res.total_duration_seconds if isinstance(nsga2_opt_res, OptimizationResult) else np.nan,
                COL_RL_DQN_TIME: rl_dqn_opt_res.total_duration_seconds if isinstance(rl_dqn_opt_res, OptimizationResult) else np.nan,
                COL_NSGA2_RATIO: np.nan,
                COL_RL_DQN_RATIO: np.nan,
            }

            # Solution quality is expressed relative to the A* optimum, so it needs an A* result
            if isinstance(a_star_opt_res, OptimizationResult) and a_star_opt_res.total_cost > 0:

                if isinstance(nsga2_opt_res, OptimizationResult):
                    analysis_dict[COL_NSGA2_RATIO] = nsga2_opt_res.total_cost / a_star_opt_res.total_cost

                if isinstance(rl_dqn_opt_res, OptimizationResult):
                    analysis_dict[COL_RL_DQN_RATIO] = rl_dqn_opt_res.total_cost / a_star_opt_res.total_cost

            analysis_dict_list.append(analysis_dict)

        # Always hand the (possibly empty) frame to the plot so stale content is cleared, e.g. when
        # switching to a scenario directory without any optimization results.
        analysis_df = pd.DataFrame(data=analysis_dict_list)
        self._analysis_plot.update_plots(analysis_df)

    def on_change_checkbox_factory_distances(self):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        self._draw_graph_in_groupbox(active_scenario.factory.create_digraph_stationary_machines(),
                                     labels=True,
                                     edge_labels=self.checkBox_factory_show_distances.isChecked(),
                                     node_size=800)

    def on_cell_selected_sim_actions(self, row_selected, column_selected):
        self._selected_task_result = self._current_task_result_list[row_selected]
        self._selected_action_idx = self._current_action_idx_list[row_selected]
        self.label_sim_selected_action.setText(str(self._selected_task_result.get_short_name(with_product=True, with_machine=True)))
        self.pushButton_sim_execute_action.setEnabled(True)

    def on_cell_selected_scenario_overview(self, row_selected, column_selected):
        self._selected_scenario_idx = row_selected
        self._update_labels_scenario_overview_details()

    def on_click_scenario_overview_reload(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_reload_all_scenarios_complete)

        self._thread_manager.run_thread(self._load_all_scenarios, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_scenario_overview_reload.setEnabled(False)

    def on_click_scenario_load(self):
        self._active_scenario_idx = self._selected_scenario_idx
        self._initialize_scenario()

    def on_click_sim_execute_action(self):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        if not self._sim_started:

            self._sim_started = True
            self._sim_start_time = time.time()
            self.pushButton_sim_reset_scenario.setEnabled(True)

        self._sim_action_idx_sequence.append(self._selected_action_idx)
        active_scenario.execute_action(self._selected_task_result)

        # A freshly executed action invalidates any previously undone actions
        self._sim_redo_stack = []

        self.pushButton_sim_execute_action.setEnabled(False)

        self._update_sim_all()
        self._update_undo_redo_enabled()

    def on_click_sim_undo_last_action(self):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        if len(active_scenario.task_result_history) == 0:
            return

        # Remember the undone action so Redo can re-execute it
        undone_task_result = active_scenario.task_result_history[-1]

        active_scenario.undo_last_action()

        # TODO move action idx sequence to Scenario
        undone_action_idx = None
        if len(self._sim_action_idx_sequence) > 0:
            undone_action_idx = self._sim_action_idx_sequence.pop(-1)

        self._sim_redo_stack.append((undone_task_result, undone_action_idx))

        self._update_sim_all()
        self._update_undo_redo_enabled()

    def on_click_sim_redo_last_action(self):

        if len(self._sim_redo_stack) == 0:
            return

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        redo_task_result, redo_action_idx = self._sim_redo_stack.pop()

        # Redo is only reachable after an undo, so the simulation has already been started
        active_scenario.execute_action(redo_task_result)

        if redo_action_idx is not None:
            self._sim_action_idx_sequence.append(redo_action_idx)

        self._update_sim_all()
        self._update_undo_redo_enabled()

    def _update_undo_redo_enabled(self) -> None:
        """Sync the undo button and the Edit menu undo/redo actions with the current history."""

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        can_undo = len(active_scenario.task_result_history) > 0
        can_redo = len(self._sim_redo_stack) > 0

        self.pushButton_sim_undo_last_action.setEnabled(can_undo)
        self.action_undo.setEnabled(can_undo)
        self.action_redo.setEnabled(can_redo)

    def on_click_sim_reset_scenario(self):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        active_scenario.reset()

        self._sim_started = False
        self._sim_start_time = time.time()
        self._sim_action_idx_sequence = []
        self._sim_redo_stack = []

        self.pushButton_sim_execute_action.setEnabled(False)

        self._update_sim_all()
        self._update_undo_redo_enabled()

    def on_click_sim_save_results(self):

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        if active_scenario.is_done():


            opt_result = OptimizationResult(action_idx_sequence=list(self._sim_action_idx_sequence),
                                            task_result_list=active_scenario.task_result_history,
                                            total_time=active_scenario.time_sum,
                                            total_energy=active_scenario.energy_sum,
                                            sequence_reliability=active_scenario.sequence_reliability,
                                            objective_function=self._objective_function,
                                            other_params_dict={},
                                            total_duration_seconds=(time.time() - self._sim_start_time),
                                            opt_method=OptimizationMethod.OPT_HUMAN)

            active_scenario.opt_result_dict[OptimizationMethod.OPT_HUMAN] = opt_result

            self._update_opt_result(OptimizationMethod.OPT_HUMAN)

            self._update_table_widget_opt_comparison()
            self.tabWidget_opt.setCurrentIndex(6)
            self.tabWidget_main.setCurrentIndex(3)

            print_with_timestamp("Simulation results saved... ")

        else:

            print_with_timestamp("Simulation results not saved, all products need to be completed... ")


    def on_click_opt_a_star_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_a_star_complete)

        self._thread_manager.run_thread(self.run_a_star_search, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_a_star_run.setEnabled(False)


    def on_click_opt_dijkstra_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_dijkstra_complete)

        self._thread_manager.run_thread(self.run_dijkstra_search, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_dijkstra_run.setEnabled(False)

    def on_click_opt_dfs_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_dfs_complete)

        self._thread_manager.run_thread(self.run_dfs, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_dfs_run.setEnabled(False)

    def on_click_opt_iddfs_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_iddfs_complete)

        self._thread_manager.run_thread(self.run_iddfs, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_iddfs_run.setEnabled(False)

    def on_click_opt_nsga2_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_nsga2_complete)

        self._thread_manager.run_thread(self.run_nsga2, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_nsga2_run.setEnabled(False)

    def on_click_opt_nsga3_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_nsga3_complete)

        self._thread_manager.run_thread(self.run_nsga3, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_nsga3_run.setEnabled(False)

    def on_click_opt_rl_dqn_training_and_inference_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_rl_dqn_training_and_inference_complete)

        self._thread_manager.run_thread(self.run_rl_dqn_run_training_and_inference, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_rl_dqn_run_training_and_inference.setEnabled(False)

    def on_click_opt_llm_agent_prompt_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_llm_agent_prompt_complete)

        self._thread_manager.run_thread(self.run_llm_agent_prompt, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_llm_agent_run_prompt.setEnabled(False)

    def on_click_opt_human_sim(self):

        self.tabWidget_main.setCurrentIndex(2)

    def on_click_opt_auto_run(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self.on_opt_auto_run_complete)

        # Separate updater without on_finished for the individual method runs: each run_* method
        # calls finish() when it is done, which must not trigger the auto-run completion callback
        method_progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                                      progress_bar_label=self.label_opt_progress)

        # Snapshot checkbox states on the GUI thread; the worker thread must not access widgets
        methods_to_run = {
            OptimizationMethod.OPT_A_STAR: self.checkBox_auto_run_a_star.isChecked(),
            OptimizationMethod.OPT_DIJKSTRA: self.checkBox_auto_run_dijkstra.isChecked(),
            OptimizationMethod.OPT_DFS: self.checkBox_auto_run_dfs.isChecked(),
            OptimizationMethod.OPT_IDDFS: self.checkBox_auto_run_iddfs.isChecked(),
            OptimizationMethod.OPT_NSGA2: self.checkBox_auto_run_nsga2.isChecked(),
            OptimizationMethod.OPT_NSGA3: self.checkBox_auto_run_nsga3.isChecked(),
            OptimizationMethod.OPT_LLM_AGENT: self.checkBox_auto_run_llm_agent.isChecked(),
            OptimizationMethod.OPT_RL_DQN: self.checkBox_auto_run_rl_dqn.isChecked(),
        }

        self.pushButton_opt_auto_run.setEnabled(False)

        self._thread_manager.run_thread(self.run_opt_auto_run, progress_updater=progress_updater,
                                        additional_kwargs=dict(method_progress_updater=method_progress_updater,
                                                               methods_to_run=methods_to_run))

    def run_opt_auto_run(self, progress_updater: PyQtProgressUpdater,
                         method_progress_updater: PyQtProgressUpdater,
                         methods_to_run: dict[OptimizationMethod, bool]):

        print_with_timestamp("Auto Run Optimization...")

        method_runners = {
            OptimizationMethod.OPT_A_STAR: self.run_a_star_search,
            OptimizationMethod.OPT_DIJKSTRA: self.run_dijkstra_search,
            OptimizationMethod.OPT_DFS: self.run_dfs,
            OptimizationMethod.OPT_IDDFS: self.run_iddfs,
            OptimizationMethod.OPT_NSGA2: self.run_nsga2,
            OptimizationMethod.OPT_NSGA3: self.run_nsga3,
            OptimizationMethod.OPT_LLM_AGENT: self.run_llm_agent_prompt,
            OptimizationMethod.OPT_RL_DQN: self.run_rl_dqn_run_training_and_inference,
        }

        for idx, scenario in enumerate(self._loaded_scenario_list):

            print_with_timestamp(f"-------------- SCENARIO {idx} - '{scenario.name}' --------------")

            self._active_scenario_idx = idx

            # Rebuild the objective function with this scenario's normalization scales.
            # Deliberately NOT calling _initialize_scenario() here: it mutates GUI widgets and
            # replaces the thread manager, neither of which is safe from this worker thread
            time_scale, energy_scale, reliability_scale = compute_cost_scales(scenario)
            self._objective_function = ObjectiveFunction(time_weight=self._time_weight,
                                                         energy_weight=self._energy_weight,
                                                         reliability_weight=self._reliability_weight,
                                                         time_scale=time_scale,
                                                         energy_scale=energy_scale,
                                                         reliability_scale=reliability_scale)

            for opt_method, run_method in method_runners.items():

                if not methods_to_run.get(opt_method, False):
                    continue

                try:
                    run_method(progress_updater=method_progress_updater)

                except Exception as e:
                    print_with_timestamp(f"Auto run: '{opt_method.name}' failed for scenario "
                                         f"'{scenario.name}': {e}")
                    continue

                # Persist immediately so an interrupted batch keeps all completed results
                opt_result = scenario.opt_result_dict[opt_method]

                if opt_result is not None:
                    opt_result.pickle_dump(output_directory=scenario.opt_result_dir_path)
                    opt_result.to_csv(output_directory=scenario.opt_result_dir_path)

        progress_updater.finish()

    def on_opt_auto_run_complete(self):

        print_with_timestamp("Auto run completed... ")

        self.pushButton_opt_auto_run.setEnabled(True)

    def on_action_show_about_dialog(self):

        dialog = QDialog(self._window)
        about_ui = Ui_Dialog()
        about_ui.setupUi(dialog)
        dialog.exec()

    def on_action_switch_scenario_directory(self):

        # Open a file dialog to select a folder, starting at the specified directory
        scenario_dir_path_str = QFileDialog.getExistingDirectory(caption="Select Scenario Directory",
                                                                 directory=str(DIR_DATA_INPUT_ALL_SCENARIO_DIRS_PATH))

        # An empty string means the dialog was cancelled/closed - keep the current directory
        # instead of switching to Path("") (the cwd), which previously froze the program.
        if not scenario_dir_path_str:
            return

        chosen_directory = Path(scenario_dir_path_str)

        # Only switch if the directory actually contains at least one scenario, otherwise the
        # scenario list would end up empty and later indexing into it would crash.
        if not self._directory_contains_scenarios(chosen_directory):
            print_with_timestamp(f"No scenarios found in '{chosen_directory}' - keeping current directory.")
            return

        self._active_sc_directory = chosen_directory

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_reload_all_scenarios_complete)

        self._thread_manager.run_thread(self._load_all_scenarios, progress_updater=progress_updater,
                                        additional_kwargs=dict())

    def on_action_export_analysis_data(self):

        # Show progress in the log tab and run the (potentially slow) export off the GUI thread
        self.tabWidget_main.setCurrentIndex(self.tabWidget_main.indexOf(self.tab_log))

        print_with_timestamp("Exporting analysis data (this may take a while)...")

        self.action_export_analysis.setEnabled(False)

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                               progress_bar_label=self.label_opt_progress,
                                               on_finished=self._on_export_analysis_data_complete)

        self._thread_manager.run_thread(self._export_analysis_data, progress_updater=progress_updater,
                                        additional_kwargs=dict())

    def _on_export_analysis_data_complete(self) -> None:

        self.action_export_analysis.setEnabled(True)

    def _export_analysis_data(self, progress_updater: PyQtProgressUpdater):

        analysis_data_dict_list = []

        for scenario in self._loaded_scenario_list:

            product_count = len(scenario.product_by_id_dict)
            task_count = len(scenario.get_sorted_product_list()[0].target_state.processing_tasks)
            skill_count = scenario.factory.get_total_processing_skill_count()
            processing_machine_count = len(scenario.factory.processing_machine_by_id_dict)
            transport_machine_count = len(scenario.factory.transport_machine_by_id_dict)
            storage_machine_count = len(scenario.factory.storage_machine_by_id_dict)
            machine_count = len(scenario.factory.machine_by_id_dict)
            connections_count = len(scenario.factory.transport_connections)
            action_catalog_count = len(scenario.sorted_action_catalog)

            state_count = scenario.calculate_total_state_count()
            transition_count = scenario.calculate_total_transition_count()

            scenario_overview_dict = {COL_SCENARIO_OVERVIEW_NAME: scenario.name,
                                      COL_SCENARIO_OVERVIEW_PRODUCTS: product_count,
                                      COL_SCENARIO_OVERVIEW_TASKS: task_count,
                                      COL_SCENARIO_OVERVIEW_PROC_SKILLS: skill_count,
                                      COL_SCENARIO_OVERVIEW_PROCESSING_MACHINES: processing_machine_count,
                                      COL_SCENARIO_OVERVIEW_TRANSPORT_MACHINES: transport_machine_count,
                                      COL_SCENARIO_OVERVIEW_STORAGE_MACHINES: storage_machine_count,
                                      COL_SCENARIO_OVERVIEW_MACHINES: machine_count,
                                      COL_SCENARIO_OVERVIEW_CONNECTIONS: connections_count,
                                      COL_SCENARIO_OVERVIEW_ACTION_CATALOG_COUNT: action_catalog_count,
                                      COL_SCENARIO_OVERVIEW_STATE_COUNT: state_count,
                                      COL_SCENARIO_OVERVIEW_TRANSITION_COUNT: transition_count,
                                      }

            a_star_expansions = 0
            a_star_steps = 0
            a_star_time_s = 0
            a_star_cost = 0

            dijkstra_expansions = 0
            dijkstra_steps = 0
            dijkstra_time_s = 0
            dijkstra_cost = 0

            nsga2_steps = 0
            nsga2_time_s = 0
            nsga2_trials = 0
            nsga2_time_per_trial_s = 0
            nsga2_cost = 0
            nsga2_cost_delta = 0
            nsga2_cost_ratio = None

            rl_dqn_steps = 0
            rl_dqn_time_s = 0
            rl_dqn_cost = 0
            rl_dqn_cost_delta = 0
            rl_dqn_cost_ratio = None

            a_star_opt_res = scenario.opt_result_dict[OptimizationMethod.OPT_A_STAR]

            if a_star_opt_res is not None:
                a_star_expansions = a_star_opt_res.other_params_dict[OPT_RES_PARAM_EXPANSIONS]
                a_star_steps = len(a_star_opt_res.task_result_list)
                a_star_time_s = a_star_opt_res.total_duration_seconds
                a_star_cost = a_star_opt_res.total_cost

            dijkstra_opt_res = scenario.opt_result_dict[OptimizationMethod.OPT_DIJKSTRA]

            if dijkstra_opt_res is not None:
                dijkstra_expansions = dijkstra_opt_res.other_params_dict[OPT_RES_PARAM_EXPANSIONS]
                dijkstra_steps = len(dijkstra_opt_res.task_result_list)
                dijkstra_time_s = dijkstra_opt_res.total_duration_seconds
                dijkstra_cost = dijkstra_opt_res.total_cost

            nsga2_opt_res = scenario.opt_result_dict[OptimizationMethod.OPT_NSGA2]

            if nsga2_opt_res is not None:

                nsga2_steps = len(nsga2_opt_res.task_result_list)

                # total_duration_seconds covers ALL best-of-N trials; also export the per-trial mean
                nsga2_time_s = nsga2_opt_res.total_duration_seconds
                nsga2_trials = nsga2_opt_res.other_params_dict.get(OPT_RES_PARAM_NUM_TRIALS, 1)
                nsga2_time_per_trial_s = nsga2_time_s / max(nsga2_trials, 1)

                nsga2_cost = nsga2_opt_res.total_cost


            rl_dqn_opt_res = scenario.opt_result_dict[OptimizationMethod.OPT_RL_DQN]

            if rl_dqn_opt_res is not None:

                rl_dqn_steps = len(rl_dqn_opt_res.task_result_list)
                rl_dqn_time_s = rl_dqn_opt_res.total_duration_seconds
                rl_dqn_cost = rl_dqn_opt_res.total_cost

            if a_star_opt_res is not None and nsga2_opt_res is not None:
                nsga2_cost_delta =  nsga2_opt_res.total_cost - a_star_opt_res.total_cost

                # Relative optimality gap (1.0 = optimal), used for the solution quality plot
                if a_star_opt_res.total_cost > 0:
                    nsga2_cost_ratio = nsga2_opt_res.total_cost / a_star_opt_res.total_cost

            if a_star_opt_res is not None and rl_dqn_opt_res is not None:
                rl_dqn_cost_delta = rl_dqn_opt_res.total_cost - a_star_opt_res.total_cost

                if a_star_opt_res.total_cost > 0:
                    rl_dqn_cost_ratio = rl_dqn_opt_res.total_cost / a_star_opt_res.total_cost

            scenario_overview_dict["A* Expansions"] = a_star_expansions
            scenario_overview_dict["A* Steps"] = a_star_steps
            scenario_overview_dict["A* Time (s)"] = a_star_time_s
            scenario_overview_dict["A* Cost"] = a_star_cost

            scenario_overview_dict["Dijkstra Expansions"] = dijkstra_expansions
            scenario_overview_dict["Dijkstra Steps"] = dijkstra_steps
            scenario_overview_dict["Dijkstra Time (s)"] = dijkstra_time_s
            scenario_overview_dict["Dijkstra Cost"] = dijkstra_cost

            scenario_overview_dict["NSGA2 Steps"] = nsga2_steps
            scenario_overview_dict["NSGA2 Trials"] = nsga2_trials
            scenario_overview_dict["NSGA2 Time (s)"] = nsga2_time_s
            scenario_overview_dict["NSGA2 Time per Trial (s)"] = nsga2_time_per_trial_s
            scenario_overview_dict["NSGA2 Cost"] = nsga2_cost
            scenario_overview_dict["NSGA2 - A* Cost"] = nsga2_cost_delta
            scenario_overview_dict["NSGA2 / A* Cost"] = nsga2_cost_ratio

            scenario_overview_dict["RL DQN Steps"] = rl_dqn_steps
            scenario_overview_dict["RL DQN Time (s)"] = rl_dqn_time_s
            scenario_overview_dict["RL DQN Cost"] = rl_dqn_cost
            scenario_overview_dict["RL DQN - A* Cost"] = rl_dqn_cost_delta
            scenario_overview_dict["RL DQN / A* Cost"] = rl_dqn_cost_ratio

            analysis_data_dict_list.append(scenario_overview_dict)

        if len(analysis_data_dict_list) == 0:
            print_with_timestamp("No scenarios to export.")
            progress_updater.finish()
            return

        analysis_data_df = pd.DataFrame(data=analysis_data_dict_list)

        file_output_path = self._active_sc_directory / "analysis_data.csv"
        analysis_data_df.to_csv(file_output_path, sep=";")

        print_with_timestamp(f"Analysis data saved to file: '{file_output_path.name}'")

        progress_updater.finish()

    def run_a_star_search(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting A* ...")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result  = astar_search(
            scenario_file_path=active_scenario.file_path,
            objective_function=self._objective_function,
            use_heuristic=True,
            time_limit_s=None,
            max_expansions=None,
            verbose=True,
            progress_updater=progress_updater,
        )

        if opt_result is not None:

            print_with_timestamp(f"[A*] Optimal path found: {opt_result.action_idx_sequence}")

            print_with_timestamp(f"[A*] cost={opt_result.total_cost:.4f} | "
                  f"time={opt_result.total_time:.4f}s | "
                  f"energy={joules_to_kwh(opt_result.total_energy):.4f}kWh | "
                  f"reliability={opt_result.sequence_reliability:.6f} | "
                  f"expansions={opt_result.other_params_dict['expansions']} | "
                  f"elapsed={opt_result.total_duration_seconds:.2f}s\n")

            active_scenario.opt_result_dict[OptimizationMethod.OPT_A_STAR] = opt_result

        else:
            print_with_timestamp("[A*] No solution within limits.")

        progress_updater.finish()

    def run_dijkstra_search(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting Dijkstra ...")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result = astar_search(
            scenario_file_path=active_scenario.file_path,
            objective_function=self._objective_function,
            use_heuristic=False,
            time_limit_s=None,
            max_expansions=None,
            verbose=True,
            progress_updater=progress_updater,
        )

        if opt_result is not None:

            print_with_timestamp(f"[Dijkstra] Optimal path found: {opt_result.action_idx_sequence}")

            print_with_timestamp(f"[Dijkstra] cost={opt_result.total_cost:.4f} | "
                                 f"time={opt_result.total_time:.4f}s | "
                                 f"energy={joules_to_kwh(opt_result.total_energy):.4f}kWh | "
                                 f"reliability={opt_result.sequence_reliability:.6f} | "
                                 f"expansions={opt_result.other_params_dict['expansions']} | "
                                 f"elapsed={opt_result.total_duration_seconds:.2f}s\n")

            active_scenario.opt_result_dict[OptimizationMethod.OPT_DIJKSTRA] = opt_result

        else:
            print_with_timestamp("[Dijkstra] No solution within limits.")

        progress_updater.finish()

    def run_dfs(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting DFS...")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result = run_iddfs(scenario_file_path=active_scenario.file_path,
                               objective_function=self._objective_function,
                               opt_method=OptimizationMethod.OPT_DFS,
                               max_steps=20,
                               verbose=True,
                               progress_updater=progress_updater)

        active_scenario.opt_result_dict[OptimizationMethod.OPT_DFS] = opt_result

        progress_updater.finish()

    def run_iddfs(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting IDDFS...")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result = run_iddfs(scenario_file_path=active_scenario.file_path,
                               objective_function=self._objective_function,
                               opt_method=OptimizationMethod.OPT_IDDFS,
                               max_steps=20,
                               verbose=True,
                               progress_updater=progress_updater)

        active_scenario.opt_result_dict[OptimizationMethod.OPT_IDDFS] = opt_result

        progress_updater.finish()

    def run_nsga2(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting NSGA-II...")

        nsga_config = NSGAConfig(max_sequence_length=self.spinBox_opt_nsga2_max_sequence_length.value(),
                                 population_size=self.spinBox_opt_nsga2_population.value(),
                                 number_generations=self.spinBox_opt_nsga2_generations.value(),
                                 mutation_probability=(self.doubleSpinBox_opt_nsga2_mutation_probability.value() / 100),
                                 penalty=1e4)

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result = run_nsga(scenario_file_path=active_scenario.file_path,
                              objective_function=self._objective_function,
                              opt_method=OptimizationMethod.OPT_NSGA2,
                              nsga_config=nsga_config,
                              progress_updater=progress_updater)

        active_scenario.opt_result_dict[OptimizationMethod.OPT_NSGA2] = opt_result

        progress_updater.finish()


    def run_nsga3(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting NSGA-III...")

        nsga_config = NSGAConfig(max_sequence_length=self.spinBox_opt_nsga3_max_sequence_length.value(),
                                 population_size=self.spinBox_opt_nsga3_population.value(),
                                 number_generations=self.spinBox_opt_nsga3_generations.value(),
                                 mutation_probability=(self.doubleSpinBox_opt_nsga3_mutation_probability.value() / 100),
                                 penalty=1e4)

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result = run_nsga(scenario_file_path=active_scenario.file_path,
                              objective_function=self._objective_function,
                              opt_method=OptimizationMethod.OPT_NSGA3,
                              nsga_config=nsga_config,
                              progress_updater=progress_updater)

        active_scenario.opt_result_dict[OptimizationMethod.OPT_NSGA3] = opt_result

        progress_updater.finish()

    def run_rl_dqn_run_training_and_inference(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting RL DQN Training & Inference...")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        # Fall back to default console output during training
        sys.stdout = sys.__stdout__

        print_with_timestamp(f"FILE: {active_scenario.file_path}")

        if not DEBUG_MODE:

            training_duration_seconds = run_training(scenario_file_path=active_scenario.file_path)

            inf_opt_result = run_inference(scenario_file_path=active_scenario.file_path,
                                       objective_function=self._objective_function,
                                       use_best_model=True,
                                       count=1, quick_eval=False)

            # TODO This should be improved
            combined_opt_result = OptimizationResult(action_idx_sequence=inf_opt_result.action_idx_sequence,
                                                     task_result_list=inf_opt_result.task_result_list,
                                                     total_time=inf_opt_result.total_time,
                                                     total_energy=inf_opt_result.total_energy,
                                                     sequence_reliability=inf_opt_result.sequence_reliability,
                                                     objective_function=inf_opt_result.objective_function,
                                                     other_params_dict={"reward": inf_opt_result.other_params_dict["reward"],
                                                                        "inference_duration": inf_opt_result.total_duration_seconds},
                                                     total_duration_seconds=training_duration_seconds,
                                                     opt_method=OptimizationMethod.OPT_RL_DQN)

            active_scenario.opt_result_dict[OptimizationMethod.OPT_RL_DQN] = combined_opt_result

        # Use text edit widget for console output again
        sys.stdout = self._pyqt_log_stream

        progress_updater.finish()

    def run_llm_agent_prompt(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting optimization using LLM Agent...")

        active_scenario = self._loaded_scenario_list[self._active_scenario_idx]

        opt_result = run_iterative_llm_scheduler(scenario_file_path=active_scenario.file_path, progress_updater=progress_updater)

        active_scenario.opt_result_dict[OptimizationMethod.OPT_LLM_AGENT] = opt_result

        progress_updater.finish()

    def _on_a_star_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_A_STAR)

        self.pushButton_opt_a_star_run.setEnabled(True)

    def _on_dijkstra_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_DIJKSTRA)

        self.pushButton_opt_dijkstra_run.setEnabled(True)

    def _on_dfs_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_DFS)

        self.pushButton_opt_dfs_run.setEnabled(True)

    def _on_iddfs_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_IDDFS)

        self.pushButton_opt_iddfs_run.setEnabled(True)

    def _on_nsga2_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_NSGA2)

        self.pushButton_opt_nsga2_run.setEnabled(True)

    def _on_nsga3_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_NSGA3)

        self.pushButton_opt_nsga3_run.setEnabled(True)

    def _on_rl_dqn_training_and_inference_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_RL_DQN)

        self.pushButton_opt_rl_dqn_run_training_and_inference.setEnabled(True)

    def _on_llm_agent_prompt_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_LLM_AGENT)

        self.pushButton_opt_llm_agent_run_prompt.setEnabled(True)

    def _on_reload_all_scenarios_complete(self) -> None:

        self.pushButton_scenario_overview_reload.setEnabled(True)

        # Defensive guard: without any loaded scenario, initializing would index into an empty list
        if len(self._loaded_scenario_list) == 0:
            print_with_timestamp("No scenarios were loaded - nothing to initialize.")
            return

        self._active_scenario_idx = 0
        self._initialize_scenario()
        self._update_table_widget_scenario_overview()
        print_with_timestamp("Reloaded all scenarios...")