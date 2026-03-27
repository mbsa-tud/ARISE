# -*- coding: utf-8 -*-

"""
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
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QTreeWidgetItem, QGroupBox, QVBoxLayout, QTreeWidget, QTableWidget, QTableWidgetItem, \
    QHeaderView, QFileDialog
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

from arise_project.gui.custom.pyqt_log_stream import PyQtLogStream
from arise_project.gui.custom.pyqt_progress_updater import PyQtProgressUpdater
from arise_project.gui.custom.thread_manager import ThreadManager
from arise_project.model.optimization_method import OptimizationMethod
from arise_project.model.optimization_result import OptimizationResult
from arise_project.model.objective import ObjectiveFunction
from arise_project.scheduler.a_star_search import astar_search
from arise_project.scheduler.depth_first_search import run_iddfs
from arise_project.scheduler.factory_dqn_training import run_inference, run_training
from arise_project.scheduler.genetic_algorithms import run_nsga
from arise_project.tools.duration_format import duration_formatting
from arise_project.tools.hash_generation import get_scenario_output_dir_path
from arise_project.tools.output_timestamp import print_with_timestamp

from src.arise_project.config.colors import COLOR_BY_SKILL_DICT
from src.arise_project.config.paths import DIR_DATA_INPUT_SCENARIOS_JSON_PATH, FILE_GUI_ICON_PATH, \
                                            FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, DIR_NAME_OPT_RESULTS

from src.arise_project.gui.generated.main_window_generated import Ui_MainWindow

from src.arise_project.model.machines import StorageMachine, \
    ProcessingMachine, TransporterMachine

from src.arise_project.model.scenario import Scenario

COL_OPT_RES_COMP_METHOD = "Method"
COL_OPT_RES_COMP_STEPS = "Steps"
COL_OPT_RES_COMP_TOTAL_TIME = "Total Time"
COL_OPT_RES_COMP_TOTAL_ENERGY = "Total Energy"
COL_OPT_RES_COMP_SEQUENCE_RELIABILITY = "Seq. Reliability"
COL_OPT_RES_COMP_TOTAL_COST = "Total Cost"
COL_OPT_RES_COMP_DURATION_SECONDS = "Duration"

COL_SIM_ACTIONS_PRODUCT = "Product"
COL_SIM_ACTIONS_TASK = "Task"
COL_SIM_ACTIONS_MACHINE = "Machine"
COL_SIM_ACTIONS_SKILL = "Skill"
COL_SIM_ACTIONS_SKILL_TYPE = "Skill Type"
COL_SIM_ACTIONS_TIME = "Time"
COL_SIM_ACTIONS_ENERGY = "Energy"
COL_SIM_ACTIONS_RELIABILITY = "Reliability"
COL_SIM_ACTIONS_NOTE = "Note"


class Ui_MainWindow_Custom(Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self._pyqt_log_stream = PyQtLogStream()

        self._sim_started = False
        self._sim_start_time = time.time()
        self._sim_action_idx_sequence = []

        self._factory_graph_fig = None
        self._factory_graph_canvas = None
        self._factory_graph_ax = None
        self._factory_graph_toolbar = None

        self._sim_scenario: Scenario | None = None
        self._opt_scenario: Scenario | None = None
        self._current_task_result_list = []
        self._current_action_idx_list = []
        self._selected_task_result = None
        self._selected_action_idx = 0

        self._scenario_file_path = FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH

        # TODO Refactor / configuration
        use_reliability = True

        if use_reliability:

            time_weight = 1 / 3
            energy_weight = 1 / 3
            reliability_weight = 1 / 3

        else:

            time_weight = 1 / 2
            energy_weight = 1 / 2
            reliability_weight = 0

        self._objective_function = ObjectiveFunction(time_weight=time_weight,
                                                     energy_weight=energy_weight,
                                                     reliability_weight=reliability_weight)

        self._opt_result_dict = {}

        self._thread_manager = ThreadManager()

    def setupUi(self, MainWindow):

        super().setupUi(MainWindow)

        # Set window icon
        MainWindow.setWindowIcon(QtGui.QIcon(str(FILE_GUI_ICON_PATH)))

        self._pyqt_log_stream.msg_signal.connect(self._on_log_stream_message)
        sys.stdout = self._pyqt_log_stream

        self.checkBox_factory_show_distances.stateChanged.connect(self.on_change_checkbox_factory_distances)

        self.tableWidget_sim_actions.cellClicked.connect(self.on_cell_selected)

        self.pushButton_sim_execute_action.clicked.connect(self.on_click_sim_execute_action)
        self.pushButton_sim_undo_last_action.clicked.connect(self.on_click_sim_undo_last_action)
        self.pushButton_sim_reset_scenario.clicked.connect(self.on_click_sim_reset_scenario)
        self.pushButton_sim_save_results.clicked.connect(self.on_click_sim_save_results)

        self.pushButton_opt_a_star_run.clicked.connect(self.on_click_opt_a_star_run)
        self.pushButton_opt_dfs_run.clicked.connect(self.on_click_opt_dfs_run)
        self.pushButton_opt_iddfs_run.clicked.connect(self.on_click_opt_iddfs_run)
        self.pushButton_opt_nsga2_run.clicked.connect(self.on_click_opt_nsga2_run)
        self.pushButton_opt_nsga3_run.clicked.connect(self.on_click_opt_nsga3_run)
        self.pushButton_opt_rl_dqn_training.clicked.connect(self.on_click_opt_rl_dqn_start_training)
        self.pushButton_opt_rl_dqn_run_inference.clicked.connect(self.on_click_opt_rl_dqn_run_inference)
        self.pushButton_opt_human_sim.clicked.connect(self.on_click_opt_human_sim)

        self.action_load_scenario_from_file.triggered.connect(self.on_action_load_scenario)
        self.action_save_scenario_to_file.triggered.connect(self.on_action_save_scenario)
        self.action_about.triggered.connect(self.on_action_show_about_dialog)

        self._init_graph_in_groupbox()

        self._initialize_scenario()

    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)

    def _initialize_scenario(self):

        self.plainTextEdit_log_output.clear()

        self._sim_started = False
        self._sim_start_time = time.time()
        self._sim_action_idx_sequence = []

        self._sim_scenario: Scenario | None = None
        self._opt_scenario: Scenario | None = None
        self._current_task_result_list = []
        self._current_action_idx_list = []
        self._selected_task_result = None
        self._selected_action_idx = 0

        self._opt_result_dict = {}

        self._thread_manager = ThreadManager()

        # Get scenario output directory path (based on hash value of scenario JSON file)
        self._scenario_output_dir_path = get_scenario_output_dir_path(scenario_file_path=self._scenario_file_path)
        self._scenario_output_dir_path.mkdir(parents=True, exist_ok=True)

        self._scenario_output_opt_result_dir_path = self._scenario_output_dir_path / DIR_NAME_OPT_RESULTS
        self._scenario_output_dir_path.mkdir(parents=True, exist_ok=True)

        # Load a scenario (product and factory)
        self._sim_scenario = Scenario(file_path=self._scenario_file_path, reset_class=True)
        print_with_timestamp(f"Loaded scenario for simulation: '{self._scenario_file_path.name}'")

        self._opt_scenario = Scenario(file_path=self._scenario_file_path, reset_class=True)
        print_with_timestamp(f"Loaded scenario for optimization: '{self._scenario_file_path.name}'")

        # Reset GUI
        for opt_method in OptimizationMethod:
            self._update_labels_opt_results(opt_method=opt_method, reset=True)

        self.tableWidget_sim_actions.clear()
        self.tableWidget_opt_a_star_results.clear()
        self.tableWidget_opt_dfs_results.clear()
        self.tableWidget_opt_iddfs_results.clear()
        self.tableWidget_opt_nsga2_results.clear()
        self.tableWidget_opt_nsga3_results.clear()
        self.tableWidget_opt_rl_dqn_results.clear()
        self.tableWidget_opt_human_results.clear()
        self.tableWidget_opt_comparison.clear()

        self.pushButton_sim_execute_action.setEnabled(False)
        self.pushButton_sim_undo_last_action.setEnabled(False)
        self.pushButton_sim_reset_scenario.setEnabled(False)
        self.pushButton_sim_save_results.setEnabled(False)

        self._update_tree_widget_factory()
        self._update_tree_widget_product()
        self._update_tree_widget_state()
        self._update_table_widget_sim_actions()

        self.label_sim_total_steps.setText(f"{self._sim_scenario.step_count}")
        self.label_sim_total_time.setText(f"{self._sim_scenario.time_sum:.2f}")
        self.label_sim_total_energy.setText(f"{self._sim_scenario.energy_sum:.2f}")
        self.label_sim_sequence_reliability.setText(f"{self._sim_scenario.sequence_reliability:.3f}")
        self.label_sim_total_cost.setText(f"{self._sim_scenario.calculate_total_cost(self._objective_function):.2f}")

        self._draw_graph_in_groupbox(self._sim_scenario.factory.create_digraph_stationary_machines(),
                                     labels=True,
                                     edge_labels=self.checkBox_factory_show_distances.isChecked(),
                                     node_size=800)

        self._update_table_widget_opt_comparison()

        self._load_opt_results()


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

    def _load_opt_results(self):

        for opt_method in OptimizationMethod:

            opt_result = OptimizationResult.pickle_load(scenario_dir=self._scenario_output_opt_result_dir_path,
                                                        opt_method=opt_method)

            if not opt_result is None:

                self._opt_result_dict[opt_method] = opt_result

                match opt_method:

                    case OptimizationMethod.OPT_A_STAR:
                        self._on_a_star_complete()
                    case OptimizationMethod.OPT_DFS:
                        self._on_dfs_complete()
                    case OptimizationMethod.OPT_IDDFS:
                        self._on_iddfs_complete()
                    case OptimizationMethod.OPT_NSGA2:
                        self._on_nsga2_complete()
                    case OptimizationMethod.OPT_NSGA3:
                        self._on_nsga3_complete()
                    case OptimizationMethod.OPT_RL_DQN:
                        self._on_rl_dqn_complete()
                    case OptimizationMethod.OPT_HUMAN:
                        self._update_opt_result(OptimizationMethod.OPT_HUMAN)

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

        for machine_key in self._sim_scenario.factory.machine_by_id_dict.keys():

            machine = self._sim_scenario.factory.machine_by_id_dict[machine_key]

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
                position_param_item.setText(0, f"Position: {machine.x:.2f}, {machine.y:.2f}")

            skills_child_item = QTreeWidgetItem(machine_child_item)
            skills_child_item.setText(0, "Skills")

            for skill_key in machine.skill_by_id_dict.keys():
                skill = machine.skill_by_id_dict[skill_key]

                skill_item = QTreeWidgetItem(skills_child_item)
                skill_item.setText(0, f"{skill.unique_id} [{type(skill).__name__}]")

                time_factor_item = QTreeWidgetItem(skill_item)
                time_factor_item.setText(0, f"Time factor: {skill.time_factor:.2f}")

                energy_factor_item = QTreeWidgetItem(skill_item)
                energy_factor_item.setText(0, f"Energy factor: {skill.energy_factor:.2f}")

                reliability_item = QTreeWidgetItem(skill_item)
                reliability_item.setText(0, f"Reliability: {skill.reliability:.3f}")

    def _update_tree_widget_product(self) -> None:
        """
        Function used to populate the tree widget containing a product definition
        :return: None
        """

        first_product = self._sim_scenario.get_sorted_product_list()[0]

        self.treeWidget_product.clear()

        product_child_item = QTreeWidgetItem(self.treeWidget_product)
        product_child_item.setText(0, "Product")
        product_child_item.setExpanded(True)

        specific_product_child_item = QTreeWidgetItem(product_child_item)
        specific_product_child_item.setText(0, f"{first_product.unique_id} "
                                               f"[{type(first_product).__name__}]")
        specific_product_child_item.setExpanded(True)

        count_child_item = QTreeWidgetItem(specific_product_child_item)
        count_child_item.setText(0, f"Count: {len(self._sim_scenario.get_sorted_product_list())}")

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

    def _update_tree_widget_state(self) -> None:

        self.treeWidget_sim_states.clear()

        product_child_item = QTreeWidgetItem(self.treeWidget_sim_states)
        product_child_item.setText(0, "Product State")
        product_child_item.setExpanded(True)

        for product in self._sim_scenario.get_sorted_product_list():

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

        if len(self._sim_scenario.task_result_history) > 0:

            for idx, task_result in enumerate(self._sim_scenario.task_result_history):

                specific_action_child_item = QTreeWidgetItem(action_history_child_item)
                specific_action_child_item.setText(0, f"{idx + 1}. {task_result.get_short_name(with_product=True, with_machine=True)}")

                task_result_time_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_time_child_item.setText(0, f"Time: {task_result.total_time:.2f}")

                task_result_energy_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_energy_child_item.setText(0, f"Energy: {task_result.total_energy:.2f}")

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

        self._current_task_result_list = self._sim_scenario.get_actions()
        self._current_action_idx_list = self._sim_scenario.get_feasible_actions_idx_list()

        action_data_dict = [{COL_SIM_ACTIONS_PRODUCT: task_result.product.unique_id,
                             COL_SIM_ACTIONS_TASK: task_result.task.unique_id,
                             COL_SIM_ACTIONS_MACHINE: task_result.machine.unique_id,
                             COL_SIM_ACTIONS_SKILL: f"{task_result.skill.unique_id}",
                             COL_SIM_ACTIONS_SKILL_TYPE: task_result.skill.type_name(),
                             COL_SIM_ACTIONS_TIME: task_result.total_time,
                             COL_SIM_ACTIONS_ENERGY: task_result.total_energy,
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

            case OptimizationMethod.OPT_HUMAN:
                table_widget = self.tableWidget_opt_human_results

            case _:
                raise NotImplementedError(f"Optimization method {opt_method} not implemented")

        opt_res = self._opt_result_dict[opt_method]
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
            label_total_time.setText("0.00")
            label_total_energy.setText("0.00")
            label_total_reliability.setText("1.000")
            label_total_cost.setText("0.00")
            label_total_duration.setText(f"0min 0s")
            label_total_timestamp.setText("")

        else:

            label_total_steps.setText(f"{self._opt_result_dict[opt_method].steps}")
            label_total_time.setText(f"{self._opt_result_dict[opt_method].total_time:.2f}")
            label_total_energy.setText(f"{self._opt_result_dict[opt_method].total_energy:.2f}")
            label_total_reliability.setText(f"{self._opt_result_dict[opt_method].sequence_reliability:.3f}")
            label_total_cost.setText(f"{self._opt_result_dict[opt_method].total_cost:.2f}")
            label_total_duration.setText(f"{duration_formatting(self._opt_result_dict[opt_method].total_duration_seconds)}")
            label_total_timestamp.setText(self._opt_result_dict[opt_method].get_timestamp_str())


    def _update_table_widget_opt_comparison(self) -> None:

        if len(self._opt_result_dict) == 0:
            return

        # Generate comparison dataframe
        opt_result_comp_data_dict_list = []

        for opt_result in self._opt_result_dict.values():

            if not isinstance(opt_result, OptimizationResult):
                continue

            opt_result_comp_data_dict = {COL_OPT_RES_COMP_METHOD: opt_result.opt_method.get_short_name(),
                                         COL_OPT_RES_COMP_STEPS: opt_result.steps,
                                         COL_OPT_RES_COMP_TOTAL_TIME: opt_result.total_time,
                                         COL_OPT_RES_COMP_TOTAL_ENERGY: opt_result.total_energy,
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

        most_recent_result_opt_method = list(self._opt_result_dict.keys())[0]
        most_recent_result_datetime = self._opt_result_dict[most_recent_result_opt_method].timestamp_dt

        for opt_method in self._opt_result_dict.keys():

            if self._opt_result_dict[opt_method].timestamp_dt > most_recent_result_datetime:
                most_recent_result_datetime = self._opt_result_dict[opt_method].timestamp_dt
                most_recent_result_opt_method = opt_method

        self.label_opt_comparison_timestamp.setText(self._opt_result_dict[most_recent_result_opt_method].get_timestamp_str())

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

                if ((col_name == COL_OPT_RES_COMP_TOTAL_TIME) or
                        (col_name == COL_OPT_RES_COMP_TOTAL_ENERGY) or
                        (col_name == COL_OPT_RES_COMP_TOTAL_COST)):

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

    def _update_opt_result(self, opt_method: OptimizationMethod):

        if self._opt_result_dict[opt_method] is None:
            return

        # Save results as pickle file
        self._opt_result_dict[opt_method].pickle_dump(output_directory=self._scenario_output_opt_result_dir_path)

        # Save exemplary task results as CSV
        self._opt_result_dict[opt_method].to_csv(output_directory=self._scenario_output_opt_result_dir_path)

        # Update table with exemplary task results
        self._update_table_widget_opt_results(opt_method)

        # Update result labels
        self._update_labels_opt_results(opt_method)

        # Update comparison table
        self._update_table_widget_opt_comparison()

    def _update_sim_all(self):

        self.label_sim_total_steps.setText(f"{self._sim_scenario.step_count}")
        self.label_sim_total_time.setText(f"{self._sim_scenario.time_sum:.2f}")
        self.label_sim_total_energy.setText(f"{self._sim_scenario.energy_sum:.2f}")
        self.label_sim_sequence_reliability.setText(f"{self._sim_scenario.sequence_reliability:.3f}")
        self.label_sim_total_cost.setText(f"{self._sim_scenario.calculate_total_cost(self._objective_function):.2f}")

        if self._sim_scenario.is_done():
            self.label_sim_is_done.setText("True")
            self.label_sim_is_done.setStyleSheet("color: forestgreen;")
            self.pushButton_sim_save_results.setEnabled(True)
        else:
            self.label_sim_is_done.setText("False")
            self.label_sim_is_done.setStyleSheet("color: crimson;")
            self.pushButton_sim_save_results.setEnabled(False)

        self._update_table_widget_sim_actions()
        self._update_tree_widget_state()

    def on_change_checkbox_factory_distances(self):

        self._draw_graph_in_groupbox(self._sim_scenario.factory.create_digraph_stationary_machines(),
                                     labels=True,
                                     edge_labels=self.checkBox_factory_show_distances.isChecked(),
                                     node_size=800)

    def on_cell_selected(self, row_selected, column_selected):
        self._selected_task_result = self._current_task_result_list[row_selected]
        self._selected_action_idx = self._current_action_idx_list[row_selected]
        self.label_sim_selected_action.setText(str(self._selected_task_result.get_short_name(with_product=True, with_machine=True)))
        self.pushButton_sim_execute_action.setEnabled(True)

    def on_click_sim_execute_action(self):

        if not self._sim_started:

            self._sim_started = True
            self._sim_start_time = time.time()
            self.pushButton_sim_reset_scenario.setEnabled(True)
            self.pushButton_sim_undo_last_action.setEnabled(True)

        self._sim_action_idx_sequence.append(self._selected_action_idx)
        self._sim_scenario.execute_action(self._selected_task_result)

        self.pushButton_sim_execute_action.setEnabled(False)

        self._update_sim_all()

    def on_click_sim_undo_last_action(self):

        self._sim_scenario.undo_last_action()

        # TODO move action idx sequence to Scenario
        if len(self._sim_action_idx_sequence) > 0:
            self._sim_action_idx_sequence.pop(-1)

        if len(self._sim_scenario.task_result_history) == 0:
            self.pushButton_sim_undo_last_action.setEnabled(False)

        self._update_sim_all()

    def on_click_sim_reset_scenario(self):

        self._sim_scenario.reset()

        self._sim_started = False
        self._sim_start_time = time.time()
        self._sim_action_idx_sequence = []

        self.pushButton_sim_execute_action.setEnabled(False)
        self.pushButton_sim_undo_last_action.setEnabled(False)

        self._update_sim_all()

    def on_click_sim_save_results(self):

        if self._sim_scenario.is_done():


            opt_result = OptimizationResult(action_idx_sequence=list(self._sim_action_idx_sequence),
                                            task_result_list=self._sim_scenario.task_result_history,
                                            total_time=self._sim_scenario.time_sum,
                                            total_energy=self._sim_scenario.energy_sum,
                                            sequence_reliability=self._sim_scenario.sequence_reliability,
                                            objective_function=self._objective_function,
                                            other_params_dict={},
                                            total_duration_seconds=(time.time() - self._sim_start_time),
                                            opt_method=OptimizationMethod.OPT_HUMAN)

            self._opt_result_dict[OptimizationMethod.OPT_HUMAN] = opt_result

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

    def on_click_opt_rl_dqn_start_training(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                              progress_bar_label=self.label_opt_progress,
                                              on_finished=self._on_rl_dqn_training_complete)

        self._thread_manager.run_thread(self.run_rl_dqn_start_training, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_rl_dqn_training.setEnabled(False)

    def on_click_opt_rl_dqn_run_inference(self):

        progress_updater = PyQtProgressUpdater(progress_bar=self.progressBar_opt,
                                              progress_bar_label=self.label_opt_progress,
                                              on_finished=self._on_rl_dqn_complete)

        self._thread_manager.run_thread(self.run_rl_dqn_inference, progress_updater=progress_updater,
                                        additional_kwargs=dict())

        self.pushButton_opt_rl_dqn_run_inference.setEnabled(False)

    def on_click_opt_human_sim(self):

        self.tabWidget_main.setCurrentIndex(2)

    def on_action_show_about_dialog(self):

        # TODO Add about dialog window
        pass

    def on_action_load_scenario(self):

        # Open a file dialog to select a folder, starting at the specified directory
        scenario_dir_path_str = QFileDialog.getExistingDirectory(caption="Select Scenario Directory",
                                                                 directory=str(DIR_DATA_INPUT_SCENARIOS_JSON_PATH))

        scenario_dir_path = Path(scenario_dir_path_str)
        scenario_file_path = scenario_dir_path / f"{scenario_dir_path.name}.json"

        if scenario_file_path.exists():

           self._scenario_file_path = scenario_file_path
           self._initialize_scenario()

        else:
            print_with_timestamp(f"Error loading scenario '{scenario_file_path.name}', file does not exist...")


    def on_action_save_scenario(self):

        print_with_timestamp("Saving scenario... (Not yet implemented)")

    def run_a_star_search(self, progress_updater: PyQtProgressUpdater):

        opt_result  = astar_search(
            scenario_file_path=self._scenario_file_path,
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
                  f"time={opt_result.total_time:.4f} | "
                  f"energy={opt_result.total_energy:.4f} | "
                  f"reliability={opt_result.sequence_reliability:.6f} | "
                  f"expansions={opt_result.other_params_dict['expansions']} | "
                  f"elapsed={opt_result.total_duration_seconds:.2f}s\n")

        else:
            print_with_timestamp("[A*] No solution within limits.")

        self._opt_result_dict[OptimizationMethod.OPT_A_STAR] = opt_result

        progress_updater.finish()

    def run_dfs(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting DFS...")

        opt_result = run_iddfs(scenario_file_path=self._scenario_file_path,
                               objective_function=self._objective_function,
                               opt_method=OptimizationMethod.OPT_DFS,
                               max_steps=20,
                               verbose=True,
                               progress_updater=progress_updater)

        self._opt_result_dict[OptimizationMethod.OPT_DFS] = opt_result

        progress_updater.finish()

    def run_iddfs(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting IDDFS...")

        opt_result = run_iddfs(scenario_file_path=self._scenario_file_path,
                               objective_function=self._objective_function,
                               opt_method=OptimizationMethod.OPT_IDDFS,
                               max_steps=20,
                               verbose=True,
                               progress_updater=progress_updater)

        self._opt_result_dict[OptimizationMethod.OPT_IDDFS] = opt_result

        progress_updater.finish()

    def run_nsga2(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting NSGA-II...")

        opt_result = run_nsga(scenario_file_path=self._scenario_file_path,
                              objective_function=self._objective_function,
                              opt_method=OptimizationMethod.OPT_NSGA2,
                              progress_updater=progress_updater)

        self._opt_result_dict[OptimizationMethod.OPT_NSGA2] = opt_result

        progress_updater.finish()


    def run_nsga3(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting NSGA-III...")

        opt_result = run_nsga(scenario_file_path=self._scenario_file_path,
                              objective_function=self._objective_function,
                              opt_method=OptimizationMethod.OPT_NSGA3,
                              progress_updater=progress_updater)

        self._opt_result_dict[OptimizationMethod.OPT_NSGA3] = opt_result

        progress_updater.finish()

    def run_rl_dqn_start_training(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting RL DQN Start Training...")

        # Fall back to default console output during training
        sys.stdout = sys.__stdout__

        run_training(scenario_file_path=self._scenario_file_path)

        # Use text edit widget for console output again
        sys.stdout = self._pyqt_log_stream

        progress_updater.finish()

    def run_rl_dqn_inference(self, progress_updater: PyQtProgressUpdater):

        print_with_timestamp(f"Starting RL DQN Inference...")

        opt_result = run_inference(scenario_file_path=self._scenario_file_path,
                                   objective_function=self._objective_function,
                                   count=1, quick_eval=False)

        self._opt_result_dict[OptimizationMethod.OPT_RL_DQN] = opt_result

        progress_updater.finish()

    def _on_a_star_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_A_STAR)

        self.pushButton_opt_a_star_run.setEnabled(True)

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

    def _on_rl_dqn_training_complete(self) -> None:

        print_with_timestamp(f"RL DQN training complete")
        self.pushButton_opt_rl_dqn_training.setEnabled(True)

    def _on_rl_dqn_complete(self) -> None:

        self._update_opt_result(OptimizationMethod.OPT_RL_DQN)

        self.pushButton_opt_rl_dqn_run_inference.setEnabled(True)
