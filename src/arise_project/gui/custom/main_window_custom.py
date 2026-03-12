# -*- coding: utf-8 -*-

"""
Module defining the main window GUI based on a class generated from a '.ui' file

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import networkx as nx
import pandas as pd
from PyQt6 import QtGui
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QTreeWidgetItem, QGroupBox, QVBoxLayout, QTreeWidget, QTableWidget, QTableWidgetItem, \
    QHeaderView

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

from src.arise_project.config.colors import COLOR_BY_SKILL_DICT
from src.arise_project.config.paths import FILE_GUI_ICON_PATH, FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.gui.generated.main_window_generated import Ui_MainWindow

from src.arise_project.model.machines import StorageMachine, \
    ProcessingMachine, TransporterMachine

from src.arise_project.model.scenario import Scenario
from src.arise_project.model.tasks import TransportTask, Task

OPT_ALGORITHMS = ["RL (DQN)", "NSGA-II", "IDDFS"]


class Ui_MainWindow_Custom(Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self._factory_graph_fig = None
        self._factory_graph_canvas = None
        self._factory_graph_ax = None
        self._factory_graph_toolbar = None

        self._scenario = None
        self._current_task_result_list = []
        self._selected_task_result = None

    def setupUi(self, MainWindow):

        super().setupUi(MainWindow)

        self.pushButton_execute_action.setEnabled(False)

        self.tableWidget_actions.cellClicked.connect(self.on_cell_selected)
        self.pushButton_execute_action.clicked.connect(self.on_click_execute_action)
        self.pushButton_undo_last_action.clicked.connect(self.on_click_undo_last_action)
        self.pushButton_reset_scenario.clicked.connect(self.on_click_reset_scenario)
        self.pushButton_run_optimization.clicked.connect(self.on_click_run_optimization)

        # Set window icon
        MainWindow.setWindowIcon(QtGui.QIcon(str(FILE_GUI_ICON_PATH)))

        # Load a scenario (product and factory)
        self._scenario = Scenario(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)

        self._update_factory_tree_widget()
        self._update_product_tree_widget()
        self._update_state_tree_widget()
        self._update_action_table_widget()

        self.label_step.setText(f"{self._scenario.step_count}")
        self.label_time.setText(f"{self._scenario.time_sum:.3f}")
        self.label_energy.setText(f"{self._scenario.energy_sum:.3f}")

        self.comboBox_optimization_algorithms.addItems(OPT_ALGORITHMS)

        self._init_graph_in_groupbox()
        self._draw_graph_in_groupbox(self._scenario.factory.create_digraph_stationary_machines(), labels=True, node_size=800)

    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)

    def _init_graph_in_groupbox(self) -> None:

        layout = self.groupBox_factory_plot.layout()

        if layout is None:
            layout = QVBoxLayout(self.groupBox_factory_plot)
            self.groupBox_factory_plot.setLayout(layout)

        self._factory_graph_fig = Figure(figsize=(5, 4), dpi=100)
        self._factory_graph_canvas = FigureCanvas(self._factory_graph_fig)
        self._factory_graph_ax = self._factory_graph_fig.add_subplot(111)
        self._factory_graph_toolbar = NavigationToolbar(self._factory_graph_canvas, self.groupBox_factory_plot)

        # Add toolbar and canvas to the group box
        layout.addWidget(self._factory_graph_toolbar)
        layout.addWidget(self._factory_graph_canvas)

    def _draw_graph_in_groupbox(self, graph: nx.Graph, labels: bool = True, node_size: int = 600) -> None:
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

            # Get labels from edge data and format the values
            edge_labels = {edge: f"{data['distance']:.3f}" for edge, data in graph.edges.items()}
            nx.draw_networkx_edge_labels(graph, pos_dict, ax=self._factory_graph_ax, edge_labels=edge_labels)

        # Keep your coordinate system
        self._factory_graph_ax.set_aspect("equal")
        self._factory_graph_ax.axis("off")
        self._factory_graph_fig.tight_layout()
        self._factory_graph_canvas.draw_idle()

    def _update_factory_tree_widget(self) -> None:
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

        for machine_key in self._scenario.factory.machine_by_id_dict.keys():

            machine = self._scenario.factory.machine_by_id_dict[machine_key]

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

    def _update_product_tree_widget(self) -> None:
        """
        Function used to populate the tree widget containing a product definition
        :return: None
        """

        first_product = self._scenario.get_sorted_product_list()[0]

        self.treeWidget_product.clear()

        product_child_item = QTreeWidgetItem(self.treeWidget_product)
        product_child_item.setText(0, "Product")
        product_child_item.setExpanded(True)

        specific_product_child_item = QTreeWidgetItem(product_child_item)
        specific_product_child_item.setText(0, f"{first_product.unique_id} "
                                               f"[{type(first_product).__name__}]")
        specific_product_child_item.setExpanded(True)

        count_child_item = QTreeWidgetItem(specific_product_child_item)
        count_child_item.setText(0, f"Count: {len(self._scenario.get_sorted_product_list())}")

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

    def _update_state_tree_widget(self) -> None:

        self.treeWidget_states.clear()

        product_child_item = QTreeWidgetItem(self.treeWidget_states)
        product_child_item.setText(0, "Product State")
        product_child_item.setExpanded(True)

        for product in self._scenario.get_sorted_product_list():

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

        action_history_child_item = QTreeWidgetItem(self.treeWidget_states)
        action_history_child_item.setText(0, "Action History")
        action_history_child_item.setExpanded(True)

        if len(self._scenario.executed_action_history) > 0:

            for idx, task_result in enumerate(self._scenario.executed_action_history):

                specific_action_child_item = QTreeWidgetItem(action_history_child_item)
                specific_action_child_item.setText(0, f"{idx + 1}. {task_result.get_short_name(with_product=True, with_machine=True)}")

                task_result_time_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_time_child_item.setText(0, f"Time: {task_result.total_time:.3f}")

                task_result_energy_child_item = QTreeWidgetItem(specific_action_child_item)
                task_result_energy_child_item.setText(0, f"Energy: {task_result.total_energy:.3f}")

        else:

            specific_action_child_item = QTreeWidgetItem(action_history_child_item)
            specific_action_child_item.setText(0, f"None")

    def _update_action_table_widget(self) -> None:
        """
        Function used to populate the table widget with all actions available at this step
        :return: None
        """

        self.tableWidget_actions.clear()

        self._current_task_result_list = self._scenario.get_actions()

        def get_note_per_task(task: Task) -> str:

            if isinstance(task, TransportTask):
                return f"-> {task.target_machine_id}"
            else:
                return "-"

        action_data_dict = [{"Product": task_result.product.unique_id,
                             "Task": task_result.task.unique_id,
                             "Machine": task_result.machine.unique_id,
                             "Skill": f"{task_result.skill.unique_id}",
                             "Skill Type": task_result.skill.type_name(),
                             "Time": task_result.total_time,
                             "Energy": task_result.total_energy,
                             "Note": get_note_per_task(task_result.task)}

                            for task_result in self._current_task_result_list]

        actions_df = pd.DataFrame(data=action_data_dict)

        self.tableWidget_actions.setRowCount(len(actions_df))
        self.tableWidget_actions.setColumnCount(len(actions_df.columns))
        self.tableWidget_actions.setHorizontalHeaderLabels(actions_df.columns.tolist())

        for row in range(len(actions_df)):

            for col in range(len(actions_df.columns)):

                value = str(actions_df.iat[row, col])
                item = QTableWidgetItem(value)

                if value in COLOR_BY_SKILL_DICT:
                    item.setForeground(QBrush(QColor(COLOR_BY_SKILL_DICT[value])))

                self.tableWidget_actions.setItem(row, col, item)

        header = self.tableWidget_actions.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _update_optimization_actions_table_widget(self) -> None:
        """
        Function used to populate the table widget with all actions available at this step
        :return: None
        """

        self.tableWidget_optimization_actions.clear()

        self._current_task_result_list = self._scenario.executed_action_history()

        def get_note_per_task(task: Task) -> str:

            if isinstance(task, TransportTask):
                return f"-> {task.target_machine_id}"
            else:
                return "-"

        action_data_dict = [{"Product": task_result.product.unique_id,
                             "Task": task_result.task.unique_id,
                             "Machine": task_result.machine.unique_id,
                             "Skill": f"{task_result.skill.unique_id}",
                             "Skill Type": task_result.skill.type_name(),
                             "Time": task_result.total_time,
                             "Energy": task_result.total_energy,
                             "Note": get_note_per_task(task_result.task)}

                            for task_result in self._current_task_result_list]

        actions_df = pd.DataFrame(data=action_data_dict)

        self.tableWidget_optimization_actions.setRowCount(len(actions_df))
        self.tableWidget_optimization_actions.setColumnCount(len(actions_df.columns))
        self.tableWidget_optimization_actions.setHorizontalHeaderLabels(actions_df.columns.tolist())

        for row in range(len(actions_df)):

            for col in range(len(actions_df.columns)):

                value = str(actions_df.iat[row, col])
                item = QTableWidgetItem(value)

                if value in COLOR_BY_SKILL_DICT:
                    item.setForeground(QBrush(QColor(COLOR_BY_SKILL_DICT[value])))

                self.tableWidget_optimization_actions.setItem(row, col, item)

        header = self.tableWidget_optimization_actions.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def update_all(self):

        self.label_step.setText(f"{self._scenario.step_count}")
        self.label_time.setText(f"{self._scenario.time_sum:.3f}")
        self.label_energy.setText(f"{self._scenario.energy_sum:.3f}")

        if self._scenario.is_done():
            self.label_is_done.setText("True")
            self.label_is_done.setStyleSheet("color: forestgreen;")
            self.pushButton_execute_action.setEnabled(False)
        else:
            self.label_is_done.setText("False")
            self.label_is_done.setStyleSheet("color: crimson;")

        self._update_action_table_widget()
        self._update_state_tree_widget()

    def on_cell_selected(self, row_selected, column_selected):
        self._selected_task_result = self._current_task_result_list[row_selected]
        self.label_selected_action.setText(str(self._selected_task_result.get_short_name(with_product=True, with_machine=True)))
        self.pushButton_execute_action.setEnabled(True)

    def on_click_execute_action(self):
        self._scenario.execute_action(self._selected_task_result)
        self.update_all()

    def on_click_undo_last_action(self):
        self._scenario.undo_last_action()
        self.update_all()

    def on_click_reset_scenario(self):
        self._scenario.reset()
        self.update_all()
        self.pushButton_execute_action.setEnabled(False)

    def on_click_run_optimization(self):
        pass

        # self._update_optimization_actions_table_widget()