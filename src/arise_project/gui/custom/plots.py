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

Module defining the plots shown in the GUI.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from abc import ABC

import pandas as pd
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabWidget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Cursor


class GroupBoxPlot(ABC):

    def __init__(self, groupBox):

        self._fig, self._ax = plt.subplots(1, 1)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setParent(groupBox)

        widget = QWidget()
        widget.setLayout(QHBoxLayout())

        widget.layout().setSpacing(0)
        widget.layout().setContentsMargins(0, 0, 0, 0)
        widget.layout().addWidget(self._canvas)

        # nav = NavigationToolbar2QT(self._canvas, widget, coordinates=False)
        # nav.setOrientation(QtCore.Qt.Orientation.Vertical)
        # nav.setStyleSheet("QToolBar { border: 0px }")
        # widget.layout().addWidget(nav)

        groupBox.setLayout(widget.layout())

        cursor = Cursor(self._ax, horizOn=True, vertOn=True, useblit=True, color='tab:red',
                        linewidth=1, linestyle='--')

        # self.fig.canvas.mpl_connect('button_press_event', self.matplotlib_cursor_onclick)


# Column names of the analysis dataframe fed into the analysis plots (mirrors the CSV export)
COL_TASKS = "PT"
COL_A_STAR_TIME = "A* Time (s)"
COL_NSGA2_TIME = "NSGA2 Time (s)"
COL_RL_DQN_TIME = "RL DQN Time (s)"
COL_NSGA2_RATIO = "NSGA2 / A* Cost"
COL_RL_DQN_RATIO = "RL DQN / A* Cost"

# Colorblind-safe hues (Okabe-Ito based); identity additionally encoded via distinct markers,
# matching tools/generate_paper_figures.py so the GUI plots look like the paper figures.
COLOR_A_STAR = "#A31515"    # dark red
COLOR_NSGA2 = "#E69F00"     # orange
COLOR_RL_DQN = "#009E73"    # bluish green


class AnalysisPlot:
    """Three matplotlib plots (shown as sub-tabs) reproducing the paper's result figures.

    The plots compare optimization duration (linear and log scale) and solution quality
    (total cost relative to the A* optimum) across A*, NSGA-II and RL DQN, over the number
    of processing tasks.
    """

    def __init__(self, groupBox):

        self._tab_widget = QTabWidget()

        self._fig_duration_linear, self._ax_duration_linear = self._new_tab("Duration (linear)")
        self._fig_duration_log, self._ax_duration_log = self._new_tab("Duration (log)")
        self._fig_cost_ratio, self._ax_cost_ratio = self._new_tab("Solution quality")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)
        groupBox.setLayout(layout)

        self.clear()

    def _new_tab(self, title: str):
        """Create a tab holding a single matplotlib canvas and return its (figure, axes)."""

        fig, ax = plt.subplots(1, 1)
        canvas = FigureCanvasQTAgg(fig)

        tab = QWidget()
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(canvas)
        tab.setLayout(tab_layout)

        self._tab_widget.addTab(tab, title)

        return fig, ax

    @staticmethod
    def _style_axes(ax) -> None:

        ax.set_xlabel("Number of processing tasks")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    def clear(self) -> None:
        """Reset all three plots to an empty state (used when no results are available)."""

        for fig, ax in ((self._fig_duration_linear, self._ax_duration_linear),
                        (self._fig_duration_log, self._ax_duration_log),
                        (self._fig_cost_ratio, self._ax_cost_ratio)):

            ax.cla()
            self._style_axes(ax)
            fig.tight_layout()
            fig.canvas.draw_idle()

    def update_plots(self, data_df: pd.DataFrame) -> None:

        if data_df is None or len(data_df) == 0:
            self.clear()
            return

        data_df = data_df.sort_values(COL_TASKS)

        self._plot_duration(self._fig_duration_linear, self._ax_duration_linear, data_df, log_scale=False)
        self._plot_duration(self._fig_duration_log, self._ax_duration_log, data_df, log_scale=True)
        self._plot_cost_ratio(self._fig_cost_ratio, self._ax_cost_ratio, data_df)

    def _plot_duration(self, fig, ax, df: pd.DataFrame, log_scale: bool) -> None:

        ax.cla()
        self._style_axes(ax)

        self._plot_series(ax, df, COL_A_STAR_TIME, COLOR_A_STAR, "o", "A*", positive_only=log_scale)
        self._plot_series(ax, df, COL_NSGA2_TIME, COLOR_NSGA2, "s", "NSGA-II (3 trials)", positive_only=log_scale)
        self._plot_series(ax, df, COL_RL_DQN_TIME, COLOR_RL_DQN, "^", "RL DQN (training)", positive_only=log_scale)

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Optimization duration (s, log scale)")
        else:
            ax.set_ylabel("Optimization duration (s)")

        ax.legend(loc="upper left", framealpha=0.9)

        fig.tight_layout()
        fig.canvas.draw_idle()

    def _plot_cost_ratio(self, fig, ax, df: pd.DataFrame) -> None:

        ax.cla()
        self._style_axes(ax)

        # A* is the optimum by construction: draw it as the 1.0 reference line
        ax.axhline(y=1.0, color=COLOR_A_STAR, linewidth=1.4, label="A* (optimum)")

        self._plot_series(ax, df, COL_NSGA2_RATIO, COLOR_NSGA2, "s", "NSGA-II (best of 3)")
        self._plot_series(ax, df, COL_RL_DQN_RATIO, COLOR_RL_DQN, "^", "RL DQN")

        ax.set_ylabel("Total cost relative to A* optimum")
        ax.set_ylim(0.95, None)
        ax.legend(loc="upper left", framealpha=0.9)

        fig.tight_layout()
        fig.canvas.draw_idle()

    @staticmethod
    def _plot_series(ax, df: pd.DataFrame, column_name: str, color: str, marker: str,
                     label: str, positive_only: bool = False) -> None:
        """Plot one method's series against the task count, skipping missing values."""

        if column_name not in df.columns:
            return

        series_df = df[[COL_TASKS, column_name]].dropna()

        # Log-scale duration plots cannot show non-positive durations
        if positive_only:
            series_df = series_df[series_df[column_name] > 0]

        if len(series_df) == 0:
            return

        ax.plot(series_df[COL_TASKS], series_df[column_name], color=color, marker=marker,
                markersize=4.5, linewidth=1.6, label=label)