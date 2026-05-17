# -*- coding: utf-8 -*-

"""
Module defining the plots shown in the GUI.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from abc import ABC
from PyQt6 import QtCore
from PyQt6.QtWidgets import QWidget, QHBoxLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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


class AnalysisPlot(GroupBoxPlot):

    def __init__(self, groupBox):
        super().__init__(groupBox)

    def update_plot(self, data_df, column_name: str = None) -> None:

        # Clear plot first
        self._ax.cla()

        if isinstance(column_name, type(None)):
            return

        data_df.plot(ax=self._ax,# y=0,
                     kind="barh",
                     x="NAME",
                     y=column_name,
                     legend=True,
                     # linestyle="-", linewidth=2,
                     # color="red",
                     label=column_name)

        self._ax.set_xlabel("Seconds")
        self._ax.grid(color='grey', linestyle='--', linewidth=0.25, which="both")

        self._fig.tight_layout()

        self._canvas.draw()
