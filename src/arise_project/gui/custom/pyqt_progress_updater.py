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

Module defining a class used to stream updates from another thread to a progress bar in PyQT6.

https://stackoverflow.com/questions/66265219/how-to-update-pyqt-progressbar-from-an-independent-function-with-arguments

Author: Patrick Fischer
Version: 0.0.3
"""

from PyQt6 import QtCore
from PyQt6.QtWidgets import QProgressBar, QLabel

from typing import Callable, Union


class PyQtProgressUpdater(QtCore.QObject):

    percentageChanged = QtCore.pyqtSignal(int)
    textChanged = QtCore.pyqtSignal(str)

    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal(object)

    def __init__(self, progress_bar: QProgressBar | None = None,
                 progress_bar_label: QLabel | None = None,
                 on_finished: Union[Callable, None] = None,
                 parent: QtCore.QObject | None = None):

        super().__init__(parent)
        self._percentage = 0
        self._text = ""

        if progress_bar is not None:
            self.percentageChanged.connect(progress_bar.setValue)

        if progress_bar_label is not None:
            self.textChanged.connect(progress_bar_label.setText)

        if on_finished is not None:
            self.finished.connect(on_finished)

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value):
        if self._percentage == value:
            return
        self._percentage = value
        self.percentageChanged.emit(self.percentage)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        if self._text == value:
            return
        self._text = value
        self.textChanged.emit(self.text)

    def start(self):
        self.started.emit()

    def finish(self, value=None):
        self.finished.emit(value)


class DummyProgressUpdater:

    def start(self):
        pass

    def finish(self, value=None):
        pass

    @property
    def percentage(self):
        return 0

    @percentage.setter
    def percentage(self, value):
        pass

    @property
    def text(self):
        return 0

    @text.setter
    def text(self, value):
        pass
