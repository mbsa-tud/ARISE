# -*- coding: utf-8 -*-

"""
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
