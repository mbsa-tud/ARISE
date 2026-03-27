# -*- coding: utf-8 -*-

"""
Module defining a class used to stream text output to a "Plain Text Edit" box in PyQT6.

Author: Patrick Fischer
Version: 0.0.3
"""

from PyQt6 import QtCore


class PyQtLogStream(QtCore.QObject):

    msg_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(PyQtLogStream, self).__init__(parent)

    def write(self, message_str):
        self.msg_signal.emit(str(message_str))
