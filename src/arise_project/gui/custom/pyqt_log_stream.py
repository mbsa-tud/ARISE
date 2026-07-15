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
