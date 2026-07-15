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

Main module of the ICM ARISE Project (GUI version)

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import sys

from src.arise_project.config.debug import DEBUG_MODE

# Import unused but necessary to avoid DLL load conflicts with PyQt6
if not DEBUG_MODE:
    from stable_baselines3 import DQN

from PyQt6 import QtWidgets

from src.arise_project.gui.custom.main_window_custom import Ui_MainWindow_Custom


def main():

    app = QtWidgets.QApplication(sys.argv)
    GUI = QtWidgets.QMainWindow()

    ui = Ui_MainWindow_Custom()
    ui.setupUi(GUI)

    GUI.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
