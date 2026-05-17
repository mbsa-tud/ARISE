# -*- coding: utf-8 -*-

"""
Main module of the ICM ARISE Project (GUI version)

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import sys

from src.arise_project.config.TEMP_DEBUGMODE import DEBUG_MODE

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
