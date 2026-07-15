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

Module defining a class used for creating, starting and managing threads for execution of tasks next to the GUI.

Author: Patrick Fischer
Version: 0.0.3
"""

import threading
from typing import Any, Callable

from src.arise_project.gui.custom.pyqt_progress_updater import PyQtProgressUpdater


class ThreadManager:

    def __init__(self):
        self.threads = {}

    def run_thread(self, target: Callable[[Any], None], progress_updater: PyQtProgressUpdater = None, additional_kwargs: dict = None) -> None:

        if progress_updater is None:
            kwargs = {}
        else:
            kwargs = dict(progress_updater=progress_updater)

        if additional_kwargs is not None:
            kwargs.update(additional_kwargs)

        new_thread = threading.Thread(target=target, args=(), kwargs=kwargs, daemon=True)
        self.threads[target.__name__] = new_thread
        new_thread.start()
