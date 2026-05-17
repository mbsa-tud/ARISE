# -*- coding: utf-8 -*-

"""
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
