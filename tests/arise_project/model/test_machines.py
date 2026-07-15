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
"""

import pytest

from src.arise_project.model.machines import DrillingMachine
from src.arise_project.model.tasks import DrillingTask
from src.arise_project.model.task_results import TaskResult
from src.arise_project.model.product import Plate
from src.arise_project.model.skills import Skill


def test_machines():

    # Create a drilling machine
    drill = DrillingMachine(name="drill", x=5, y=10)

    # Create a plate
    plate = Plate(width=10, height=10,
                  processing_tasks=[DrillingTask(center_x=2, center_y=2, radius=2)],
                  starting_location_id=drill.unique_id,
                  target_location_id=drill.unique_id)

    # Make sure the current & target state are correctly initialized
    assert len(plate.target_state.processing_tasks) == 1
    assert len(plate.current_state.processing_tasks) == 0
    assert len(plate.state_history_list) == 1

    print(f"\n\nProduct: {plate}")
    print(f"Initial state: {plate.current_state.processing_tasks} - Target state: {plate.target_state.processing_tasks} - State history: {plate.state_history_list}")

    # Make sure there is only one remaining task
    remaining_processing_tasks = plate.get_remaining_processing_tasks()
    assert len(remaining_processing_tasks) == 1

    print(f"Remaining processing tasks: {remaining_processing_tasks}")

    # Process plate using the drilling machine
    processing_task = list(remaining_processing_tasks)[0]

    task_result = drill.process(product=plate, task=processing_task)

    assert isinstance(task_result, TaskResult)

    assert (isinstance(task_result.skill, Skill) and isinstance(task_result.total_time, float) and
            isinstance(task_result.total_energy, float) and isinstance(task_result.success_bool, bool))

    print(f"\nDrilling done: {task_result}")

    # Due to the possibility that the execution of a processing task fails, consider both outcomes
    if task_result.success_bool:

        # Make sure the current & target state as well as the history are updated correctly
        assert len(plate.target_state.processing_tasks) == 1
        assert len(plate.current_state.processing_tasks) == 1
        assert len(plate.state_history_list) == 2

        # Make sure there are no more remaining tasks
        remaining_processing_tasks = plate.get_remaining_processing_tasks()
        assert len(remaining_processing_tasks) == 0

    else:

        # Make sure the current & target state as well as the history are not updated
        assert len(plate.target_state.processing_tasks) == 1
        assert len(plate.current_state.processing_tasks) == 0
        assert len(plate.state_history_list) == 1

        # Make sure there is still one more remaining task
        remaining_processing_tasks = plate.get_remaining_processing_tasks()
        assert len(remaining_processing_tasks) == 1

    print(f"\nCurrent state: {plate.current_state.processing_tasks} - Target state: {plate.target_state.processing_tasks} - State history: {plate.state_history_list}")
    plate.print_processing_history()

    print(f"\nRemaining processing tasks: {remaining_processing_tasks}")

    print("\nTest passed: Drilling machine processed product")
