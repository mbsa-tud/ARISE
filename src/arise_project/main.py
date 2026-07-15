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

Main module of the ICM ARISE Project

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.model.cost_normalization import compute_cost_scales
from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH

from src.arise_project.scheduler.factory_dqn_training import run_training, run_inference


if __name__ == "__main__":

    training_bool = False

    scenario_file_path = FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH

    time_scale, energy_scale, reliability_scale = compute_cost_scales(
        ScenarioCore(file_path=scenario_file_path, reset_class=True))

    # TODO use objective function for training
    objective_function = ObjectiveFunction(time_weight=1/3,
                                           energy_weight=1/3,
                                           reliability_weight=1/3,
                                           time_scale=time_scale,
                                           energy_scale=energy_scale,
                                           reliability_scale=reliability_scale)

    if training_bool:
        run_training(scenario_file_path=scenario_file_path)
    else:
        run_inference(scenario_file_path=scenario_file_path,
                      objective_function=objective_function,
                      count=5, quick_eval=True, use_best_model=False)

