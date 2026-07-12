# -*- coding: utf-8 -*-

"""
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

