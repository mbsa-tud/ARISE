import pytest

from src.arise_project.model.machines import DrillingMachine, CuttingMachine, MillingMachine, ConveyorBelt, \
    AutomatedGuidedVehicle, ThreeAxesRobot, StorageMachine
from src.arise_project.model.tasks import DrillingTask, MillingTask, CuttingTask
from src.arise_project.model.product import Plate
from src.arise_project.model.skills import DrillingSkill, MillingSkill, CuttingSkill, TransportSkill, StoreSkill, \
    RetrieveSkill


def test_abbreviations():
    """
    Test: Makes sure that each abbreviation is unique.
    :return: None
    """

    class_list = [DrillingMachine, MillingMachine, CuttingMachine,
                  ConveyorBelt, AutomatedGuidedVehicle, ThreeAxesRobot, StorageMachine,
                  DrillingSkill, MillingSkill, CuttingSkill, TransportSkill, StoreSkill, RetrieveSkill,
                  DrillingTask, MillingTask, CuttingTask,
                  Plate]

    abbreviations_set = set()

    for class_type in class_list:

        assert hasattr(class_type, "_ABBREVIATION")
        assert class_type._ABBREVIATION not in abbreviations_set
        abbreviations_set.add(class_type._ABBREVIATION)

    print(f"\n{list(abbreviations_set)}")
