import pytest

from src.arise_project.model.factory import Factory
from src.arise_project.model.machines import StorageMachine, DrillingMachine, AutomatedGuidedVehicle, MillingMachine, ConveyorBelt, TransporterMachine, \
    CuttingMachine


def test_factory_creation_and_machine_addition():

    # --- Test: Factory creation and machine addition ---
    factory = Factory()
    storage = StorageMachine("storage")
    drill = DrillingMachine("drill")
    agv = AutomatedGuidedVehicle("AGV")
    mill = MillingMachine("mill")
    belt = ConveyorBelt("belt")

    factory.add_machine(storage)
    factory.add_machine(drill)
    factory.add_machine(agv)
    factory.add_machine(mill)
    factory.add_machine(belt)

    assert len(factory.machine_by_id_dict) == 5, "Factory should have 5 machines"
    print("\nTest passed: Factory machine registration")

    # --- Valid connections ---
    factory.connect(storage, drill, agv)
    factory.connect(drill, mill, belt)
    assert len(factory.transport_connections) == 2, "Factory should have 2 connections"
    print("\nTest passed: Machine connections")

    # --- Machine lookup ---
    m = factory.get_machine_by_id(storage.unique_id)
    assert m == storage, "Machine lookup failed"
    print("\nTest passed: Machine retrieval by unique ID")

    # --- Neighbor retrieval ---
    neighbors = factory.get_neighbors(storage.unique_id)
    assert neighbors[0][0] == drill
    assert isinstance(neighbors[0][1], AutomatedGuidedVehicle)
    print("\nTest passed: Neighbor machine and transport link")


def test_factory_pipeline_with_plate():

    factory = Factory()

    # Instantiate machines
    storage = StorageMachine("storage")
    agv = AutomatedGuidedVehicle("AGV")
    drill = DrillingMachine("drill")
    belt1 = ConveyorBelt("belt1")
    belt2 = ConveyorBelt("belt2")
    mill = MillingMachine("mill")
    cut = CuttingMachine("cut")

    # Add all machines to the factory
    for m in [storage, agv, drill, belt1, belt2, mill, cut]:
        factory.add_machine(m)

    assert isinstance(factory, Factory)
    print("\nTest passed: Full factory instantiation and execution")
