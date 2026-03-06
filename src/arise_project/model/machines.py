# -*- coding: utf-8 -*-

"""
Module containing the machine class definitions

Author: Patrick Fischer
Version: 0.0.2
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.2"

from abc import ABC, abstractmethod
from typing import Type

from src.arise_project.model.tasks import ProcessingTask, TransportTask, Task
from src.arise_project.model.task_results import TaskResult
from src.arise_project.model.product import Product
from src.arise_project.model.skills import DrillingSkill, MillingSkill, CuttingSkill, TransportSkill, StoreSkill, \
    RetrieveSkill, Skill


class Machine(ABC):
    """
    Abstract class defining an abstract machine
    """

    # Class instantiation counter for unique id generation (not used in abstract class)
    _unique_id_ctr: int = 0
    _ABBREVIATION = "XX"

    def __init__(self, name: str, unique_id: str, skill_set: set[Skill]) -> None:

        if len(skill_set) == 0:
            raise ValueError("Skill set cannot be empty")

        self._name = name
        self._unique_id = unique_id

        self._skill_set = skill_set.copy()   # Available skills (as a set)
        self._skill_by_id_dict = {}          # Available skills (as a dictionary by name)

        for skill in skill_set:
            self._skill_by_id_dict[skill.unique_id] = skill

        self._occupied_product = None

    def __eq__(self, other) -> bool:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Comparing objects by unique identifier.
        :param other: Object to compare
        :return: True if unique identifier matches, False otherwise
        """
        return isinstance(other, Machine) and self._unique_id == other.unique_id

    def __hash__(self) -> int:
        """
        Required to add object to a set. Using a set instead of a list as access is faster, order is not needed and
        there cannot be any duplicates in the set. Hashing the unique identifier to make sure hash is unique as well.
        :return: hash value of unique identifier (int)
        """
        return hash(self._unique_id)

    def __lt__(self, other):
        return self._unique_id < other.unique_id

    def __repr__(self) -> str:
        return f"{self._unique_id} ({self.__class__.__name__}): {list(self.skill_by_id_dict.keys())}"

    @classmethod
    def _generate_unique_id(cls) -> str:

        # Warning: Add semaphore for access control when implementing multiprocessing in the future
        # to avoid race conditions, otherwise unique id may not be unique
        cls._unique_id_ctr += 1

        unique_id = f"{cls._ABBREVIATION}{cls._unique_id_ctr:02d}"

        return unique_id

    @classmethod
    def reset_unique_id_ctr(cls) -> None:
        cls._unique_id_ctr = 0

    @classmethod
    def get_abbreviation(cls) -> str:
        return cls._ABBREVIATION

    @property
    def name(self) -> str:
        return self._name

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def skill_set(self) -> set[Skill]:
        return self._skill_set

    @property
    def skill_by_id_dict(self) -> dict[str, Skill]:
        return self._skill_by_id_dict

    @property
    def occupied_product(self) -> Product:
        return self._occupied_product

    def calc_task_energy(self, processing_task: ProcessingTask) -> tuple[float, Skill]:

        resulting_energy_value = 999_999
        resulting_used_skill = None

        # A task may be completed using different skills, therefore go through all possible skills
        for possible_skill_type in processing_task.possible_skill_types:

            # Get skill of the skill type
            skill = self.get_skill_by_type(possible_skill_type)

            # Skip if skill is not offered
            if skill is None:
                continue

            # Find skill with the lowest energy consumption
            if skill.energy_factor < resulting_energy_value:
                resulting_energy_value = skill.energy_factor
                resulting_used_skill = skill

                continue

        return resulting_energy_value, resulting_used_skill

    def calc_task_time(self, processing_task: ProcessingTask) -> tuple[float, Skill]:

        resulting_time_value = 999_999
        resulting_used_skill = None

        # A task may be completed using different skills, therefore go through all possible skills
        for possible_skill_type in processing_task.possible_skill_types:

            # Get skill of the skill type
            skill = self.get_skill_by_type(possible_skill_type)

            # Skip if skill is not offered
            if skill is None:
                continue

            # Find skill with the lowest time consumption
            if skill.energy_factor < resulting_time_value:
                resulting_time_value = skill.time_factor
                resulting_used_skill = skill

                continue

        return resulting_time_value, resulting_used_skill

    def calc_task_reliability(self, processing_task: ProcessingTask) -> tuple[float, Skill]:

        resulting_reliability_value = 0
        resulting_used_skill = None

        # A task may be completed using different skills, therefore go through all possible skills
        for possible_skill_type in processing_task.possible_skill_types:

            # Get skill of the skill type
            skill = self.get_skill_by_type(possible_skill_type)

            # Skip if skill is not offered
            if skill is None:
                continue

            # Find skill with the highest reliability
            if skill.energy_factor > resulting_reliability_value:
                resulting_reliability_value = skill.reliability
                resulting_used_skill = skill

                continue

        return resulting_reliability_value, resulting_used_skill

    def get_skill_by_type(self, skill_type: Type[Skill]) -> Skill | None:
        """
        Find a skill object of the provided type
        """

        for skill in self._skill_set:

            if isinstance(skill, skill_type):
                return skill

        return None

    @abstractmethod
    def calculate(self, product: Product, task: Task, skill: Skill = None) -> TaskResult:
        """
        Calculate the result of executing the specified task using an available skill, but DO NOT
        update the state of the product yet. This is done using the "process" method.

        :param product: product to process (Product)
        :param task: task to process (Task)
        :param skill: skill used for processing, optional (Skill | None)
        :return: task result (TaskResult)
        """

        pass

    @abstractmethod
    def process(self, product: Product, task: Task, skill: Skill = None) -> TaskResult:
        """
        Execute the specified task of the product using an available (or specified) skill.

        :param product: product to process (Product)
        :param task: task to process (ProcessingTask)
        :param skill: skill used for processing, optional (Skill | None)
        :return: task result (TaskResult)
        """
        pass


class StationaryMachine(Machine):
    """
    Stationary machine with a fixed (x, y) position.
    """

    def __init__(self, name: str, unique_id: str, skill_set: set[Skill], x: float = 0, y: float = 0) -> None:
        super().__init__(name=name, unique_id=unique_id, skill_set=skill_set)
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @abstractmethod
    def calculate(self, product: Product, task: Task, skill: Skill = None) -> TaskResult:
        """
        Calculate the result of executing the specified task using an available skill, but DO NOT
        update the state of the product yet. This is done using the "process" method.

        :param product: product to process (Product)
        :param task: task to process (Task)
        :param skill: skill used for processing, optional (Skill | None)
        :return: task result (TaskResult)
        """

        pass

    @abstractmethod
    def process(self, product: Product, task: Task, skill: Skill = None) -> TaskResult:
        """
        Execute the specified task of the product using an available (or specified) skill.

        :param product: product to process (Product)
        :param task: task to process (ProcessingTask)
        :param skill: skill used for processing, optional (Skill | None)
        :return: task result (TaskResult)
        """

        pass


class ProcessingMachine(StationaryMachine):
    """
    Stationary processing machine with fixed (x, y) position and location-enforced processing.
    """

    def __init__(self, name: str, unique_id: str, skill_set: set[Skill], x: float = 0, y: float = 0) -> None:
        super().__init__(name=name, unique_id=unique_id, skill_set=skill_set, x=x, y=y)

    def calculate(self, product: Product, task: ProcessingTask, skill: Skill = None) -> TaskResult:
        """
        Calculate the result of executing the specified processing task using an available skill, but DO NOT
        update the state of the product yet. This is done using the "process" method.

        :param product: product to process (Product)
        :param task: processing task to process (ProcessingTask)
        :param skill: skill used for processing, optional (Skill | None)
        :return: task result (TaskResult)
        """

        if skill is None:

            possible_skills_list = []

            # A task may be completed using different skills, therefore go through all possible skills
            for possible_skill_type in task.possible_skill_types:

                # Get skill of the skill type
                skill = self.get_skill_by_type(possible_skill_type)

                # Skip if skill is not offered
                if skill is None:
                    continue

                possible_skills_list.append(skill)

            if len(possible_skills_list) == 0:
                raise ValueError(
                    f"Machine '{self._unique_id}' has no skills that can execute processing task '{task.unique_id}'.")

            # TODO Develop logic to select which skill to use if multiple are applicable, right now use first one
            selected_skill: Skill = possible_skills_list[0]

        else:
            selected_skill = skill

        total_time, total_energy, success_bool = task.execute(selected_skill)

        return TaskResult(product=product,
                          machine=self,
                          task=task,
                          skill=selected_skill,
                          total_time=total_time,
                          total_energy=total_energy,
                          success_bool=success_bool)

    def process(self, product: Product, task: ProcessingTask, skill: Skill = None) -> TaskResult:
        """
        Execute the specified processing task of the product using an available (or specified) skill.

        :param product: product to process (Product)
        :param task: processing task to process (ProcessingTask)
        :param skill: skill used for processing, optional (Skill | None)
        :return: task result (TaskResult)
        """

        task_result = self.calculate(product=product, task=task, skill=skill)

        # If successful, update product state
        if task_result.success_bool:
            product.update_state_by_task_result(task_result=task_result)

        return task_result


class DrillingMachine(ProcessingMachine):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "DM"

    def __init__(self, name: str | None = None, x: float = 0.0, y: float = 0.0,
                 drilling_skill: DrillingSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = DrillingMachine._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        # In case no skill is provided, use the default skill parameters
        if drilling_skill is None:
            drilling_skill = DrillingSkill(time_factor=1.0, energy_factor=1.0, reliability=1.0)

        super().__init__(name=name, unique_id=unique_id, skill_set={drilling_skill}, x=x, y=y)


class MillingMachine(ProcessingMachine):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "MM"

    def __init__(self, name: str | None = None, x: float = 0.0, y: float = 0.0,
                 milling_skill: MillingSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = MillingMachine._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        # In case no skill is provided, use the default skill parameters
        if milling_skill is None:
            milling_skill = MillingSkill(time_factor=1.0, energy_factor=1.0, reliability=1.0)

        super().__init__(name=name, unique_id=unique_id, skill_set={milling_skill}, x=x, y=y)


class CuttingMachine(ProcessingMachine):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "CM"

    def __init__(self, name: str | None = None, x: float = 0.0, y: float = 0.0,
                 cutting_skill: CuttingSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = CuttingMachine._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        # In case no skill is provided, use the default skill parameters
        if cutting_skill is None:
            cutting_skill = CuttingSkill(time_factor=1.0, energy_factor=1.0, reliability=1.0)

        super().__init__(name=name, unique_id=unique_id, skill_set={cutting_skill}, x=x, y=y)


class TransporterMachine(Machine):
    """
    Mobile machines with transport capability. No fixed coordinates.
    """

    def __init__(self, name: str, unique_id: str, transport_skill: TransportSkill = None) -> None:

        # In case no skill is provided, use the default skill parameters
        if transport_skill is None:
            transport_skill = TransportSkill(time_factor=1.0, energy_factor=1.0, reliability=1.0)

        super().__init__(name=name, unique_id=unique_id, skill_set={transport_skill})

    def calculate(self, product: Product, task: TransportTask, skill: Skill = None) -> TaskResult:
        """
        Calculate the result of executing the specified transport task using a transport skill, but DO NOT
        update the location of the product yet. This is done using the "process" method.

        :param product: product to transport (Product)
        :param task: transport task to process (TransportTask)
        :param skill: skill used for transportation, optional (Skill | None)
        :return: task result (TransportTaskResult)
        """

        transport_skill = self.get_skill_by_type(TransportSkill)

        total_time, total_energy, success_bool = task.execute(selected_skill=transport_skill)

        return TaskResult(product=product,
                          machine=self,
                          task=task,
                          skill=transport_skill,
                          total_time=total_time,
                          total_energy=total_energy,
                          success_bool=success_bool)

    def process(self, product: Product, task: TransportTask, skill: Skill = None) -> TaskResult:
        """
        Execute the specified transport task of the product using an available (or specified) skill.

        :param product: product to process (Product)
        :param task: transport task to process (TransportTask)
        :param skill: skill used for transportion, optional (Skill | None)
        :return: task result (TaskResult)
        """

        task_result = self.calculate(product=product, task=task)

        # If successful, update product location
        if task_result.success_bool:
            product.update_state_by_task_result(task_result=task_result)

        return task_result


class ConveyorBelt(TransporterMachine):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "CB"

    def __init__(self, name: str | None = None, transport_skill: TransportSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = ConveyorBelt._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        super().__init__(name=name, unique_id=unique_id, transport_skill=transport_skill)


class AutomatedGuidedVehicle(TransporterMachine):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "AGV"

    def __init__(self, name: str | None = None, transport_skill: TransportSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = AutomatedGuidedVehicle._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        super().__init__(name=name, unique_id=unique_id, transport_skill=transport_skill)


class ThreeAxesRobot(TransporterMachine):

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "TAR"

    def __init__(self, name: str | None = None, transport_skill: TransportSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = ThreeAxesRobot._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        super().__init__(name=name, unique_id=unique_id, transport_skill=transport_skill)


class StorageMachine(StationaryMachine):
    """
    Stationary storage machine that holds multiple products in queue.
    """

    # Class instantiation counter for unique id generation
    _unique_id_ctr: int = 0
    _ABBREVIATION = "SM"

    def __init__(self, name: str | None = None, x: float = 0.0, y: float = 0.0,
                 store_skill: StoreSkill = None,
                 retrieve_skill: RetrieveSkill = None) -> None:

        # Generate a unique identifier for this specific class using the class abbreviation
        unique_id = StorageMachine._generate_unique_id()

        # In case no name is provided, use generated unique id
        if name is None or len(name) == 0:
            name = unique_id

        self._storage = []
        self._occupied_product = None

        # In case no skill is provided, use the default skill parameters
        if store_skill is None:
            store_skill = StoreSkill(time_factor=1.0, energy_factor=1.0, reliability=1.0)

        # In case no skill is provided, use the default skill parameters
        if retrieve_skill is None:
            retrieve_skill = RetrieveSkill(time_factor=1.0, energy_factor=1.0, reliability=1.0)

        super().__init__(name=name, unique_id=unique_id, skill_set={store_skill, retrieve_skill}, x=x, y=y)

    @property
    def storage(self) -> list[Product]:
        return self._storage

    def calculate(self, product: Product, task: Task, skill: Skill = None) -> TaskResult:
        """
        Calculate the result of executing the specified transport task using a transport skill, but DO NOT
        update the location of the product yet. This is done using the "process" method.

        :param product: product to transport (Product)
        :param task: task to process (Task)
        :param skill: skill used for transportation, optional (Skill | None)
        :return: task result (TransportTaskResult)
        """

        # TODO Add storage skill and storage task
        raise NotImplementedError()

    def process(self, product: Product, task: Task, skill: Skill = None) -> TaskResult:
        """
        Execute the specified transport task of the product using an available (or specified) skill.

        :param product: product to process (Product)
        :param task: task to process (Task)
        :param skill: skill used for transportion, optional (Skill | None)
        :return: task result (TaskResult)
        """

        # TODO Add storage skill and storage task
        raise NotImplementedError()
