import json
from pathlib import Path

import pytest

from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH
from src.arise_project.model.scenario import ScenarioCore, Scenario

TINY_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scenario.json"


def test_load_valid_scenario_builds_expected_factory_and_action_catalog():

    scn = ScenarioCore(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, reset_class=True)

    # 1 storage + 2 milling + 2 cutting + 2 drilling machines, plus 1 AGV transporter
    assert len(scn.factory.stationary_machine_by_id_dict) == 7
    assert len(scn.factory.transport_machine_by_id_dict) == 1

    # One product with 3 processing tasks (DT1, MT1, CT1)
    assert len(scn.product_by_id_dict) == 1
    product = scn.get_sorted_product_list()[0]
    assert len(product.target_state.processing_tasks) == 3

    # Action catalog ordering must be deterministic across independent loads (RL/search rely on this)
    scn2 = ScenarioCore(file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, reset_class=True)
    assert len(scn.sorted_action_catalog) > 0
    assert [a.task_id for a in scn.sorted_action_catalog] == [a.task_id for a in scn2.sorted_action_catalog]


def test_load_scenario_with_schema_violation_raises_value_error(tmp_path):

    # Missing the required top-level "product" key
    broken_scenario = {"factory": {"storage_machines": [], "processing_machines": [], "transporter_machines": []}}
    broken_scenario_path = tmp_path / "broken_scenario.json"
    broken_scenario_path.write_text(json.dumps(broken_scenario))

    with pytest.raises(ValueError, match="failed JSON schema validation"):
        ScenarioCore(file_path=broken_scenario_path, reset_class=True)


def test_load_additional_info_with_schema_violation_raises_value_error(tmp_path):

    broken_scenario = {"factory": {"storage_machines": [], "processing_machines": [], "transporter_machines": []}}
    broken_scenario_path = tmp_path / "broken_scenario.json"
    broken_scenario_path.write_text(json.dumps(broken_scenario))

    # Bypass __init__ (avoids creating real data directories) to unit-test this method in isolation,
    # it has the same validate-then-raise pattern as ScenarioCore._load_from_json
    scenario = object.__new__(Scenario)

    with pytest.raises(ValueError, match="failed JSON schema validation"):
        scenario._load_additional_info_from_json(broken_scenario_path)


def test_step_and_undo_round_trip():

    scn = ScenarioCore(file_path=TINY_SCENARIO_PATH, reset_class=True)

    # Only the transport action (ST1 -> DM1) is feasible from the starting storage location
    assert scn.get_feasible_actions_idx_list().tolist() == [1]

    # Action 0 (drilling) is illegal while at ST1 and must not change any state
    task_result, product_done, all_done = scn.step_by_action_idx(0)
    assert task_result is None
    assert product_done is None and all_done is None
    assert scn.step_count == 0

    # Golden path: transport to DM1 -> drill -> transport back to ST1
    scn.step_by_action_idx(1)
    assert scn.step_count == 1
    assert scn.time_sum == pytest.approx(1.0)
    assert scn.energy_sum == pytest.approx(1.0)

    _, product_done, all_done = scn.step_by_action_idx(0)
    assert scn.step_count == 2
    assert scn.time_sum == pytest.approx(3.0)
    assert product_done is False and all_done is False

    _, product_done, all_done = scn.step_by_action_idx(2)
    assert scn.step_count == 3
    assert scn.time_sum == pytest.approx(4.0)
    assert product_done is True and all_done is True
    assert scn.is_done()

    # Undo must exactly reverse the last action
    scn.undo_last_action()
    assert scn.step_count == 2
    assert scn.time_sum == pytest.approx(3.0)
    assert scn.energy_sum == pytest.approx(3.0)
    assert not scn.is_done()

    # Undoing back to the start must restore the initial (zeroed) cost state
    scn.undo_last_action()
    scn.undo_last_action()
    assert scn.step_count == 0
    assert scn.time_sum == pytest.approx(0.0)
    assert scn.energy_sum == pytest.approx(0.0)