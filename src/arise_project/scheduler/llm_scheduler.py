# -*- coding: utf-8 -*-

"""
Scheduling using a large language model (LLM) via API

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import time
import json
from pathlib import Path

from jsonschema import validate, ValidationError
from openai import OpenAI

from src.arise_project.tools.output_timestamp import print_with_timestamp
from src.arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, FILE_SCENARIO_JSON_SCHEMA_PATH
from src.arise_project.gui.custom.pyqt_progress_updater import PyQtProgressUpdater, DummyProgressUpdater
from src.arise_project.model.objective import ObjectiveFunction
from src.arise_project.model.cost_normalization import compute_cost_scales
from src.arise_project.model.optimization_method import OptimizationMethod
from src.arise_project.model.optimization_result import OptimizationResult
from src.arise_project.model.scenario import ScenarioCore
from src.arise_project.model.task_results import TaskResult
from src.arise_project.scheduler.llm.api_client import OpenAIGPTClient


OPT_RES_PARAM_REASONS = "reasons"
OPT_RES_PARAM_TOTAL_TOKEN_COUNT = "total_token_count"
OPT_RES_PARAM_AVG_RESPONSE_TIME = "average_response_time"


class PlannerAgent:

    def __init__(self, client: OpenAI, model="gpt-5"):

        self.client = client
        self.model = model
        self.conversation_id = None

        self.schema = {
                          "type": "object",
                          "properties": {
                            "selected_index": {
                              "type": "integer"
                            },
                            "reason": {
                              "type": "string"
                            }
                          },
                          "required": [
                            "selected_index",
                            "reason"
                          ],
                          "additionalProperties": False
                        }

    def step(self, base_prompt: str, state: dict, actions: list[TaskResult]) -> tuple[dict, int, float | int]:

        actions_str = self.format_actions(actions)

        prompt = f"""
                    {base_prompt}
                    
                    OBJECTIVE (VERY IMPORTANT):
                    Prioritize:
                    - low total time
                    - low total energy
                    - high reliability
                    - minimal transport steps
                    
                    CURRENT PLAN:
                    {json.dumps(state["history"], indent=2)}
                    
                    AVAILABLE ACTIONS:
                    {actions_str}
                    
                    Choose EXACTLY ONE action by returning its index.
                    
                    Rules:
                    - Only choose an index from the list
                    - Prefer actions that move toward goal completion
                    - Consider long-term impact, not just immediate gain
                    
                    Return JSON ONLY.
                """

        start = time.time()

        response = self.client.responses.create(
            model=self.model,
            conversation=self.conversation_id,
            input=[
                {"role": "system",
                 "content": "You are an expert industrial planner optimizing multi-objective production processes."},
                {"role": "user", "content": prompt}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "planner_step",
                    "schema": self.schema,
                }
            }
        )

        end = time.time()

        self.conversation_id = response.conversation

        decision = json.loads(response.output_text)

        usage = response.usage
        total_token_count = usage.input_tokens + usage.output_tokens

        response_time_seconds = end - start

        return decision, total_token_count, response_time_seconds

    def format_actions(self, task_result_list: list[TaskResult]):

        lines = []

        for i, task_result in enumerate(task_result_list):
            lines.append(f"{i}: {task_result.get_long_str()}")

        return "\n".join(lines)


def run_planner(agent: PlannerAgent,
                base_prompt, scenario: ScenarioCore,
                objective_function: ObjectiveFunction,
                max_steps=20, progress_updater: PyQtProgressUpdater = DummyProgressUpdater()):

    progress_updater.percentage = 0
    progress_updater.text = "Starting LLM planning..."

    start_time = time.time()

    state = {"history": []}
    result_path = []
    total_token_count_sum = 0
    total_response_time_seconds = 0

    for step in range(max_steps):

        progress_updater.percentage = int(round((step / max_steps) * 100, 0))
        progress_updater.text = f"Deciding {step + 1} of {max_steps} (max)"

        current_task_result_list = scenario.get_actions()
        current_action_idx_list = scenario.get_feasible_actions_idx_list()

        if not current_task_result_list:
            print("No more actions available.")
            break

        decision, total_token_count, response_time_seconds = agent.step(base_prompt, state, current_task_result_list)

        total_token_count_sum += total_token_count
        total_response_time_seconds += response_time_seconds

        idx = decision["selected_index"]

        if idx < 0 or idx >= len(current_task_result_list):
            print("Invalid index from model!")
            break

        chosen_action = current_task_result_list[idx]
        real_action_idx = current_action_idx_list[idx]

        # Apply action in your simulator
        scenario.step_by_action_idx(real_action_idx)
        result_path.append(real_action_idx)

        # Track history
        state["history"].append({
            "step": step,
            "chosen_index": idx,
            "action": chosen_action.get_long_str(),
            "reason": decision.get("reason", "")
        })

        print_with_timestamp(f"LLM Agent: Step {step + 1} -> Action idx: {real_action_idx} - {str(chosen_action)} -> {decision.get("reason", "")}")

        if scenario.is_done():
            print_with_timestamp("Goal reached!")
            break

    average_response_time_seconds = total_response_time_seconds / len(result_path)

    progress_updater.percentage = 100
    progress_updater.text = "Done."

    return OptimizationResult(action_idx_sequence=result_path,
                              task_result_list=scenario.task_result_history,
                              total_time=scenario.time_sum,
                              total_energy=scenario.energy_sum,
                              sequence_reliability=scenario.sequence_reliability,
                              objective_function=objective_function,
                              other_params_dict={OPT_RES_PARAM_REASONS: str(state["history"]),
                                                 OPT_RES_PARAM_TOTAL_TOKEN_COUNT: total_token_count_sum,
                                                 OPT_RES_PARAM_AVG_RESPONSE_TIME: average_response_time_seconds},
                              total_duration_seconds=(time.time() - start_time),
                              opt_method=OptimizationMethod.OPT_LLM_AGENT)


def run_iterative_llm_scheduler(scenario_file_path: Path,
                                progress_updater: PyQtProgressUpdater = DummyProgressUpdater()) -> OptimizationResult | None:
    old_base_prompt_str = """
                          You are responsible for optimizing the production process within a simplified virtual factory. 
                          A skill-based manufacturing approach is used. This means the factory consists of machines that
                          each offer specific skills such as drilling or cutting. There are three types of machines: 
                          processing, storage and transport. More specifically, processing machines are subdivided into
                          cutting, drilling and milling machines. The target state of a product is defined 
                          as the set of tasks that need to be executed until completion as well as the target location. 
                          For example, the product starts as a metal plate that needs to have a hole drilled, a line cut into
                          it and milling done on it and it needs to end up in a specific storage machine. The starting state
                          consist of an empty set of completed tasks and a predefined storage machine as the starting location. 
                          The skill of a machine can only be used if the product has been transported to this machine. 
                          There may be more than one product. Each product, each task of a specific product and each skill of 
                          a machine has a unique identifier. In this context, an action is the execution of a specific skill 
                          to complete a specific task associated with a specific product. An action key is the set of 
                          product_id, task_id and skill_id values. The execution of an action will lead to costs such as 
                          energy consumption and elapsed time. In addition, each skill has a reliability value from 0.0 to 1.0. 
                          Goal is the multi-objective optimization of the action sequence which leads the product from a 
                          starting state to the target state. The associated time and energy costs for an action are calculated 
                          by multiplying the time and energy factors either by the length of the cut for a cutting skill, 
                          the radius of the hole for the cutting or milling skill (which is a simplified assumption). You 
                          are provided a JSON which contains details regarding the factory, the machines, their connections 
                          through transport machines as well as the products. Take your time and find an optimal sequence of
                          actions to get the product or products from the starting state to the target state. Respond in JSON 
                          by providing a sequence of action keys. Think about how to solve this problem.
                       """

    base_prompt_str = """
                        You are responsible for optimizing the production process within a simplified virtual factory. 
                        A skill-based manufacturing approach is used. This means the factory consists of machines that
                        each offer specific skills such as drilling or cutting. There are three types of machines: 
                        processing, storage and transport. More specifically, processing machines are subdivided into
                        cutting, drilling and milling machines. The skill of a machine can only be used if the product 
                        has been transported to this machine. There may be more than one product. Each product, each task 
                        of a specific product and each skill of a machine has a unique identifier. The execution of an action 
                        will lead to costs such as energy consumption and elapsed time. In addition, each skill has a 
                        reliability value from 0.0 to 1.0. Goal is the multi-objective optimization of the action sequence 
                        which leads the product from a starting state to the target state. You are provided a JSON which 
                        contains details regarding the factory, the machines, their connections through transport machines 
                        as well as the products. Take your time and find and analyze this setup. Your objective is to 
                        find an optimal sequence of actions to get the product or products from the starting state to the target state.
                        You will solve step by step and be provided all feasible actions in the current state as well
                        as all previous actions taken up to the current state. For example, try to move a product to
                        a machine for processing and then process it there before transporting it to another machine. 
                        In some cases, it may make sense to process a product multiple times in one machine. For example,
                        if three holes are to be drilled, it makes sense to move the product to a drilling machine and then
                        drill all three holes there. Don't just move a product aimlessly through a factory. Think 
                        about where a product needs to go beforehand and find a path to follow. 
                        Abbreviations are used, for example DM = drilling machine, MM = milling machine, 
                        CM = cutting machine, and also DS = drilling skill, MS = milling skill,
                        CS = cutting skill, TS = transport skill and very similarly for tasks as well.
                        
                    """

    progress_updater.text = "Start iterative LLM scheduling"
    progress_updater.percentage = 0

    scn = ScenarioCore(file_path=scenario_file_path, reset_class=True)

    # TODO modify and use existing client
    client = OpenAI() # OpenAIGPTClient()

    agent = PlannerAgent(client=client, model="gpt-5")

    time_scale, energy_scale, reliability_scale = compute_cost_scales(scn)

    # TODO this currently is not considered
    objective_function = ObjectiveFunction(time_weight=1/3,
                                           energy_weight=1/3,
                                           reliability_weight=1/3,
                                           time_scale=time_scale,
                                           energy_scale=energy_scale,
                                           reliability_scale=reliability_scale)

    opt_result = run_planner(agent=agent, base_prompt=base_prompt_str, objective_function=objective_function, scenario=scn, progress_updater=progress_updater)

    return opt_result


def run_llm_scheduler():

    client = OpenAIGPTClient()

    prompt_str = ("You are responsible for optimizing the production process within a simplified virtual factory. "
                  "A skill-based manufacturing approach is used. This means the factory consists of machines that"
                  "each offer specific skills such as drilling or cutting. There are three types of machines: "
                  "processing, storage and transport. More specifically, processing machines are subdivided into"
                  "cutting, drilling and milling machines. The target state of a product is defined "
                  "as the set of tasks that need to be executed until completion as well as the target location. "
                  "For example, the product starts as a metal plate that needs to have a hole drilled, a line cut into"
                  "it and milling done on it and it needs to end up in a specific storage machine. The starting state"
                  "consist of an empty set of completed tasks and a predefined storage machine as the starting location. "
                  "The skill of a machine can only be used if the product has been transported to this machine. "
                  "There may be more than one product. Each product, each task of a specific product and each skill of "
                  "a machine has a unique identifier. In this context, an action is the execution of a specific skill "
                  "to complete a specific task associated with a specific product. An action key is the set of "
                  "product_id, task_id and skill_id values. The execution of an action will lead to costs such as "
                  "energy consumption and elapsed time. In addition, each skill has a reliability value from 0.0 to 1.0. "
                  "Goal is the multi-objective optimization of the action sequence which leads the product from a "
                  "starting state to the target state. The associated time and energy costs for an action are calculated "
                  "by multiplying the time and energy factors either by the length of the cut for a cutting skill, "
                  "the radius of the hole for the cutting or milling skill (which is a simplified assumption). You "
                  " are provided a JSON which contains details regarding the factory, the machines, their connections "
                  "through transport machines as well as the products. Take your time and find an optimal sequence of"
                  "actions to get the product or products from the starting state to the target state. Respond in JSON "
                  " by providing a sequence of action keys. Think about how to solve this problem.")

    file_path = FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH

    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)

    with open(FILE_SCENARIO_JSON_SCHEMA_PATH, 'r') as schema_file:
        schema = json.load(schema_file)

    # Validate the JSON data against the schema
    try:
        validate(instance=data_dict, schema=schema)

    except ValidationError as e:
        print("Unable to load scenario due to JSON validation error:", e.message)

    prompt_str += f" {data_dict}"

    response = client.request_json_only_response(prompt=prompt_str,
                                                 temperature=1.0)

    print(response)



def main():

    client = OpenAIGPTClient()

    client.request_json_only_response(prompt="Give me an example response", temperature=1.0)


if __name__ == '__main__':
    run_iterative_llm_scheduler(scenario_file_path=FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH)
    # main()
    #run_llm_scheduler()