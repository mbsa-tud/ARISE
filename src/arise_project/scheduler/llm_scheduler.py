# -*- coding: utf-8 -*-

"""
Scheduling using a large language model (LLM) via API

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

import json
from jsonschema import validate, ValidationError


from arise_project.config.paths import FILE_SCENARIO_SIMPLE_PLATE_FACTORY_PATH, FILE_SCENARIO_JSON_SCHEMA_PATH
from src.arise_project.scheduler.llm.api_client import OpenAIGPTClient

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

    # main()
    run_llm_scheduler()