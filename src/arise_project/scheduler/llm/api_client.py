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

Custom LLM api client for generating responses in plain text and in JSON.

Use of the "Singleton" design pattern to only allow single instances of API clients for prompt requests.

Made use of code from https://github.com/Patrick-Hummel/AI_Simscape_Model_Generator (same author)

Last modification: 20.11.2025
"""

__version__ = "1"
__author__ = "Patrick Fischer"

import json
import sys
from typing import Tuple

from jsonschema import validate, Draft202012Validator
from jsonschema.exceptions import ValidationError, SchemaError

from pathlib import Path
from time import time

from dotenv import load_dotenv

from openai import OpenAI

from src.arise_project.config.paths import DIR_LLM_SCHEDULER_DEFAULT_JSON_SCHEMA_FILE, DIR_LLM_ENV_FILE
from src.arise_project.scheduler.llm.response import ResponseData

# Load api-keys as environment variables (before other project imports)
load_dotenv(DIR_LLM_ENV_FILE)

# -- OPENAI --
OPENAI_GPT41 = "gpt-4.1"
OPENAI_GPT5 = "gpt-5"

# -- Cost calculation --
MODEL_PRICES_USD_PER_TOKEN_NOVEMBER_2025_DICT = {
    OPENAI_GPT41: {"input": 3/1e6, "output": 25/1e6},
    OPENAI_GPT5: {"input": 1.25/1e6, "output": 10/1e6}
}


class Singleton(type):
    def __init__(cls, name, bases, mmbs):
        super(Singleton, cls).__init__(name, bases, mmbs)
        # Instantiate lazily on first use instead of at class-definition/import time, so that
        # importing this module does not require an OpenAI API key - only actually using the
        # client does. This keeps the rest of the application working without an '.env' file.
        cls._instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance


class OpenAIGPTClient(metaclass=Singleton):

    def __init__(self):
        self.client = OpenAI()
        print("-> OpenAI API client created")

        # Load JSON schema and prepare to be added to function call parameter
        with open(DIR_LLM_SCHEDULER_DEFAULT_JSON_SCHEMA_FILE, 'r') as file:
            self.json_response_schema = json.load(file)

        self.json_response_schema.pop('$schema', None)

    def request(self, prompt: str, temperature: float = 1.0, model_name: str = OPENAI_GPT5) -> ResponseData:

        start_time = time()

        response = self.client.responses.create(
            model=model_name,
            temperature=temperature,
            input=prompt
        )

        response_data = ResponseData(response_str=response.output[0].content[0].text,
                                     input_tokens=response.usage.input_tokens,
                                     output_tokens=response.usage.output_tokens,
                                     time_seconds=time() - start_time,
                                     model_name=model_name)

        print_token_count_and_cost(response_data)

        return response_data

    def request_json_only_response(self, prompt: str, temperature: float = 1.0, model_name: str = OPENAI_GPT5) -> ResponseData:

        start_time = time()

        response = self.client.responses.create(
            model=model_name,
            temperature=temperature,
            input=[
                {"role": "system", "content": "Produce output that conforms exactly to the provided JSON schema."},
                {"role": "user", "content": prompt}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "task_plan",
                    "schema": self.json_response_schema,
                }
            }
        )

        response_data = ResponseData(response_str=response.output[1].content[0].text,
                                     input_tokens=response.usage.input_tokens,
                                     output_tokens=response.usage.output_tokens,
                                     time_seconds=time() - start_time,
                                     model_name=model_name)

        print_token_count_and_cost(response_data)

        data = validate_json_str(json_text=response_data.response_str, schema=self.json_response_schema)

        # TODO Remove - Debugging / Testing
        print(data)

        return response_data

    conversation_id = None

    def ask_llm(self, user_input: str):

        global conversation_id

        response = self.client.responses.create(
            model="gpt-5",
            conversation=conversation_id,  # <-- keeps context
            input=[
                {"role": "system", "content": "Always return valid JSON matching the schema."},
                {"role": "user", "content": user_input}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "task_plan",
                    "schema": self.json_response_schema,
                }
            }
        )

        # Save conversation ID for next turn
        conversation_id = response.conversation

        return response.output[0].content[0].text


def validate_json_str(json_text: str, schema: dict) -> dict:
    """
    Parses a JSON string and validates it against a JSON Schema.
    Returns the parsed dict if valid; raises ValidationError/SchemaError otherwise.
    """
    # Parse JSON text first
    data = json.loads(json_text)

    # Optional: pre-validate schema itself (useful in dev)
    Draft202012Validator.check_schema(schema)

    try:
        # Validate the JSON instance against the schema
        validate(instance=data, schema=schema)

    except ValidationError as err:

        sys.stderr.write(err.message + "\n")

    except SchemaError as err:

        sys.stderr.write(err.message + "\n")

    return data


def token_cost_calculation(input_tokens: int, output_tokens: int, model_name: str) -> Tuple[float, float]:

    if model_name not in MODEL_PRICES_USD_PER_TOKEN_NOVEMBER_2025_DICT:
        raise ValueError(f"Please define price per input/output token of {model_name}")

    prices = MODEL_PRICES_USD_PER_TOKEN_NOVEMBER_2025_DICT[model_name]

    input_token_cost = input_tokens * prices["input"]
    output_token_cost = output_tokens * prices["output"]

    return input_token_cost, output_token_cost


def print_token_count_and_cost(response_data: ResponseData) -> None:

    # Calculate cost of response
    input_cost, output_cost = token_cost_calculation(input_tokens=response_data.input_tokens,
                                                     output_tokens=response_data.output_tokens,
                                                     model_name=response_data.model_name)

    print(f"\n[MODEL: {response_data.model_name}\n"
          f"Input tokens: {response_data.input_tokens} -> USD {input_cost:.5f} \n"
          f"Output tokens: {response_data.output_tokens} -> USD {output_cost:.5f} \n"
          f"Total = {response_data.input_tokens + response_data.output_tokens} -> USD {input_cost + output_cost:.5f} \n"
          f"Response time: {response_data.time_seconds:.3f} s]")
