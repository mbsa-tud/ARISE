# -*- coding: utf-8 -*-

"""
A class representing a response from a large language model.

Made use of code from https://github.com/Patrick-Hummel/AI_Simscape_Model_Generator (same author)

Last modification: 20.11.2025
"""

__version__ = "1"
__author__ = "Patrick Fischer"

from dataclasses import dataclass


@dataclass
class ResponseData:

    response_str: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    time_seconds: float = 0.0
    model_name: str = ""
