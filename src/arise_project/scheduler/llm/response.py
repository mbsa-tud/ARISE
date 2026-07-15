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
