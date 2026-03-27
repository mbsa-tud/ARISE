# -*- coding: utf-8 -*-

"""
Module defining a function used for adding a timestamp to console output.

Author: Patrick Fischer
Version: 0.0.3
"""

from datetime import datetime

def print_with_timestamp(output_text: str) -> None:

    print(f"<{datetime.now().strftime("%H:%M:%S.%f")[:-4]}> {output_text}")
