# -*- coding: utf-8 -*-

"""
Module defining a class used for creating, starting and managing threads for execution of tasks next to the GUI.

Author: Patrick Fischer
Version: 0.0.3
"""

def duration_formatting(time_sec: float) -> str:
    """
    This function formats a duration in seconds according to the following format: "Xh Xmin Xs" or "Xs Xms"

    :param time_sec: A duration in seconds (float)
    :return: Formatted string (easily readable for humans)
    """

    time_string = ""
    show_milliseconds = time_sec < 60

    hours = 0
    minutes = 0
    seconds = 0
    milliseconds = 0

    # Calculate hours
    if time_sec >= 60*60:
        hours = int(time_sec / (60 * 60))
        time_sec -= hours * (60 * 60)

    # Calculate minutes
    if time_sec >= 60:
        minutes = int(time_sec / 60)
        time_sec -= minutes * 60

    # Calculate seconds and milliseconds
    if time_sec > 0:

        # Round seconds if milliseconds are not required
        if not show_milliseconds:

            seconds = round(time_sec)

            # Rounding up may increase larger unit of time
            if seconds == 60:
                minutes += 1
                seconds = 0

                if minutes == 60:
                    hours += 1
                    minutes = 0

        else:
            seconds = int(time_sec)
            time_sec -= seconds
            milliseconds = round(time_sec * 1e3)

    # Only add non-zero values to string
    if hours > 0:
        time_string += f"{hours:d}h "

    if minutes > 0 or hours > 0:
        time_string += f"{minutes:d}min "

    if seconds > 0 or minutes > 0:
        time_string += f"{seconds:d}s "

    if show_milliseconds:
        time_string += f"{milliseconds:d}ms"

    # Remove unnecessary leading or trailing spaces and return
    return time_string.strip()
