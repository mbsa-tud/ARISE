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

Module generating the result figures for the paper from the exported analysis_data.csv.

Reads the analysis export (GUI: "export analysis") from the analysis_task_count scenario
directory and writes the figures referenced by the paper into the docs/ directory.
Only scenarios with results for all three compared methods (A*, NSGA-II, RL DQN) are plotted.

Developed with the help of AI (partly AI-generated).

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Match the IEEE paper body font
matplotlib.rcParams["font.family"] = "Times New Roman"

from src.arise_project.config.paths import DIR_DATA_INPUT_SC_DIR_ANALYSIS_TASK_COUNT

FILE_ANALYSIS_CSV = DIR_DATA_INPUT_SC_DIR_ANALYSIS_TASK_COUNT / "analysis_data.csv"
DIR_OUTPUT = Path(__file__).resolve().parents[3] / "docs"

# Colorblind-safe hues (Okabe-Ito based), identity additionally encoded via distinct markers
COLOR_A_STAR = "#A31515"    # dark red
COLOR_NSGA2 = "#E69F00"     # orange
COLOR_RL_DQN = "#009E73"    # bluish green

COL_TASKS = "PT"
COL_A_STAR_TIME = "A* Time (s)"
COL_NSGA2_TIME = "NSGA2 Time (s)"
COL_RL_DQN_TIME = "RL DQN Time (s)"
COL_NSGA2_RATIO = "NSGA2 / A* Cost"
COL_RL_DQN_RATIO = "RL DQN / A* Cost"


def _new_axes():

    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300, layout="constrained")

    ax.set_xlabel("Number of processing tasks", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=10)

    return fig, ax


def plot_duration(df: pd.DataFrame, log_scale: bool, file_name: str) -> None:

    fig, ax = _new_axes()

    ax.plot(df[COL_TASKS], df[COL_A_STAR_TIME], color=COLOR_A_STAR, marker="o",
            markersize=4, linewidth=1.6, label="A*")

    ax.plot(df[COL_TASKS], df[COL_NSGA2_TIME], color=COLOR_NSGA2, marker="s",
            markersize=4, linewidth=1.6, label="NSGA-II (3 trials)")

    ax.plot(df[COL_TASKS], df[COL_RL_DQN_TIME], color=COLOR_RL_DQN, marker="^",
            markersize=4.5, linewidth=1.6, label="RL DQN (training)")

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Optimization duration (s, log scale)", fontsize=11)
    else:
        ax.set_ylabel("Optimization duration (s)", fontsize=11)

    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)

    output_path = DIR_OUTPUT / file_name
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_cost_ratio(df: pd.DataFrame, file_name: str) -> None:

    fig, ax = _new_axes()

    # A* is the optimum by construction: draw it as the 1.0 reference line
    ax.axhline(y=1.0, color=COLOR_A_STAR, linewidth=1.4, label="A* (optimum)")

    ax.plot(df[COL_TASKS], df[COL_NSGA2_RATIO], color=COLOR_NSGA2, marker="s",
            markersize=4, linewidth=1.6, label="NSGA-II (best of 3)")

    ax.plot(df[COL_TASKS], df[COL_RL_DQN_RATIO], color=COLOR_RL_DQN, marker="^",
            markersize=4.5, linewidth=1.6, label="RL DQN")

    ax.set_ylabel("Total cost relative to A* optimum", fontsize=11)
    ax.set_ylim(0.95, None)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)

    output_path = DIR_OUTPUT / file_name
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:

    df = pd.read_csv(FILE_ANALYSIS_CSV, sep=";")

    # Only scenarios where all three compared methods produced a result
    df = df[(df[COL_A_STAR_TIME] > 0) & (df[COL_NSGA2_TIME] > 0) & (df[COL_RL_DQN_TIME] > 0)]
    df = df.sort_values(COL_TASKS)

    print(f"Plotting {len(df)} scenarios ({df[COL_TASKS].min()}-{df[COL_TASKS].max()} tasks)")

    plot_duration(df, log_scale=True, file_name="fig_results_opt_duration_comp.png")
    plot_duration(df, log_scale=False, file_name="fig_results_opt_duration_comp_linear.png")
    plot_cost_ratio(df, file_name="fig_results_cost_ratio_comp.png")

    # Console summary of the numbers cited in the paper text
    growth = (df[COL_A_STAR_TIME].iloc[-1] / df[COL_A_STAR_TIME].iloc[0]) ** (
        1 / (df[COL_TASKS].iloc[-1] - df[COL_TASKS].iloc[0]))
    print(f"A* mean growth factor per task: {growth:.2f}")

    for _, row in df.iterrows():
        n = row[COL_TASKS]
        faster_nsga = row[COL_NSGA2_TIME] < row[COL_A_STAR_TIME]
        faster_dqn = row[COL_RL_DQN_TIME] < row[COL_A_STAR_TIME]
        if faster_nsga or faster_dqn:
            print(f"{n} tasks: A* {row[COL_A_STAR_TIME]:.1f}s | NSGA2 {row[COL_NSGA2_TIME]:.1f}s"
                  f"{' (faster)' if faster_nsga else ''} | DQN {row[COL_RL_DQN_TIME]:.1f}s"
                  f"{' (faster)' if faster_dqn else ''}")


if __name__ == "__main__":
    main()
