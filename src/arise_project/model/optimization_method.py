# -*- coding: utf-8 -*-

"""
Module defining the OptimizationMethod enums for the action sequence optimization.

Author: Patrick Fischer
Version: 0.0.3
"""

__author__ = "Patrick Fischer"
__version__ = "0.0.3"

from enum import Enum


class OptimizationMethod(Enum):

    OPT_A_STAR = "A_STAR"
    OPT_DIJKSTRA = "DIJKSTRA"
    OPT_DFS = "DFS"
    OPT_IDDFS = "IDDFS"
    OPT_NSGA2 = "NSGA2"
    OPT_NSGA3 = "NSGA3"
    OPT_RL_DQN = "RL_DQN"
    OPT_LLM_AGENT = "LLM_AGENT"
    OPT_HUMAN = "HUMAN"

    def get_short_name(self) -> str:

        match self:

            case OptimizationMethod.OPT_A_STAR:
                return "A*"
            case OptimizationMethod.OPT_DIJKSTRA:
                return "DIJKSTRA"
            case OptimizationMethod.OPT_DFS:
                return "DFS"
            case OptimizationMethod.OPT_IDDFS:
                return "IDDFS"
            case OptimizationMethod.OPT_NSGA2:
                return "NSGA-II"
            case OptimizationMethod.OPT_NSGA3:
                return "NSGA-III"
            case OptimizationMethod.OPT_RL_DQN:
                return "RL / DQN"
            case OptimizationMethod.OPT_LLM_AGENT:
                return "LLM AGENT"
            case OptimizationMethod.OPT_HUMAN:
                return "Human"

    def get_long_name(self) -> str:

        match self:

            case OptimizationMethod.OPT_A_STAR:
                return "A* Algorithm"
            case OptimizationMethod.OPT_DIJKSTRA:
                return "Dijkstra Algorithm"
            case OptimizationMethod.OPT_DFS:
                return "Depth-First Search"
            case OptimizationMethod.OPT_IDDFS:
                return "Iterative Deepening Depth-First Search"
            case OptimizationMethod.OPT_NSGA2:
                return "Non-dominated Sorting Genetic Algorithm (II)"
            case OptimizationMethod.OPT_NSGA3:
                return "Non-dominated Sorting Genetic Algorithm (III)"
            case OptimizationMethod.OPT_RL_DQN:
                return "Reinforcement Learning / Deep Q-Network"
            case OptimizationMethod.OPT_LLM_AGENT:
                return "Large Language Model Agent"
            case OptimizationMethod.OPT_HUMAN:
                return "Human (Simulation Results)"
