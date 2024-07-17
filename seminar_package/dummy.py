import numpy as np
from typing import Union, List, Dict

def parameter_generator() -> Dict[str, object]:
    """
    Method to generate fake parameters
    @return: the dict of parameters
    """
    return {
        "distance matrix": np.array([[0, 4, 40, 20],
                                    [6, 0, 35, 160],
                                    [60, 40, 0, 60],
                                    [20, 140, 70, 0]]),
        "travel time matrix": np.array([[1, 8, 50, 40],
                                        [10, 1, 75, 120],
                                        [40, 70, 1, 125],
                                        [50, 115, 114, 1]]),
        "driver trailer matrix": np.array([[0, 1], [1, 0], [0, 1]]),
        "forecast matrix": np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,0,0, 20, 20, 30, 30, 40, 40, 40, 40, 40, 40, 40, 30, 40, 40, 40, 30, 30, 20, 20, 10, 10],
                                     [0, 0, 0, 10, 10, 20, 20, 20, 30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 10, 10, 10, 0, 0]]),
        "driver cost": 0.4,
        "min interval": 480,
        "max driving": 480,
        "number driver": 3,
        "number trailer": 2,
        "number customer": 2,
        "trailer capacity": [500, 800],
        "trailer initial quantity": [100, 200],
        "trailer cost": 0.6,
        "setup time": [0, 30, 25, 35],
        "tank capacity": [float('inf'), float('inf'), 1000, 2000],
        "tank initial quantity": [0, 0, 600, 700],
        "safety level": [float('-inf'), float('-inf'), 200, 300],
        "time horizon": 1440,
        "time windows": np.array([[0, 0, 216],
                                  [0, 1200, 1440],
                                  [1, 0, 240],
                                  [1, 1976, 1440],
                                  [2, 240, 456]])
    }


def generate_solution():
    """
    Generates a JSON that can be used for checking feasiblity.
    """
    solution = [
        {
            "Driver index": 0,
            "Trailer index": 1,
            "Start time": 4,
            "Operations": [
                {
                "Location index": 1,
                "Arrival time": 12,
                "Quantity": -400
                },
                {
                "Location index": 2,
                "Arrival time": 123,
                "Quantity": 350
                }
            ]
        },
        {
            "Driver index": 1,
            "Trailer index": 0,
            "Start time": 8,
            "Operations": [
                {
                "Location index": 1,
                "Arrival time": 16,
                "Quantity": -350
                },
                {
                "Location index": 3,
                "Arrival time": 100,
                "Quantity": 400
                },
            ]
        },
        {
            "Driver index": 2,
            "Trailer index": 1,
            "Start time": 295,
            "Operations": [
                {
                "Location index": 1,
                "Arrival time": 299,
                "Quantity": -200
                },
                {
                "Location index": 3,
                "Arrival time": 360,
                "Quantity": 200
                }
            ]
        }
    ]
    return solution
