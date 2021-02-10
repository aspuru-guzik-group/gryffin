#!/usr/bin/env python 
from gryffin import Gryffin
import shutil
import numpy as np


def test_recommend():
    config = {
        "general": {
            "save_database": False,
            "scratch_dir": 'tests_scratch',
            "num_cpus": 1,
            "boosted": True,
            "sampling_strategies": 2,
            "random_seed": 42,
        },
        "parameters": [{"name": "param_0", "type": "continuous", "low": 0, "high": 1, "size": 1},
                       {"name": "param_1", "type": "continuous", "low": 0, "high": 1, "size": 1}],
        "objectives": [{"name": "obj", "goal": "minimize"}]
        }

    observations = [
        {'param_0': np.array([0.3]), 'param_1': np.array([0.4]), 'obj': 0.1},
        {'param_0': np.array([0.5]), 'param_1': np.array([0.6]), 'obj': 0.2},
    ]

    gryffin = Gryffin(config_dict=config)
    _ = gryffin.recommend(observations=observations)

    # cleanup folders
    shutil.rmtree('tests_scratch')
