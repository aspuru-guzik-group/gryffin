import os

import numpy as np

from gryffin import Gryffin
import olympus
from olympus import Surface

objective = Surface(kind='Dejong', param_dim=2)

def compute_objective(param):
    param['obj'] = objective.run([val for key, val in param.items()])[0][0]
    return param

config = {
    "general": {
        "random_seed": 42,
        "verbosity": 0,
        "boosted":  False,
    },
    "parameters": [
        {"name": "x_0", "type": "continuous", "low": 0.0, "high": 1.0},
        {"name": "x_1", "type": "continuous", "low": 0.0, "high": 1.0},
    ],
    "objectives": [
        {"name": "obj", "goal": "min"},
    ]
}

def known_constraints(param):
    return param['x_0'] + param['x_1'] < 1.2


gryffin = Gryffin(config_dict=config, known_constraints=known_constraints)
sampling_strategies = [1, -1]


observations = []
MAX_ITER = 100

for num_iter in range(MAX_ITER):
    print('-'*20, 'Iteration:', num_iter+1, '-'*20)

    # Select alternating sampling strategy (i.e. lambda value presented in the Phoenics paper)
    select_ix = num_iter % len(sampling_strategies)
    sampling_strategy = sampling_strategies[select_ix]

    # Query for new parameters
    params  = gryffin.recommend(
        observations = observations, 
        sampling_strategies=[sampling_strategy]
    )

    param = params[0]
    print('  Proposed Parameters:', param, end=' ')

    # Evaluate the proposed parameters.
    observation = compute_objective(param)
    print('==> Merit:', observation['obj'])

    # Append this observation to the previous experiments
    observations.append(param)