import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
from gryffin import Gryffin

def objective(x):
    
    def sigmoid(x, l, k, x0):
        return l / (1 + np.exp(-k*(x-x0)))

    sigs = [sigmoid(x, -1, 40, 0.2),
            sigmoid(x,  1, 40, 0.4),
            sigmoid(x,  -0.7, 50, 0.6),
            sigmoid(x, 0.7, 50, 0.9)
           ]

    return np.sum(sigs, axis=0) + 1

def compute_objective(param):
    x = param['x']
    param['obj'] = objective(x)
    return param


config = {
    "general": {
        "random_seed": 42,
    },
    "parameters": [
        {"name": "x", "type": "continuous", "low": 0., "high": 1., "size": 1}
    ],
    "objectives": [
        {"name": "obj", "goal": "min"}
    ]
}

gryffin = Gryffin(config_dict=config, silent=True)

observations = []
MAX_ITER = 15

for num_iter in range(MAX_ITER):
    print('-'*20, 'Iteration:', num_iter+1, '-'*20)
    
    # Query for new parameters
    params = gryffin.recommend(observations=observations)
    
    # Params is a list of dict, where each dict containts the proposed parameter values, e.g., {'x':0.5}
    # in this example, len(params) == 1 and we select the single set of parameters proposed
    param = params[0]
    print('  Proposed Parameters:', param, end=' ')
        
    # Evaluate the proposed parameters. "compute_objective" takes param, which is a dict, and adds the key "obj" with the
    # objective function value
    observation = compute_objective(param)
    print('==> :', observation['obj'])
    
    # Append this observation to the previous experiments
    observations.append(observation)