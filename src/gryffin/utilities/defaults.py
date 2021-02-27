#!/usr/bin/env python

import json

__author__ = 'Florian Hase'

# =============================
# Default general configuration
# =============================
default_general_configurations = {
    'num_cpus':               1,  # Options are a number, or 'all'
    'boosted':                True,
    'auto_desc_gen':          False,
    'batches':                1,
    'sampling_strategies':    2,
    'softness':               0.001,  # softness of Chimera for multiobj optimizations
    'feas_sensitivity':       1,  # sensitivity to feasibility constraints
    'random_seed':            100691,
    'sampler':               'uniform',
    'save_database':          True,
    'scratch_dir':           './.scratch',
    'continuous_optimizer':  'adam',
    'categorical_optimizer': 'naive',
    'discrete_optimizer':    'naive',
    'verbosity': {
        'default':          2,
        'bayesian_network': 3,
        'random_sampler':   2,
    }
}


# ==============================
# Default database configuration
# ==============================
default_database_configurations = {
    'format':           'sqlite',
    'path':             './SearchProgress',
    'log_observations':  True,
    'log_runtimes':      True,
}

# =============================
# Default overall configuration
# =============================
default_configuration = {
    'general': {
        key: default_general_configurations[key] for key in default_general_configurations.keys()
    },
    'database': {
        key: default_database_configurations[key] for key in default_database_configurations.keys()
    },
    'parameters': [
        {'name': 'param_0', 'type': 'continuous', 'low': 0, 'high': 1, 'size': 1},
        {'name': 'param_1', 'type': 'continuous', 'low': 0, 'high': 1, 'size': 1}
        # {'name': 'param_1', 'type': 'categorical', 'category_details': {'A':[1,2], 'B'[2,1], ..., 'Z':[4,5]},
    ],
    'objectives': [
        {'name': 'obj_0', 'goal': 'min', 'tolerance': 0.2, 'absolute': False},
        {'name': 'obj_1', 'goal': 'max', 'tolerance': 0.2, 'absolute': False},
    ]
}


def get_config_defaults(json_file=None):
    """Returns the default configurations for Gryffin.

    Parameters
    ----------
    json_file: str
        Whether to write the default configurations to a json file with this name. Default is None, i.e.
        do not save json file.

    Returns
    -------
    config : dict
        The default configurations for Gryffin, either as dict or json string.
    """
    if json_file is not None:
        with open(json_file, 'w') as f:
            json.dump(default_configuration, f, indent=4)

    return default_configuration
