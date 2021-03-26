#!/usr/bin/env python

import json

__author__ = 'Florian Hase'

# =============================
# Default general configuration
# =============================
default_general_configuration = {
    'num_cpus':               1,  # Options are a number, or 'all'
    'boosted':                True,
    'caching':                True,
    'auto_desc_gen':          False,
    'batches':                1,
    'sampling_strategies':    2,
    'softness':               0.001,  # softness of Chimera for multiobj optimizations
    'feas_approach':          'fwa',  # "fwa"="feasibility-weighted acquisition" OR "fai"="feasibility-acquisition interpolation"
    'feas_sensitivity':       1,  # sensitivity to feasibility constraints
    'random_seed':            100691,
    'save_database':          False,
    'acquisition_optimizer':  'adam',  # options are "adam" or "genetic"
    'verbosity': {
        'default':          2,
        'bayesian_network': 3,
        'random_sampler':   2,
    }
}


# ==============================
# Default database configuration
# ==============================
default_database_configuration = {
    'format':           'sqlite',
    'path':             './SearchProgress',
    'log_observations':  True,
    'log_runtimes':      True,
}

# =========================
# Default BNN configuration
# =========================
default_regression_model_configuration = {
    'num_epochs':  2 * 10**3,
    'learning_rate': 0.05,
    'num_draws': 10**3,
    'num_layers': 3,
    'hidden_shape': 6,
    'weight_loc': 0.,
    'weight_scale': 1.,
    'bias_loc': 0.,
    'bias_scale': 1.
}

default_classification_model_configuration = {
    'num_epochs':  2 * 10**3,
    'learning_rate': 0.05,
    'num_draws': 10**3,
    'num_layers': 3,
    'hidden_shape': 6,
    'weight_loc': 0.,
    'weight_scale': 1.,
    'bias_loc': 0.,
    'bias_scale': 1.
}

# =============================
# Default overall configuration
# =============================
default_configuration = {
    'general': {
        key: default_general_configuration[key] for key in default_general_configuration.keys()
    },
    'database': {
        key: default_database_configuration[key] for key in default_database_configuration.keys()
    },
    'regression_model': {
        key: default_regression_model_configuration[key] for key in default_regression_model_configuration.keys()
    },
    'classification_model': {
        key: default_classification_model_configuration[key] for key in default_classification_model_configuration.keys()
    },
    'parameters': [
        {'name': 'param_0', 'type': 'continuous', 'low': 0, 'high': 1},
        {'name': 'param_1', 'type': 'continuous', 'low': 0, 'high': 1}
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
