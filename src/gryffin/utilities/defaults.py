#!/usr/bin/env python

__author__ = 'Florian Hase'

#=======================================================================
# DEFAULT GENERAL CONFIGURATIONS

default_general_configurations = {
	'auto_desc_gen':          False,
	'backend':               'tensorflow',
	'batches':                1,
	'boosted':                True,
	'parallel':               True,
	'random_seed':            100691,
	'sampler':               'uniform',
	'sampling_strategies':    2,
	'scratch_dir':           './.scratch',
	'softness':               0.001,
	'feas_sensitivity':       1,
	'continuous_optimizer':  'adam',
	'categorical_optimizer': 'naive',
	'discrete_optimizer':    'naive',
	'verbosity': {
			'default':          2,
			'bayesian_network': 3,
			'random_sampler':   2,
		}
}

#=======================================================================
# DEFAULT DATABASE CONFIGURATIONS

default_database_configurations = {
	'format':           'sqlite',
	'path':             './SearchProgress',
	'log_observations':  True,
	'log_runtimes':      True,
}

#=======================================================================

default_configuration = {
	'general': {
			key: default_general_configurations[key] for key in default_general_configurations.keys()
		},
    'database': {
			key: default_database_configurations[key] for key in default_database_configurations.keys()
		},
	'parameters': [
			{'name': 'param_0', 'type': 'continuous', 'low':  0.0, 'high': 10.0, 'size': 2},
			{'name': 'param_1', 'type': 'continuous', 'low': -1.0, 'high':  1.0, 'size': 1},
		],
	'objectives': [ 
			{'name': 'obj_0', 'goal': 'minimize', 'hierarchy': 0, 'tolerance': 0.2},
			{'name': 'obj_1', 'goal': 'maximize', 'hierarchy': 1, 'tolerance': 0.2},
		]
}

#=======================================================================


