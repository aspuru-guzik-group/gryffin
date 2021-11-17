#!/usr/bin/env python

import numpy as np
from gryffin import Gryffin

from gryffin.benchmark_functions import CatCamel, CatAckley


PARAM_DIM = 2
NUM_OPTS = 21
BUDGET = 24
SAMPLING_STRATEGIES = np.array([-1, 1])
TYPE = 'naive' # 'naive', 'static', 'dynamic'
auto_desc_gen = False

surface = CatCamel(num_dims=PARAM_DIM, num_opts=NUM_OPTS)

if TYPE == 'naive':
	param_0_details = {f'x_{i}': None for i in range(NUM_OPTS)}
	param_1_details  = {f'x_{i}': None for i in range(NUM_OPTS)}
elif TYPE in ['static', 'dynamic']:
	param_0_details = {f'x_{i}': [i] for i in range(NUM_OPTS)}
	param_1_details  = {f'x_{i}': [i] for i in range(NUM_OPTS)}
	if TYPE == 'dynamic':
		auto_desc_gen = True
else:
	raise NotImplementedError


# define Gryffin config
config = {
	"general": {
		"num_cpus": 1,
		"auto_desc_gen": auto_desc_gen,
		"batches": 1,
		"sampling_strategies": 1,
		"boosted":  False,
		"caching": True,
		"random_seed": 2021,
		"acquisition_optimizer": "genetic",
		"verbosity": 4
		},
	"parameters": [
		{"name": "param_0", "type": "categorical", "category_details": param_0_details},
		{"name": "param_1", "type": "categorical", "category_details": param_1_details},
	],
	"objectives": [
		{"name": "obj", "goal": "min"},
	]
}

gryffin = Gryffin(config_dict=config)
observations = []
for iter_ in range(BUDGET):
	print(f'\nITER : {iter_+1}')
	select_ix = iter_ % len(SAMPLING_STRATEGIES)
	sampling_strategy = SAMPLING_STRATEGIES[select_ix]

	samples = gryffin.recommend(observations, sampling_strategies=[sampling_strategy])
	sample = samples[0]
	observation = surface([sample['param_0'], sample['param_1']])
	sample['obj'] = observation
	observations.append(sample)

	print(f'SAMPLE : {sample}')
