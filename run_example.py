#!/usr/bin/env python

#==========================================================================

import numpy as np

from gryffin import Gryffin

# choose the synthetic function from
# ... Dejong:
# ...

import olympus
from olympus.surfaces import Surface

#==========================================================================
# `global` variables

BUDGET = 24
PLOT = True
SAMPLING_SRATEGIES = np.array([-1, 1])
surface = Surface(kind='Dejong', param_dim=2)

#==========================================================================

# gryffin config
config = {
    "general": {
        "num_cpus": 4,
        "auto_desc_gen": False,
        "batches": 1,
        "sampling_strategies": 1,
        "boosted":  False,
        "caching": True,
        "random_seed": 2022,
        "acquisition_optimizer": "adam", # "adam" or "genetic"
        "verbosity": 3
        },
    "parameters": [
        {"name": "param_0", "type": "continuous", "low": 0.0, "high": 1.0},
        {"name": "param_1", "type": "continuous", "low": 0.0, "high": 1.0},
    ],
    "objectives": [
		{"name": "obj", "goal": "min"},
    ]
}


# initialize gryffin
gryffin = Gryffin(config_dict=config,)

#==========================================================================
# plotting instructions (optional)

if PLOT:
	import matplotlib.pyplot as plt
	import seaborn as sns
	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes = axes.flatten()
	plt.ion()

#==========================================================================

observations = []
for iter in range(BUDGET):


	# alternating sampling strategies
	select_ix = iter % len(SAMPLING_SRATEGIES)
	sampling_strategy = SAMPLING_SRATEGIES[select_ix]

	# get a new sample
	samples  = gryffin.recommend(
		observations = observations, sampling_strategies=[sampling_strategy]
	)

	sample = samples[0]

	# get measurements for samples
	observation = surface.run([val for key, val in sample.items()])[0][0]
	print(observation)
	sample['obj'] = observation

	print(sample)
	print(observations)

	if PLOT:
		# optional instructions just for plotting
		for ax in axes:
			ax.clear()

		# plotting ground truth
		x_domain = np.linspace(0., 1., 50)
		y_domain = np.linspace(0., 1., 50)
		X, Y = np.meshgrid(x_domain, y_domain)
		Z    = np.zeros((len(x_domain), len(y_domain)))
		for x_index, x_element in enumerate(x_domain):
			for y_index, y_element in enumerate(y_domain):
				loss_value = surface.run([x_element, y_element])[0][0]
				Z[y_index, x_index] = loss_value

		contours = axes[0].contour(X, Y, Z, 3, colors='black')
		axes[0].clabel(contours, inline=True, fontsize=8)
		axes[0].imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)



		for obs_index, obs in enumerate(observations):
			axes[0].plot(obs['param_0'], obs['param_1'], marker = 'o', color = '#1a1423', markersize = 7, alpha=0.8)

		if len(observations) >= 1:
			# plot the final observation
			axes[0].plot(observations[-1]['param_0'], observations[-1]['param_1'], marker = 'D', color = '#5b2333', markersize = 8)

		axes[0].set_ylim(0., 1.)



		plt.pause(0.05)


	# add measurements to cache
	observations.append(sample)
