#!/usr/bin/env python

#==========================================================================

import numpy as np

from gryffin import Gryffin

# choose the synthetic function from
# ... Dejong:
# ...
from benchmark_functions import Dejong as Benchmark
from category_writer     import CategoryWriter

#==========================================================================
# `global` variables

BUDGET      = 24
CONFIG_FILE = 'config.json'

NUM_DIMS    = 2
NUM_OPTS    = 21

#==========================================================================

# write categories
category_writer = CategoryWriter(num_opts = NUM_OPTS, num_dims  = NUM_DIMS)
category_writer.write_categories(home_dir = './',     num_descs = 2, with_descriptors = False)

# create benchmark function
benchmark = Benchmark(num_dims = NUM_DIMS, num_opts = NUM_OPTS)

# initialize gryffin
gryffin = Gryffin(CONFIG_FILE)

#==========================================================================
# plotting instructions (optional)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper', font_scale = 1.5)

colors = sns.color_palette('RdYlBu', 8)
colors = [colors[-1], colors[0]]

fig = plt.figure(figsize = (6, 6))
ax0 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.subplot2grid((2, 2), (1, 0))
ax2 = plt.subplot2grid((2, 2), (1, 1))
axs = [ax0, ax1, ax2]
plt.ion()

#==========================================================================

observations = []
for _ in range(BUDGET):

	# get a new sample
	samples  = gryffin.recommend(observations = observations)

	# get measurements for samples
	new_observations  = []
	for sample in samples:
		param         = np.array([sample['param_0'][0], sample['param_1'][0]])
		measurement   = benchmark(param)
		sample['obj'] = measurement
		new_observations.append(sample)

	# optional instructions just for plotting
	for ax in axs:
		ax.cla()

	# plotting ground truth
	x_domain = np.linspace(0., 1., NUM_OPTS)
	y_domain = np.linspace(0., 1., NUM_OPTS)
	X, Y = np.meshgrid(x_domain, y_domain)
	Z    = np.zeros((len(x_domain), len(y_domain)))
	for x_index, x_element in enumerate(x_domain):
		for y_index, y_element in enumerate(y_domain):
			loss_value = benchmark(['x_{}'.format(x_index), 'x_{}'.format(y_index)])
			Z[y_index, x_index] = loss_value

	levels = np.linspace(np.amin(Z), np.amax(Z), 256)
	ax0.imshow(Z, plt.cm.bone_r, origin = 'lower', aspect = 'auto')

	# plotting surrogates
	kernel              = gryffin.bayesian_network.kernel_contribution
	sampling_parameters = gryffin.bayesian_network.sampling_param_values

	x_domain = np.linspace(0., NUM_OPTS - 1, NUM_OPTS)
	y_domain = np.linspace(0., NUM_OPTS - 1, NUM_OPTS)
	Z    = np.zeros((len(x_domain), len(y_domain)))
	for x_index, x_element in enumerate(x_domain):
		for y_index, y_element in enumerate(y_domain):
			param = np.array([x_element, y_element])   #, dtype = np.float32)
			num, den = kernel(param)
			loss_value = (num + sampling_parameters[0]) * den
			Z[y_index, x_index] = loss_value

	levels = np.linspace(np.amin(Z), np.amax(Z), 256)
#	ax1.contourf(X, Y, Z, cmap = plt.cm.bone_r, levels = levels)
	ax1.imshow(Z, plt.cm.bone_r, origin = 'lower', aspect = 'auto')

	# plotting surrogates
	Z    = np.zeros((len(x_domain), len(y_domain)))
	for x_index, x_element in enumerate(x_domain):
		for y_index, y_element in enumerate(y_domain):
			param = np.array([x_element, y_element])   #, dtype = np.float32)
			num, den = kernel(param)
			loss_value = (num + sampling_parameters[1]) * den
			Z[y_index, x_index] = loss_value

	levels = np.linspace(np.amin(Z), np.amax(Z), 256)
	ax2.imshow(Z, plt.cm.bone_r, origin = 'lower', aspect = 'auto')

	for obs_index, obs in enumerate(observations):
		ax0.plot(int(obs['param_0'][0][2:]), int(obs['param_1'][0][2:]), marker = 'o', color = colors[obs_index % len(colors)], markersize = 5)
		ax1.plot(int(obs['param_0'][0][2:]), int(obs['param_1'][0][2:]), marker = 'o', color = colors[obs_index % len(colors)], markersize = 5)
		ax2.plot(int(obs['param_0'][0][2:]), int(obs['param_1'][0][2:]), marker = 'o', color = colors[obs_index % len(colors)], markersize = 5)

	for obs_index, obs in enumerate(new_observations):
		ax0.plot(int(obs['param_0'][0][2:]), int(obs['param_1'][0][2:]), marker = 'D', color = colors[obs_index % len(colors)], markersize = 8)
		ax1.plot(int(obs['param_0'][0][2:]), int(obs['param_1'][0][2:]), marker = 'D', color = colors[obs_index % len(colors)], markersize = 8)
		ax2.plot(int(obs['param_0'][0][2:]), int(obs['param_1'][0][2:]), marker = 'D', color = colors[obs_index % len(colors)], markersize = 8)

	plt.pause(0.05)


	# add measurements to cache
	observations.extend(new_observations)
