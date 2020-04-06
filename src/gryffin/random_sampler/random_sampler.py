#!/usr/bin/env python 
  
__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

from gryffin.utilities import Logger
from gryffin.utilities import GryffinUnknownSettingsError

#========================================================================

class RandomSampler(Logger):

	def __init__(self, config_general, config_params):
		self.config_general = config_general
		self.config_params  = config_params
		verbosity           = self.config_general.verbosity
		if 'random_sampler' in self.config_general.verbosity:
			verbosity = self.config_general.verbosity['random_sampler']
		Logger.__init__(self, 'RandomSampler', verbosity)

		if self.config_general.sampler == 'sobol':
			from .sobol   import SobolContinuous
			from .uniform import UniformCategorical, UniformDiscrete
			self.continuous_sampler  = SobolContinuous()
			self.categorical_sampler = UniformCategorical()
			self.discrete_sampler    = UniformDiscrete()
		elif self.config_general.sampler == 'uniform':
			from .uniform import UniformCategorical, UniformContinuous, UniformDiscrete
			self.continuous_sampler  = UniformContinuous()
			self.categorical_sampler = UniformCategorical()
			self.discrete_sampler    = UniformDiscrete()
		else:
			GryffinUnknownSettingsError('did not understanding sampler setting: "%s".\n\tChoose from "uniform" or "sobol"' % self.config_general.sampler)


	def draw(self, num = 1):
		samples = []
		for param_index, param_settings in enumerate(self.config_params):
			specs = param_settings['specifics']
			if param_settings['type'] == 'continuous':
				sampled_values = self.continuous_sampler.draw(specs['low'], specs['high'], (num, param_settings['size']))
				samples.append(sampled_values)
			elif param_settings['type'] == 'categorical':
				sampled_values = self.categorical_sampler.draw(len(specs['options']), (num, param_settings['size']))
				samples.append(sampled_values)
			elif param_settings['type'] == 'discrete':
				sampled_values = self.discrete_sampler.draw(specs['low'], specs['high'], (num, param_settings['size']))
				samples.append(sampled_values)
			else:
				GryffinUnknownSettingsError('did not understand parameter type: "%s" of parameter "%s".\n\t(%s) Choose from "continuous", "discrete" or "categorical"' % (param_settings['type'], self.config_params[param_index]['name'], self.template))
		samples = np.concatenate(samples, axis = 1)
		self.log('generated uniform samples: \n%s' % str(samples), 'DEBUG')
		return samples


	def perturb(self, pos, num = 1, scale = 0.05):
		samples = []
		for param_index, param_settings in enumerate(self.config_params):
			specs = param_settings['specifics']
			if param_settings['type'] == 'continuous':
				sampled_values  = self.continuous_sampler.draw(-scale, scale, (num, param_settings['size']))
				sampled_values *= specs['high'] - specs['low']
				close_samples   = pos[param_index] + sampled_values
				close_samples   = np.where(close_samples < specs['low'],  specs['low'],  close_samples)
				close_samples   = np.where(close_samples > specs['high'], specs['high'], close_samples)
				samples.append(close_samples)
			elif param_settings['type'] in ['categorical', 'discrete']:
				sampled_values = pos[param_index] * np.ones((num, param_settings['size'])).astype(np.float32)
				samples.append(sampled_values)
			else:
				GryffinUnknownSettingsError('did not understand settings')
		samples = np.concatenate(samples, axis = 1)
		return samples


	def normal_samples(self, loc = 0., scale = 1., num = 1):
		samples = []
		for param_index, param_settings in enumerate(self.config_params):
			specs = param_settings['specifics']
			if param_settings['type'] == 'continuous':
				param_range = specs['high'] - specs['low']
				sampled_values = np.random.normal(0., scale * param_range, (num, param_settings['size'])) + loc[param_index]
				samples.append(sampled_values)
			elif param_settings['type'] == 'categorical':
				sampled_values = self.categorical_sampler.draw(len(specs['options']), (num, param_settings['size']))
				samples.append(sampled_values)
			elif param_settings['type'] == 'discrete':
				sampled_values = self.discrete_sampler.draw(specs['low'], specs['high'], (num, param_settings['size']))
				samples.append(sampled_values)
			else:
				GryffinUnknownSettingsError('did not understand variable type: "%s" of parameter "%s".\n\t(%s) Choose from "continuous", "discrete" or "categorical"' % (param_settings['type'], param_settings['name'], self.template))
		samples = np.concatenate(samples, axis = 1)
		return samples				

