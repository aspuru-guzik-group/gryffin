#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np

#========================================================================

class UniformCategorical(object):
	
	def __init__(self):
		pass

	def draw(self, num_options, size):
		return np.random.choice(num_options, size = size).astype(np.float32)

#========================================================================


class UniformContinuous(object):
	
	def __init__(self):
		pass

	def draw(self, low, high, size):
		return np.random.uniform(low = low, high = high, size = size).astype(np.float32)

#========================================================================


class UniformDiscrete(object):

	def __init__(self):
		pass

	def draw(self, low, high, size):
		return np.random.randint(low = 0, high = high - low, size = size).astype(np.float32)


