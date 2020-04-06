#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import sys
import numpy as np 

from gryffin.utilities import GryffinModuleError

try:
	import sobol_seq
except ModuleNotFoundError:
	_, error_message, _ = sys.exc_info()
	extension = '\n\tTry installing the sobol_seq package or use uniform sampling instead.'
	GryffinModuleError(str(error_message) + extension)

#========================================================================

class SobolCategorical(object):

	def __init__(self):
		pass

#========================================================================


class SobolContinuous(object):

	def __init__(self, seed = None):
		if seed is None:
			seed = np.random.randint(low = 0, high = 10**5)
		self.seed = seed


	def draw(self, low, high, size):
		num, dim = size[0], size[1]
		samples = []
		for _ in range(num):
			vector, seed = sobol_seq.i4_sobol(dim, self.seed)
			sample = (high - low) * vector + low
			self.seed = seed
			samples.append(sample)
		return np.array(samples)


#========================================================================


