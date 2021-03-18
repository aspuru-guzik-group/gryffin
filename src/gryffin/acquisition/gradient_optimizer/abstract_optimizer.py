#!/usr/bin/env python 
  
__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

#========================================================================

class AbstractOptimizer(object):

	dx = 1e-6

	def __init__(self, func, *args, **kwargs):
		self.func = func
		for key, value in kwargs.items():
			setattr(self, str(key), value)


	def _set_func(self, func, pos = None):
		self.func = func
		if pos is not None:
			self.pos     = pos
			self.num_pos = len(pos)


	def grad(self, sample, step = None):
		if step is None: step = self.dx
		gradients = np.zeros(len(sample), dtype = np.float32)
		perturb   = np.zeros(len(sample), dtype = np.float32)
		for pos_index, pos in enumerate(self.pos):
#			print('___________________', pos, sample)
			if pos is None: continue
			perturb[pos] += step
			gradient = (self.func(sample + perturb) - self.func(sample - perturb)) / (2. * step)
#			print('___________________', self.func(sample), self.func(sample + perturb), self.func(sample - perturb))
#			print('___________________', pos, gradient, step, perturb)
			gradients[pos] = gradient
			perturb[pos] -= step
		return gradients
	

