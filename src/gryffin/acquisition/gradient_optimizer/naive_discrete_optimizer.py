#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

from . import AbstractOptimizer

#========================================================================

class NaiveDiscreteOptimizer(AbstractOptimizer):

	def __init__(self, func = None, *args, **kwargs):
		AbstractOptimizer.__init__(self, func, *args, **kwargs)


	def set_func(self, func, pos = None, highest = None):
		self.highest = highest
		self._set_func(func, pos)


	def get_update(self, vector):
		func_best = self.func(vector)
		for pos_index, pos in enumerate(self.pos):
			if pos is None: continue

			current = vector[pos]
			perturb = np.random.choice(self.highest[pos_index])
			vector[pos] = perturb

			func_cand = self.func(vector)
			if func_cand < func_best:
				func_best = func_cand
			else:
				vector[pos] = current
		return vector



	def old_get_update(self, vector):
		func_best = self.func(vector)
		for pos_index in range(self.num_pos):
			if self.pos[pos_index] is None: continue

			current = vector[pos_index]
			perturb = np.random.choice(self.highest[pos_index])
			vector[pos_index] = perturb

			func_cand = self.func(vector)
			if func_cand < func_best:
				func_best = func_cand
			else:
				vector[pos_index] = current
		return vector

#========================================================================


