#!/usr/bin/env python

import numpy as np


#====================================================================


#====================================================================

class Transformation:

	def __init__(self, kind='identity'):
		''' naming convention of transformation methods - transform_kind()'''
		self.kind = kind


	def transform_simplex(self, cubes):
		'''
		converts and n-cube (used for optimization) to an n+1 simplex (used
		as features for Gemini)
		'''
		features = []
		for cube in cubes:
			cube = (1 - 2 * 1e-6) * np.squeeze(np.array([c for c in cube])) + 1e-6
			simpl = np.zeros(len(cube)+1)
			sums = np.sum(cube / (1 - cube))

			alpha = 4.0
			simpl[-1] = alpha / (alpha + sums)
			for _ in range(len(simpl)-1):
				simpl[_] = (cube[_] / (1 - cube[_])) / (alpha + sums)
			features.append(np.array(simpl))
		return np.array(features)

	def transform_identity(self, cubes):
		'''
		perform identity transformation on a n-cube (used for optimization and prediction)
		'''
		return cubes


	def __call__(self, cubes):
		'''
		general purpose method for transforming n-dimensional cube used for optimization
		with Phoenics to geometric object used as features for predictive model
		'''
		transform_fn = getattr(self, f'transform_{self.kind}')
		return transform_fn(cubes)





if __name__ == '__main__':
	# test n=5
	cubes = [[0.9, 0.05, 0.2, 0.3, 0.5], [0.24, 0.05, 0.2, 0.2, 0.5]]

	trans = Transformation('simplex')

	features = trans(cubes)
	print(features, [np.sum(f) for f in features], [f.shape[0] for f in features])
