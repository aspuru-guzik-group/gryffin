#!/usr/bin/env python 
  
# cython: profile=True

__author__ = 'Florian Hase'

#========================================================================

import  cython 
cimport cython

from cython.parallel import prange

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp, abs, round

#========================================================================

cdef double _gauss(double x, double loc, double sqrt_prec):
	cdef double argument, result
	argument = 0.5 * ( sqrt_prec * (x - loc) )**2
	if argument > 200.:
		result = 0.
	else:
		result   = exp( - argument) * sqrt_prec * 0.3989422804014327	# the number is 1. / np.sqrt(2 * np.pi)
	return result

#========================================================================

cdef class KernelEvaluator:

	cdef int    num_samples, num_obs, num_kernels, num_cats
	cdef double lower_prob_bound, inv_vol
	
	cdef np.ndarray np_locs, np_sqrt_precs, np_cat_probs
	cdef np.ndarray np_kernel_types, np_kernel_sizes
	cdef np.ndarray np_objs
	cdef np.ndarray np_probs

	var_dict = {}

	def __init__(self, locs, sqrt_precs, cat_probs, kernel_types, kernel_sizes, lower_prob_bound, objs, inv_vol):
		
		self.np_locs          = locs
		self.np_sqrt_precs    = sqrt_precs
		self.np_cat_probs     = cat_probs
		self.np_kernel_types  = kernel_types
		self.np_kernel_sizes  = kernel_sizes
		self.np_objs          = objs

		self.num_samples      = locs.shape[0]
		self.num_obs          = locs.shape[1]
		self.num_kernels      = locs.shape[2]
		self.lower_prob_bound = lower_prob_bound
		self.inv_vol          = inv_vol

		self.np_probs = np.zeros(self.num_obs, dtype = np.float64)


	@cython.boundscheck(False)
	@cython.cdivision(True)
	cdef double [:] _probs(self, double [:] sample):

		cdef int    sample_index, obs_index, feature_index, kernel_index
		cdef int    num_indices
		cdef int    num_continuous, num_categorical
		cdef double total_prob, prec_prod, exp_arg_sum

		cdef double [:, :, :] locs       = self.np_locs
		cdef double [:, :, :] sqrt_precs = self.np_sqrt_precs 
		cdef double [:, :, :] cat_probs  = self.np_cat_probs

		cdef int [:] kernel_types = self.np_kernel_types
		cdef int [:] kernel_sizes = self.np_kernel_sizes

		cdef double inv_sqrt_two_pi = 0.3989422804014327

		cdef double [:] probs = self.np_probs
		for obs_index in range(self.num_obs):
			probs[obs_index] = 0.

		cdef double cat_prob
		cdef double obs_probs

		# get number of continuous variables
		num_continuous = 0
		while kernel_index < self.num_kernels:
			num_continuous += 1
			kernel_index   += kernel_sizes[kernel_index]


#		print(self.num_obs, self.num_samples, self.num_kernels)

		for obs_index in range(self.num_obs):
			obs_probs = 0.

			for sample_index in range(self.num_samples):
				total_prob     = 1.
				prec_prod      = 1.
				exp_arg_sum    = 0.
				feature_index, kernel_index = 0, 0

				while kernel_index < self.num_kernels:

					if kernel_types[kernel_index] == 0:

						prec_prod      = prec_prod * sqrt_precs[sample_index, obs_index, kernel_index]
						exp_arg_sum    = exp_arg_sum + (sqrt_precs[sample_index, obs_index, kernel_index] * (sample[feature_index] - locs[sample_index, obs_index, kernel_index]))**2

					elif kernel_types[kernel_index] == 1:
						total_prob *= cat_probs[sample_index, obs_index, kernel_index + <int>round(sample[feature_index])]

					kernel_index  += kernel_sizes[kernel_index]
					feature_index += 1

				obs_probs += total_prob * prec_prod * exp( - 0.5 * exp_arg_sum) #* inv_sqrt_two_pi**num_continuous

				if sample_index == 100:
					if 0.01 * obs_probs * inv_sqrt_two_pi**num_continuous < self.lower_prob_bound:
						probs[obs_index] = 0.01 * obs_probs
						break

			else:
				probs[obs_index] = obs_probs * inv_sqrt_two_pi**num_continuous / self.num_samples
		return probs





#	@cython.boundscheck(False)	
#	@cython.cdivision(True)
	cdef double [:] _OLD_probs(self, double [:] sample):

		cdef int sample_index, obs_index, feature_index, kernel_index
		cdef int num_indices
		cdef double total_prob

		cdef double [:, :, :] locs       = self.np_locs
		cdef double [:, :, :] sqrt_precs = self.np_sqrt_precs 
		cdef double [:, :, :] cat_probs  = self.np_cat_probs
	
		cdef int [:] kernel_types     = self.np_kernel_types
		cdef int [:] kernel_sizes     = self.np_kernel_sizes 

		cdef double [:] probs = self.np_probs
		for obs_index in range(self.num_obs):
			probs[obs_index] = 0.

		cdef double cat_prob
		cdef double obs_probs
	
#		for sample_index in range(self.num_samples):
		for obs_index in range(self.num_obs):		
			
			obs_probs = 0.

#			for obs_index in range(self.num_obs):
			for sample_index in range(self.num_samples):
	
				total_prob    = 1.
				feature_index = 0
				kernel_index  = 0

				while kernel_index < self.num_kernels:
#				for feature_index in range(num_indices):	

					if kernel_types[kernel_index] == 0:
						total_prob *= _gauss(sample[feature_index], locs[sample_index, obs_index, kernel_index], sqrt_precs[sample_index, obs_index, kernel_index])

					elif kernel_types[kernel_index] == 1:

						cat_prob = cat_probs[sample_index, obs_index, kernel_index + <int>round(sample[feature_index])]
#						if cat_prob < lower_prob_bounds[feature_index]:
#						if cat_prob < 1e-2:
#							break

						total_prob *= cat_prob

					kernel_index  += kernel_sizes[kernel_index]
					feature_index += 1

#				else:
				obs_probs += total_prob

				if sample_index == 100:
					if 0.01 * obs_probs < self.lower_prob_bound:
						probs[obs_index] = 0.01 * obs_probs
						break

#			probs[obs_index] = obs_probs / self.num_samples
			else:
				probs[obs_index] = obs_probs / self.num_samples
#			print(np.asarray(probs), np.asarray(sample), np.asarray(kernel_types))
		return probs


#	@cython.boundscheck(False)
	cpdef get_kernel(self, np.ndarray sample):

		cdef int obs_index
		cdef double temp_0, temp_1
		cdef double inv_den
		
		cdef double [:] sample_memview = sample
		probs_sample = self._probs(sample_memview)

		# construct numerator and denominator of acquisition
		cdef double num = 0.
		cdef double den = 0.
		cdef double [:] objs = self.np_objs 

		for obs_index in range(self.num_obs):
			temp_0 = objs[obs_index]
			temp_1 = probs_sample[obs_index]
			num += temp_0 * temp_1
			den += temp_1

		inv_den = 1. / (self.inv_vol + den)

#		print('-->', num, den)
#		print('OBJETIVES', self.np_objs)

		return num, inv_den, probs_sample
	





