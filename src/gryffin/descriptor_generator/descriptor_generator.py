#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os
import copy
import time
import uuid
import pickle
import subprocess

import numpy as np
import multiprocessing

from gryffin.utilities import Logger
from gryffin.utilities.decorators import thread


class DescriptorGenerator(Logger):

	eta       = 1e-3
	max_iter  = 10**3
	def __init__(self, config):

		self.config        = config
		self.is_generating = False
		self.exec_name = '%s/descriptor_generator/generation_process.py' % self.config.get('home')

		# define registers
		self.auto_gen_descs     = {}
		self.comp_corr_coeffs   = {}
		self.gen_descs_cov      = {}
		self.min_corrs          = {}	
		self.reduced_gen_descs  = {}
		self.weights            = {}
		self.sufficient_indices = {}

		if self.config.get('num_cpus') == 'all':
			self.num_cpus = multiprocessing.cpu_count()
		else:
			self.num_cpus = int(self.config.get('num_cpus'))

	@thread
	def single_generate(self, descs, objs, feature_index, result_dict=None):

		# collect all relevant properties
		sim_dict = {}
		for prop in dir(self):
			if callable(getattr(self, prop)) or prop.startswith(('__', 'W', 'config')):
				continue
			sim_dict[prop] = getattr(self, prop)

		sim_dict['num_samples'] = descs.shape[0]
		sim_dict['num_descs']   = descs.shape[1]

		sim_dict['descs']       = descs
		sim_dict['objs']        = objs
		sim_dict['grid_descs']  = self.config.feature_descriptors[feature_index]
	
		identifier = str(uuid.uuid4())[:8]
		config_name = '%s/descriptor_generation_%d_%s.pkl' % (self.config.get('scratch_dir'), feature_index, identifier)
		with open(config_name, 'wb') as content:
			pickle.dump(sim_dict, content)

		subprocess.call('python %s %s' % (self.exec_name, config_name), shell=True)
		print('SUBMITTED DESC GENERATION')		
		results_name = '%s/completed_descriptor_generation_%d_%s.pkl' % (self.config.get('scratch_dir'), feature_index, identifier)

		# wait for results to be written
		while not os.path.isfile(results_name):
			time.sleep(0.05)
		current_size = 0
		while current_size != os.path.getsize(results_name):
			current_size = os.path.getsize(results_name)
			time.sleep(0.05)

		time.sleep(0.2)
		try:		
			with open(results_name, 'rb') as content:
				results = pickle.load(content)
		except EOFError:
			time.sleep(2)
			with open(results_name, 'rb') as content:
				results = pickle.load(content)
		
		self.min_corrs[feature_index]          = results['min_corrs']
		self.auto_gen_descs[feature_index]     = results['auto_gen_descs']
		self.comp_corr_coeffs[feature_index]   = results['comp_corr_coeffs']
		self.gen_descs_cov[feature_index]      = results['gen_descs_cov']
		self.reduced_gen_descs[feature_index]  = results['reduced_gen_descs']
		self.weights[feature_index]            = results['weights']	
		self.sufficient_indices[feature_index] = results['sufficient_indices']

		result_dict[feature_index] = results['reduced_gen_descs']

		os.remove(config_name)
		os.remove(results_name)

	@thread
	def generate(self, obs_params, obs_objs):

		import time 
		start = time.time()

		self.is_generating  = True
		result_dict         = {}
		feature_types       = self.config.feature_types
		feature_descriptors = self.config.feature_descriptors

		for feature_index, feature_options in enumerate(self.config.feature_options):

			if feature_types[feature_index] == 'continuous': 
				self.weights[feature_index]           = None
				self.reduced_gen_descs[feature_index] = None
				result_dict[feature_index]            = None
				continue

			if feature_descriptors[feature_index] is None:
				self.weights[feature_index]           = None
				self.reduced_gen_descs[feature_index] = None
				result_dict[feature_index]            = None
				continue

			if feature_descriptors[feature_index].shape[1] == 1:
				self.weights[feature_index]           = np.array([[1.]])
				self.reduced_gen_descs[feature_index] = feature_descriptors[feature_index]
				result_dict[feature_index]            = feature_descriptors[feature_index]
				continue

			sampled_params      = obs_params[:, feature_index].astype(np.int32)
			sampled_descriptors = feature_descriptors[feature_index][sampled_params]
			sampled_objs        = np.reshape(obs_objs, (len(obs_objs), 1)) 

			self.single_generate(sampled_descriptors, sampled_objs, feature_index, result_dict)

			# avoid parallel execution if not desired
			if self.num_cpus == 1:
				if feature_types[feature_index] == 'continuous':
					continue
				while not feature_index in result_dict:
					time.sleep(0.1)

		for feature_index in range(len(self.config.feature_options)):
			if feature_types[feature_index] == 'continuous':
				continue
			while not feature_index in result_dict:
				time.sleep(0.1)

		gen_feature_descriptors = [result_dict[feature_index] for feature_index in range(len(result_dict.keys()))]
		self.gen_feature_descriptors = gen_feature_descriptors
		self.is_generating = False

		end = time.time()
		self.desc_gen_time = end - start

	def get_descriptors(self):
		while self.is_generating:
			time.sleep(0.1)

		if hasattr(self, 'gen_feature_descriptors'):
			print('[TIME:  ', self.desc_gen_time, '  (descriptor generation)')
			return self.gen_feature_descriptors
		else:
			return self.config.feature_descriptors

	def get_summary(self):
		
		summary = {}
		feature_types = self.config.feature_types
		# If we have not generated new descriptors
		if not hasattr(self, 'gen_feature_descriptors'):
			for feature_index in range(len(self.config.feature_options)):
				contribs = {}
				if feature_types[feature_index] == 'continuous':
					continue
				feature_descriptors = self.config.feature_descriptors[feature_index]
				if feature_descriptors is None:
					continue
				for desc_index in range(feature_descriptors.shape[1]):
					desc_summary_dict = {}
					desc_summary_dict['relevant_given_descriptors']     = np.arange(len(feature_descriptors[:, desc_index]))
					desc_summary_dict['given_descriptor_contributions'] = np.ones(len(feature_descriptors[:, desc_index]))
					contribs['descriptor_%d' % desc_index] = copy.deepcopy(desc_summary_dict)
				summary['feature_%d' % feature_index] = copy.deepcopy(contribs)
			return summary

		# If we have generated new descriptors
		for feature_index in range(len(self.config.feature_options)):
	
			if feature_types[feature_index] == 'continuous':
				continue
	
			weights            = self.weights[feature_index]
			sufficient_indices = self.sufficient_indices[feature_index]
			print('sufficient_indices', sufficient_indices)

			if weights is None:
				continue
			if len(sufficient_indices) == 0:
				continue

			# normalize weights
			normed_weights = np.empty(weights.shape)
			for index, weight_elements in enumerate(weights):
				normed_weights[index] = weight_elements / np.sum(np.abs(weight_elements))

			# identify contributing indices
			contribs = {}
			for new_desc_index in sufficient_indices:
				desc_summary_dict = {}
				relevant_weights  = normed_weights[new_desc_index]

				sorting_indices = np.argsort(np.abs(relevant_weights))
				cumulative_sum  = np.cumsum(np.abs(relevant_weights[sorting_indices]))
				include_indices = np.where(cumulative_sum > 0.1)[0]

				relevant_given_descriptors = sorting_indices[include_indices]
				desc_summary_dict['relevant_given_descriptors']     = relevant_given_descriptors
				desc_summary_dict['given_descriptor_contributions'] = weights[new_desc_index]
				contribs['descriptor_%d' % new_desc_index] = copy.deepcopy(desc_summary_dict)
			summary['feature_%d' % feature_index] = copy.deepcopy(contribs)

		return summary


