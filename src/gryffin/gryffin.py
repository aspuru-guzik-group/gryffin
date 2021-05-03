#!/usr/bin/env python

__author__ = 'Florian Hase'

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .acquisition           import Acquisition
from .bayesian_network      import BayesianNetwork
from .descriptor_generator  import DescriptorGenerator
from .observation_processor import ObservationProcessor
from .random_sampler        import RandomSampler
from .sample_selector       import SampleSelector
from .predictive_model 		import PredictiveModel, TrainingSetGenerator
from .utilities             import ConfigParser, Logger, GryffinNotFoundError


class Gryffin(Logger):

	def __init__(self, config_file=None, config_dict=None):

		Logger.__init__(self, 'Gryffin', verbosity=0)

		# parse configuration
		self.config = ConfigParser(config_file, config_dict)
		self.config.parse()
		self.config.set_home(os.path.dirname(os.path.abspath(__file__)))

		np.random.seed(self.config.get('random_seed'))
		self.update_verbosity(self.config.get('verbosity'))
		self.create_folders()

		self.random_sampler            = RandomSampler(self.config.general, self.config.parameters)
		self.obs_processor             = ObservationProcessor(self.config)
		self.descriptor_generator      = DescriptorGenerator(self.config)
		self.descriptor_generator_feas = DescriptorGenerator(self.config)
		self.bayesian_network          = BayesianNetwork(self.config)
		self.bayesian_network_feas     = BayesianNetwork(self.config)
		self.acquisition               = Acquisition(self.config)
		self.sample_selector           = SampleSelector(self.config)

		self.predictive_model 		   = PredictiveModel(self.config)
		self.training_set_gen 		   = TrainingSetGenerator(self.config)

		self.iter_counter = 0

	def create_folders(self):

		if not os.path.isdir(self.config.get('scratch_dir')):
			try:
				os.mkdir(self.config.get('scratch_dir'))
			except FileNotFoundError:
				GryffinNotFoundError('Could not create scratch directory: %s' % self.config.get('scratch_dir'))

		if self.config.get('save_database') is True and not os.path.isdir(self.config.get_db('path')):
			try:
				os.mkdir(self.config.get_db('path'))
			except FileNotFoundError:
				GryffinNotFoundError('Could not create database directory: %s' % self.config.get_db('path'))

		if self.config.get('save_database') is True:
			from .database_handler import DatabaseHandler
			self.db_handler = DatabaseHandler(self.config)

	def recommend(self,
				  observations=None,
				  proxy_observations=None,
				  predictive_model=None,
				  as_array=False):
		'''

		Args:
			observations (list) : observations of the main surfaces being optimized
			proxy_measurements (list) : observations of an auxilliary surface to be used
										by a predictive model
			predictive_model (callable): predictive model callable
			as_array (bool):
		'''

		# handle the predictive model - if callable given, overwrite the current preddictive model
		if predictive_model == None:
			if self.config.get_pred('model_kind') == None:
				self.log('No predictive model specified, continuing with regular optimization', 'WARNING')
			elif type(self.config.get_pred('model_kind')) == str:
				self.log(f'Continuing with built in predictive model {self.predictive_model.kind}', 'INFO')
		elif not callable(getattr(predictive_model, 'predict')):
			if type(self.config.get_pred('model_kind')) == str:
				self.log('Specified predictive model object is not callable, resorting to predictive model from config', 'WARNING')
				self.predictive_model = PredictiveModel(self.config)
			elif self.config.get_pred('model_kind') == None:
				self.log('You must pass an object with a predict() method as a predictive model', 'FATAL')
		else:
			self.log('Callable predictive model specified - Updating model', 'WARNING')
			self.predictive_model = predictive_model
			setattr(self.predictive_model, 'is_internal', False)

		from datetime import datetime
		start_time = datetime.now()

		if observations is None:
			# no observations, need to fall back to random sampling
			samples = self.random_sampler.draw(num=self.config.get('batches') * self.config.get('sampling_strategies'))
			if self.config.process_constrained:
				dominant_features = self.config.feature_process_constrained
				samples[:, dominant_features] = samples[0, dominant_features]

		elif len(observations) == 0:
			self.log('Could not find any observations, falling back to random sampling', 'WARNING')
			samples = self.random_sampler.draw(num=self.config.get('batches') * self.config.get('sampling_strategies'))
			if self.config.process_constrained:
				dominant_features = self.config.feature_process_constrained
				samples[:, dominant_features] = samples[0, dominant_features]

		else:
			# TODO : clean up these checks
			if proxy_observations is None:
				if len(observations) >= 2 and not self.predictive_model.is_internal:
					pass
				elif len(observations) >= 2 and self.predictive_model.is_internal:
					self.log('To use a built in predictive model, you must provide some proxy measurements to recommend', 'FATAL')

			else:
				if len(observations) >= 2 and len(proxy_observations) == 0 and self.predictive_model.is_internal:
					self.log('To use a built in predictive model, you must provide some proxy measurements to recommend', 'FATAL')


			# extract descriptors
			descriptors = self.descriptor_generator.get_descriptors()
			descriptors_feas = self.descriptor_generator_feas.get_descriptors()

			# get arrays of params and objectives  (i.e. without the duplicated points)
			raw_params, raw_objs = self.training_set_gen.get_arrays(observations)

			# process the observations
			obs_params_kwn, obs_objs_kwn, mirror_mask_kwn, obs_params_ukwn, obs_objs_ukwn, mirror_mask_ukwn = self.obs_processor.process(observations)
			if self.predictive_model.is_internal:
				# process the proxy observations
#				proxy_obs_params_kwn, proxy_obs_objs_kwn, proxy_mirror_mask_kwn, proxy_obs_params_ukwn, proxy_obs_objs_ukwn, proxy_mirror_mask_ukwn = self.obs_processor.process(proxy_observations)

				# get the raw proxy observations
				proxy_raw_params, proxy_raw_objs = self.training_set_gen.get_arrays(proxy_observations)

				# set the descriptors
				self.training_set_gen.set_descriptors(descriptors)
				# construct the training set
				training_set = self.training_set_gen.construct_training_set(raw_params, raw_objs,
															 				proxy_raw_params, proxy_raw_objs)
				# train the predictive model - re-train regime
				self.predictive_model.train(training_set)


			# run descriptor generation
			if self.config.get('auto_desc_gen'):
				if len(obs_params_kwn) > 2:
					self.descriptor_generator.generate(obs_params_kwn, obs_objs_kwn)
				if len(obs_params_ukwn) > 2:
					self.descriptor_generator_feas.generate(obs_params_ukwn, obs_objs_ukwn)

			# get lambda values for exploration/exploitation
			sampling_param_values = self.bayesian_network.sampling_param_values
			dominant_strategy_index = self.iter_counter % len(sampling_param_values)
			dominant_strategy_value = np.array([sampling_param_values[dominant_strategy_index]])

			# sample bnn for known parameters
			if obs_params_kwn.shape[0] > 0:
				self.bayesian_network.sample(obs_params_kwn, obs_objs_kwn)
				self.bayesian_network.build_kernels(descriptors)
				kernel_contribution = self.bayesian_network.kernel_contribution
				# if we have kwn samples ==> pick params with best merit
				best_params = obs_params_kwn[np.argmin(obs_objs_kwn)]
			else:
				# empty kernel contributions - cannot sample anything is obs_params_kwn is empty!
				kernel_contribution = self.bayesian_network.empty_kernel_contribution
				# if we have do not have any feasible sample ==> pick any feasible param at random
				best_params_idx = np.random.choice(np.flatnonzero(obs_objs_ukwn == obs_objs_ukwn.min()))
				best_params = obs_params_ukwn[best_params_idx]

			# get sensitivity parameter and do some checks
			feas_sensitivity = self.config.get('feas_sensitivity')
			if feas_sensitivity < 0.0:
				self.log('Config parameter `feas_sensitivity` should be positive, applying np.abs()', 'WARNING')
				feas_sensitivity = np.abs(feas_sensitivity)
			elif feas_sensitivity == 0.0:
				self.log('Config parameter `feas_sensitivity` cannot be zero, falling back to default value of 1', 'WARNING')
				feas_sensitivity = 1.0

			# sample from BNN for feasibility surrogate is we have at least one unfeasible point
			if obs_params_ukwn.shape[0] > 0:
				self.bayesian_network_feas.sample(obs_params_ukwn, obs_objs_ukwn)
				self.bayesian_network_feas.build_kernels(descriptors_feas)
				# fraction of unfeasible samples - use mask to avoid counting mirrored samples
				unfeas_frac = sum(obs_objs_ukwn[mirror_mask_ukwn]) / len(obs_objs_ukwn[mirror_mask_ukwn])
				unfeas_frac = unfeas_frac ** feas_sensitivity  # adjust sensitivity to presence of unfeasible samples
				kernel_contribution_feas = self.bayesian_network_feas.kernel_contribution
			else:
				kernel_contribution_feas = self.bayesian_network_feas.empty_kernel_contribution
				unfeas_frac = 0.

			# if there are process constraining parameters, run those first
			if self.config.process_constrained:
				proposed_samples = self.acquisition.propose(best_params,
															kernel_contribution,
															kernel_contribution_feas,
															unfeas_frac,
															sampling_param_values,
															predictive_model=self.predictive_model,
															training_set_gen=self.training_set_gen,
															inverse_volume=self.bayesian_network.inverse_volume,
															dominant_samples=None)
				constraining_samples = self.sample_selector.select(self.config.get('batches'),
																   proposed_samples,
																   kernel_contribution,
																   kernel_contribution_feas,
																   unfeas_frac,
																   dominant_strategy_value,
																   obs_params_ukwn,
																   predictive_model=self.predictive_model,
																   training_set_gen=self.training_set_gen,
																   inverse_volume=self.bayesian_network.inverse_volume,)
			else:
				constraining_samples = None

			# then select the remaining proposals
			proposed_samples = self.acquisition.propose(best_params,
														kernel_contribution,
														kernel_contribution_feas,
														unfeas_frac,
														sampling_param_values,
														predictive_model=self.predictive_model,
														training_set_gen=self.training_set_gen,
														inverse_volume=self.bayesian_network.inverse_volume,
														dominant_samples=constraining_samples)

			# note: provide `obs_params_ukwn` as it contains the params for _all_ samples, including the unfeasible ones
			samples = self.sample_selector.select(self.config.get('batches'),
												  proposed_samples,
												  kernel_contribution,
												  kernel_contribution_feas,
												  unfeas_frac,
												  sampling_param_values,
												  obs_params_ukwn,
												  predictive_model=self.predictive_model,
												  training_set_gen=self.training_set_gen,
			  									  inverse_volume=self.bayesian_network.inverse_volume)

			# store info so to be able to recontruct surrogate and acquisition function if needed
			self.last_kernel_contribution = kernel_contribution
			self.last_kernel_contribution_feas = kernel_contribution_feas
			self.last_sampling_param_values = sampling_param_values
			self.last_unfeas_frac = unfeas_frac
			self.last_params_kwn = obs_params_kwn[mirror_mask_kwn]
			self.last_objs_kwn = obs_objs_kwn[mirror_mask_kwn]
			self.last_params_ukwn = obs_params_ukwn[mirror_mask_ukwn]
			self.last_objs_ukwn = obs_objs_ukwn[mirror_mask_ukwn]
			self.last_recommended_samples = samples

		end_time = datetime.now()
		self.log('[TIME]:  ' + str(end_time - start_time) + ',   (overall)', 'INFO')

		if as_array:
			# return as is
			return_samples = samples
		else:
			# convert to list of dictionaries
			param_names   = self.config.param_names
			param_options = self.config.param_options
			param_types   = self.config.param_types
			sample_dicts  = []
			for sample in samples:
				sample_dict  = {}
				lower, upper = 0, self.config.param_sizes[0]
				for param_index, param_name in enumerate(param_names):
					param_type = param_types[param_index]

					if param_type == 'continuous':
						sample_dict[param_name] = sample[lower:upper]

					elif param_type == 'categorical':
						options = param_options[param_index]
						parsed_options = [options[int(element)] for element in sample[lower:upper]]
						sample_dict[param_name] = parsed_options

					elif param_type == 'discrete':
						options = param_options[param_index]
						parsed_options = [options[int(element)] for element in sample[lower:upper]]
						sample_dict[param_name] = parsed_options

					if param_index == len(self.config.param_names) - 1:
						break
					lower  = upper
					upper += self.config.param_sizes[param_index + 1]
				sample_dicts.append(sample_dict)
			return_samples = sample_dicts

		if self.config.get('save_database') is True:
			db_entry = {'start_time': start_time, 'end_time': end_time,
						'received_obs': observations, 'suggested_params': return_samples}
			if self.config.get('auto_desc_gen') is True:
				# save summary of learned descriptors
				descriptor_summary = self.descriptor_generator.get_summary()
				db_entry['descriptor_summary'] = descriptor_summary
			self.db_handler.save(db_entry)

		self.iter_counter += 1
		return return_samples

	def construct_training_set(self, observations, proxy_observations=None):
		''' Used to generate training set for external predictive model
		'''
		# set descriptors
		descriptors = self.descriptor_generator.get_descriptors()
		self.training_set_gen.set_descriptors(descriptors)
		# process the observations
#		obs_params_kwn, obs_objs_kwn, mirror_mask_kwn, _, __, ___ = self.obs_processor.process(observations)
#		proxy_obs_params_kwn, proxy_obs_objs_kwn, proxy_mirror_mask_kwn, _, __, ___ = self.obs_processor.process(proxy_observations)
		raw_params, raw_objs = self.training_set_gen.get_arrays(observations)
		proxy_raw_params, proxy_raw_objs = self.training_set_gen.get_arrays(proxy_observations)
		# construct the training set
		training_set = self.training_set_gen.construct_training_set(raw_params, raw_objs,
											  						proxy_raw_params, proxy_raw_objs)

		return training_set


	def construct_context_set(self, observations):
		''' Used to generate training set for external predictive model
		'''
		# set descriptors
		descriptors = self.descriptor_generator.get_descriptors()
		self.training_set_gen.set_descriptors(descriptors)
		# process observations
#		obs_params_kwn, obs_objs_kwn, mirror_mask_kwn, _, __, ___ = self.obs_processor.process(observations)
		raw_params, raw_objs = self.training_set_gen.get_arrays(observations)
		# construct the context set
		context_set = self.training_set_gen.construct_context_set(raw_params, raw_objs)

		return context_set

	def read_db(self, outfile='database.csv', verbose=True):
		self.db_handler.read_db(outfile, verbose)

	def get_surrogate(self, x):
		"""
		Retrieve the last surrogate function
		"""
		pass

	def get_acquisition(self, x, lambda_strategy=None, separate=False):
		"""
		Retrieve the last acquisition functions for a specific lambda value.
		"""
		num, inv_den = self.last_kernel_contribution(x)
		num_feas, inv_den_feas = self.last_kernel_contribution_feas(x)
		if lambda_strategy is None:
			values = []
			for l in self.last_sampling_param_values:
				acq_samp = (num + l) * inv_den
				acq_feas = (num_feas + l) * inv_den_feas
				if separate is False:
					acquisition = self.last_unfeas_frac * acq_feas + (1. - self.last_unfeas_frac) * acq_samp
					values.append(acquisition)
				else:
					values.append([acq_samp, acq_feas])
			return values
		else:
			acq_samp = (num + lambda_strategy) * inv_den
			acq_feas = (num_feas + lambda_strategy) * inv_den_feas
			if separate is False:
				acquisition = self.last_unfeas_frac * acq_feas + (1. - self.last_unfeas_frac) * acq_samp
				return acquisition
			else:
				return acq_samp, acq_feas
