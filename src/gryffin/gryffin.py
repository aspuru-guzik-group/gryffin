#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os, sys
import numpy as np 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .acquisition           import Acquisition
from .bayesian_network      import BayesianNetwork
from .descriptor_generator  import DescriptorGenerator
from .observation_processor import ObservationProcessor
from .random_sampler        import RandomSampler
from .sample_selector       import SampleSelector
from .utilities             import ConfigParser, Logger, GryffinNotFoundError

#========================================================================

class Gryffin(Logger):

	def __init__(self, config_file = None, config_dict = None):
	
		Logger.__init__(self, 'Gryffin', verbosity = 0)

		# parse configuration
		self.config = ConfigParser(config_file, config_dict)
		self.config.parse()
		self.config.set_home(os.path.dirname(os.path.abspath(__file__)))
	
		np.random.seed(self.config.get('random_seed'))	
		self.update_verbosity(self.config.get('verbosity'))
		self.create_folders()

		self.random_sampler       = RandomSampler(self.config.general, self.config.parameters)
		self.obs_processor        = ObservationProcessor(self.config)
		self.descriptor_generator = DescriptorGenerator(self.config)
		self.bayesian_network     = BayesianNetwork(self.config)
		self.acquisition          = Acquisition(self.config)
		self.sample_selector      = SampleSelector(self.config)

		self.iter_counter = 0


	def create_folders(self):

		if not os.path.isdir(self.config.get('scratch_dir')):
			try:
				os.mkdir(self.config.get('scratch_dir'))
			except FileNotFoundError:
				GryffinNotFoundError('Could not create scratch directory: %s' % self.config.get('scratch_dir'))

		if self.config.get_db('has_db') and not os.path.isdir(self.config.get_db('path')):
			try:
				os.mkdir(self.config.get_db('path'))
			except FileNotFoundError:
				GryffinNotFoundError('Could not create database directory: %s' % self.config.get_db('path'))		

		if self.config.get_db('has_db'):
			from .database_handler import DatabaseHandler
			self.db_handler = DatabaseHandler(self.config)



	def recommend(self, observations = None, as_array = False):
		
		from datetime import datetime
		start_time = datetime.now()

		if observations is None:
			# no observations, need to fall back to random sampling
			samples = self.random_sampler.draw(num = self.config.get('batches') * self.config.get('sampling_strategies'))
			if self.config.process_constrained:
				dominant_features = self.config.feature_process_constrained
				samples[:, dominant_features] = samples[0, dominant_features]

		elif len(observations) == 0:
			self.log('Could not find any observations, falling back to random sampling', 'WARNING')
			samples = self.random_sampler.draw(num = self.config.get('batches') * self.config.get('sampling_strategies'))
			if self.config.process_constrained:
				dominant_features = self.config.feature_process_constrained
				samples[:, dominant_features] = samples[0, dominant_features]

		else:
			obs_params, obs_objs = self.obs_processor.process(observations)	

			# run descriptor generation
			if self.config.get('auto_desc_gen') and len(obs_params) > 2:
				self.descriptor_generator.generate(obs_params, obs_objs)
			
			self.bayesian_network.sample(obs_params, obs_objs)

			# extract descriptors and build kernels
			descriptors = self.descriptor_generator.get_descriptors()

			self.bayesian_network.build_kernels(descriptors)
			sampling_param_values   = self.bayesian_network.sampling_param_values
			dominant_strategy_index = self.iter_counter % len(sampling_param_values)
			dominant_strategy_value = np.array([sampling_param_values[dominant_strategy_index]])

			# prepare sample generation / selection
			best_params         = obs_params[np.argmin(obs_objs)]
			kernel_contribution = self.bayesian_network.kernel_contribution

			# if there are process constraining parameters, run those first
			if self.config.process_constrained:
				proposed_samples     = self.acquisition.propose(best_params, kernel_contribution, dominant_strategy_value)
				constraining_samples = self.sample_selector.select(self.config.get('batches'), proposed_samples, kernel_contribution, dominant_strategy_value, obs_params)
			else:
				constraining_samples = None

			# then select the remaining proposals
			proposed_samples = self.acquisition.propose(
					best_params, kernel_contribution, sampling_param_values, 
					dominant_samples  = constraining_samples,
					dominant_strategy = dominant_strategy_index,
				)

			samples = self.sample_selector.select(
					self.config.get('batches'), proposed_samples, kernel_contribution, sampling_param_values, obs_params
				)


		end_time = datetime.now()
		self.log('[TIME]:  ' + str(end_time - start_time) + ',   (overall)', 'INFO')
		#print('[TIME]:  ', end_time - start_time, '  (overall)')
		#print('***********************************************')


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


		if self.config.get_db('has_db'):
			db_entry = {'start_time': start_time, 'end_time': end_time, 
						'received_obs': observations, 'suggested_params': return_samples}
			if self.config.get('auto_desc_gen'):
				# get summary of learned descriptors
				descriptor_summary = self.descriptor_generator.get_summary()
				db_entry['descriptor_summary'] = descriptor_summary
			self.db_handler.save(db_entry)

		self.iter_counter += 1
		return return_samples



	def read_db(self, outfile = 'database.csv', verbose = True):
		self.db_handler.read_db(outfile, verbose)


#========================================================================

if __name__ == '__main__':

	observations = [
			{'param_0': [-1.0, -1.0], 'param_1':  [1.0], 'obj_0': 0.1, 'obj_1': 0.2},
			{'param_0': [1.0, 1.0],  'param_1': [-1.0], 'obj_0': 0.2, 'obj_1': 0.1},
		]

	gryffin = Gryffin()
	samples = gryffin.recommend(observations = observations)
	print(samples)
