#!/usr/bin/env python 

__author__ = 'Florian Hase'

from .acquisition import Acquisition
from .bayesian_network import BayesianNetwork
from .descriptor_generator import DescriptorGenerator
from .observation_processor import ObservationProcessor, param_vectors_to_dicts
from .random_sampler import RandomSampler
from .sample_selector import SampleSelector
from .utilities import ConfigParser, Logger, GryffinNotFoundError
from .utilities import parse_time, memory_usage

import os
import numpy as np
import pandas as pd
import time


class Gryffin(Logger):

    def __init__(self, config_file=None, config_dict=None, known_constraints=None):

        Logger.__init__(self, 'Gryffin', verbosity=0)

        # parse configuration
        self.config = ConfigParser(config_file, config_dict)
        self.config.parse()
        self.config.set_home(os.path.dirname(os.path.abspath(__file__)))

        # parse constraints function
        self.known_constraints = known_constraints

        np.random.seed(self.config.get('random_seed'))
        self.update_verbosity(self.config.get('verbosity'))
        self._create_folders()

        # Instantiate all objects needed
        self.random_sampler = RandomSampler(self.config, self.known_constraints)
        self.obs_processor = ObservationProcessor(self.config)
        self.descriptor_generator = DescriptorGenerator(self.config)
        self.descriptor_generator_feas = DescriptorGenerator(self.config)
        self.bayesian_network = BayesianNetwork(config=self.config, classification=False)
        self.bayesian_network_feas = BayesianNetwork(config=self.config, classification=True)
        self.acquisition = Acquisition(self.config, self.known_constraints)
        self.sample_selector = SampleSelector(self.config)

        self.iter_counter = 0
        self.sampling_param_values = None
        self.sampling_strategies = None

    def _create_folders(self):
        if self.config.get('save_database') is True and not os.path.isdir(self.config.get_db('path')):
            try:
                os.mkdir(self.config.get_db('path'))
            except FileNotFoundError:
                GryffinNotFoundError('Could not create database directory: %s' % self.config.get_db('path'))

        if self.config.get('save_database') is True:
            from .database_handler import DatabaseHandler
            self.db_handler = DatabaseHandler(self.config)

    def recommend(self, observations=None, sampling_strategies=None, as_array=False):
        """Recommends the next set(s) of parameters based on the provided observations.

        Parameters
        ----------
        observations : list
        sampling_strategies : list
        as_array : bool

        Returns
        -------
        params : list
        """

        start_time = time.time()
        if sampling_strategies is None:
            num_sampling_strategies = self.config.get('sampling_strategies')
            sampling_strategies = np.linspace(1, -1, num_sampling_strategies)
        else:
            sampling_strategies = np.array(sampling_strategies)
            num_sampling_strategies = len(sampling_strategies)

        # register last sampling strategies
        self.sampling_strategies = sampling_strategies

        # no observations, need to fall back to random sampling
        if observations is None or len(observations) == 0:
            self.log('Could not find any observations, falling back to random sampling', 'WARNING')
            samples = self.random_sampler.draw(num=self.config.get('batches') * num_sampling_strategies)
            if self.config.process_constrained:
                dominant_features = self.config.feature_process_constrained
                samples[:, dominant_features] = samples[0, dominant_features]

        # we have observations
        else:
            obs_params_kwn, obs_objs_kwn, mirror_mask_kwn, \
            obs_params_ukwn, obs_objs_ukwn, mirror_mask_ukwn = self.obs_processor.process_observations(observations)

            # run descriptor generation
            if self.config.get('auto_desc_gen'):
                if len(obs_params_kwn) > 2:
                    self.descriptor_generator.generate_descriptors(obs_params_kwn, obs_objs_kwn)
                # we run descriptor generation for unknown points only if we have at least 1 infeasible point,
                #  otherwise they are all feasible and there is no point running this. Remember that
                #  feasible = 0 and infeasible = 1.
                if len(obs_params_ukwn) > 2 and np.sum(obs_objs_ukwn) > 0.1:
                    self.descriptor_generator_feas.generate_descriptors(obs_params_ukwn, obs_objs_ukwn)

            # extract descriptors and build kernels
            descriptors = self.descriptor_generator.get_descriptors()
            descriptors_feas = self.descriptor_generator_feas.get_descriptors()

            # get lambda values for exploration/exploitation
            self.sampling_param_values = sampling_strategies * self.bayesian_network.inverse_volume
            dominant_strategy_index = self.iter_counter % len(self.sampling_param_values)
            dominant_strategy_value = np.array([self.sampling_param_values[dominant_strategy_index]])

            # sample bnn for known parameters
            if obs_params_kwn.shape[0] > 0:
                self.bayesian_network.sample(obs_params_kwn, obs_objs_kwn)
                self.bayesian_network.build_kernels(descriptors)
                # if we have kwn samples ==> pick params with best merit
                best_params = obs_params_kwn[np.argmin(obs_objs_kwn)]
            else:
                # if we have do not have any feasible sample ==> pick any feasible param at random
                best_params_idx = np.random.choice(np.flatnonzero(obs_objs_ukwn == obs_objs_ukwn.min()))
                best_params = obs_params_ukwn[best_params_idx]

            # If we have called build_kernels, we'll have actual kernels
            # otherwise the default kernel_contribution return (0, volume)
            kernel_contribution = self.bayesian_network.kernel_contribution

            # sample from BNN for feasibility surrogate is we have at least one unfeasible point
            if np.sum(obs_objs_ukwn) > 0.1:
                # use mask to avoid using mirrored samples here
                self.bayesian_network_feas.sample(obs_params_ukwn[mirror_mask_ukwn], obs_objs_ukwn[mirror_mask_ukwn])
                self.bayesian_network_feas.build_kernels(descriptors_feas)

            # If we have called build_kernels, we'll have actual surrogate
            # otherwise surrogate always returns zero, i.e. feasible
            probability_infeasible = self.bayesian_network_feas.surrogate
            # prior_1 is fraction of unfeasible samples
            frac_infeasible = self.bayesian_network_feas.prior_1

            # if there are process constraining parameters, run those first
            if self.config.process_constrained:
                proposed_samples = self.acquisition.propose(best_params, kernel_contribution,
                                                            probability_infeasible, frac_infeasible,
                                                            self.sampling_param_values, num_samples=200,
                                                            dominant_samples=None)
                constraining_samples = self.sample_selector.select(self.config.get('batches'), proposed_samples,
                                                                   self.acquisition.eval_acquisition,
                                                                   dominant_strategy_value, obs_params_ukwn)
            else:
                constraining_samples = None

            # then select the remaining proposals
            # note num_samples get multiplied by the number of input variables
            proposed_samples = self.acquisition.propose(best_params, kernel_contribution, probability_infeasible,
                                                        frac_infeasible, self.sampling_param_values, num_samples=200,
                                                        dominant_samples=constraining_samples)

            # note: provide `obs_params_ukwn` as it contains the params for _all_ samples, including the unfeasible ones
            samples = self.sample_selector.select(self.config.get('batches'), proposed_samples,
                                                  self.acquisition.eval_acquisition,
                                                  self.sampling_param_values, obs_params_ukwn)

            # store info so to be able to recontruct surrogate and acquisition function if needed
            self.last_kernel_contribution = kernel_contribution
            self.last_probability_infeasible = probability_infeasible
            self.last_sampling_strategies = sampling_strategies
            self.last_frac_infeasibile = frac_infeasible
            self.last_params_kwn = obs_params_kwn[mirror_mask_kwn]
            self.last_objs_kwn = obs_objs_kwn[mirror_mask_kwn]
            self.last_params_ukwn = obs_params_ukwn[mirror_mask_ukwn]
            self.last_objs_ukwn = obs_objs_ukwn[mirror_mask_ukwn]
            self.last_recommended_samples = samples

        GB, MB, kB = memory_usage()
        self.log(f'[MEM]:  {GB} GB, {MB} MB, {kB} kB', 'INFO')
        end_time = time.time()
        self.log('[TIME]:  ' + parse_time(start_time, end_time) + '  (overall)', 'INFO')

        if as_array:
            # return as is
            return_samples = samples
        else:
            # return as dictionary
            return_samples = param_vectors_to_dicts(param_vectors=samples, param_names=self.config.param_names,
                                                    param_options=self.config.param_options,
                                                    param_types=self.config.param_types)

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

    def read_db(self, outfile='database.csv', verbose=True):
        self.db_handler.read_db(outfile, verbose)

    @staticmethod
    def _df_to_list_of_dicts(df):
        list_of_dicts = []
        for index, row in df.iterrows():
            d = {}
            for col in df.columns:
                d[col] = [row[col]]
            list_of_dicts.append(d)
        return list_of_dicts

    def get_surrogate(self, params, feasibility=False):
        """
        Retrieve the surrogate function.

        Parameters
        ----------
        params : list or DataFrame
            list of dicts with input parameters to evaluate. Alternatively it can also be a pandas DataFrame where
            each column name corresponds to one of the input parameters in Gryffin.
        feasibility : bool
            whether to return the feasibility surrogate. Default is False.

        Returns
        -------
        y_pred : list
            surrogate model evaluated at the locations defined in params.
        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        if feasibility is True:
            surrogate = self.bayesian_network_feas.surrogate
        else:
            surrogate = self.bayesian_network.surrogate

        X = self.obs_processor.process_params(params)
        y_preds = []
        for x in X:
            y_pred = surrogate(x)
            y_preds.append(y_pred)
        return y_preds

    def get_acquisition(self, X):
        """
        Retrieve the last acquisition functions for a specific lambda value.
        """
        if isinstance(X, pd.DataFrame):
            X = self._df_to_list_of_dicts(X)
        X_parsed = self.obs_processor.process_params(X)

        # collect acquisition values
        acquisition_values = {}
        for batch_index, sampling_param in enumerate(self.sampling_param_values):
            acquisition_values_at_l = []
            for Xi_parsed in X_parsed:
                acq_value = self.acquisition.eval_acquisition(Xi_parsed, batch_index)
                acquisition_values_at_l.append(acq_value)

            lambda_value = self.sampling_strategies[batch_index]
            acquisition_values[lambda_value] = acquisition_values_at_l

        return acquisition_values

