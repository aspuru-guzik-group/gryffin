#!/usr/bin/env python 

__author__ = 'Florian Hase, Matteo Aldeghi'

from .acquisition import Acquisition
from .bayesian_network import BayesianNetwork
from .descriptor_generator import DescriptorGenerator
from .observation_processor import ObservationProcessor, param_vectors_to_dicts, param_dicts_to_vectors
from .random_sampler import RandomSampler
from .sample_selector import SampleSelector
from .utilities import ConfigParser, Logger, GryffinNotFoundError
from .utilities import parse_time, memory_usage

import os
import numpy as np
import pandas as pd
import time


class Gryffin(Logger):

    def __init__(self, config_file=None, config_dict=None, known_constraints=None, silent=False):
        """
        silent : bool
            whether to suppress all standard output. If True, the ``verbosity`` settings in ``config`` will be
            overwritten. Default is False.
        """

        # parse configuration
        self.config = ConfigParser(config_file, config_dict)
        self.config.parse()
        self.config.set_home(os.path.dirname(os.path.abspath(__file__)))

        # set verbosity
        if silent is True:
            self.verbosity = 2
            self.config.general.verbosity = 2
        else:
            self.verbosity = self.config.get('verbosity')

        Logger.__init__(self, 'Gryffin', verbosity=self.verbosity)

        # parse constraints function
        self.known_constraints = known_constraints

        # store timings for possible analysis
        self.timings = {}

        np.random.seed(self.config.get('random_seed'))
        self._create_folders()  # folders created only if we are saving to database

        # Instantiate all objects needed
        self.random_sampler = RandomSampler(self.config, constraints=self.known_constraints)
        self.obs_processor = ObservationProcessor(self.config)
        self.descriptor_generator = DescriptorGenerator(self.config)
        self.descriptor_generator_feas = DescriptorGenerator(self.config)
        self.bayesian_network = BayesianNetwork(config=self.config)
        self.acquisition = Acquisition(self.config, known_constraints=self.known_constraints)
        self.sample_selector = SampleSelector(self.config)

        self.iter_counter = 0
        self.sampling_param_values = None
        self.sampling_strategies = None
        self.num_batches = None
        # attributes used mainly for investigation/debugging
        self.parsed_input_data = {}
        self.proposed_samples = None

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
        self.log('', 'INFO')
        self.log_chapter("Gryffin", line='=', style='bold #d9ed92')

        start_time = time.time()
        if sampling_strategies is None:
            num_sampling_strategies = self.config.get('sampling_strategies')
            sampling_strategies = np.linspace(1, -1, num_sampling_strategies)
        else:
            sampling_strategies = np.array(sampling_strategies)
            num_sampling_strategies = len(sampling_strategies)

        # register last sampling strategies
        self.sampling_strategies = sampling_strategies
        self.num_batches = self.config.get('batches')

        # print summary of what will be proposed
        num_recommended_samples = self.num_batches * num_sampling_strategies
        samples_str = 'samples' if num_recommended_samples > 1 else 'sample'
        batches_str = 'batches' if self.num_batches > 1 else 'batch'
        strategy_str = 'strategies' if num_sampling_strategies > 1 else 'strategy'
        self.log(f'Gryffin will propose {num_recommended_samples} {samples_str}: {self.num_batches} {batches_str} with'
                 f' {num_sampling_strategies} sampling {strategy_str}', 'INFO')

        # -----------------------------------------------------
        # no observations, need to fall back to random sampling
        # -----------------------------------------------------
        if observations is None or len(observations) == 0:
            self.log('Could not find any observations, falling back to random sampling', 'WARNING')
            samples = self.random_sampler.draw(num=num_recommended_samples)
            if self.config.process_constrained:
                dominant_features = self.config.feature_process_constrained
                samples[:, dominant_features] = samples[0, dominant_features]

        # --------------------
        # we have observations
        # --------------------
        else:
            self.log(f'{len(observations)} observations found', 'INFO')
            # obs_params == all observed parameters
            # obs_objs == all observed objective function evaluations (including NaNs)
            # obs_feas == whether observed parameters are feasible (0) or infeasible (1)
            # mask_kwn == mask that selects only known/feasible params/objs (including mirrored params)
            # mask_mirror == mask that selects the parameters that have been mirrored across opt bounds
            obs_params, obs_objs, obs_feas, mask_kwn, mask_mirror = self.obs_processor.process_observations(observations)

            # keep for inspection/debugging
            self.parsed_input_data['obs_params'] = obs_params
            self.parsed_input_data['obs_objs'] = obs_objs
            self.parsed_input_data['obs_feas'] = obs_feas
            self.parsed_input_data['mask_kwn'] = mask_kwn
            self.parsed_input_data['mask_mirror'] = mask_mirror

            # -----------------------------
            # Build categorical descriptors
            # -----------------------------
            can_generate_desc = len(obs_params[mask_kwn]) > 2 or (len(obs_params) > 2 and np.sum(obs_feas) > 0.1)
            if self.config.get('auto_desc_gen') is True and can_generate_desc is True:
                self.log_chapter('Descriptor Refinement')
                start = time.time()
                with self.console.status("Refining categories descriptors..."):
                    # only feasible points with known objectives
                    if len(obs_params[mask_kwn]) > 2:
                        self.descriptor_generator.generate_descriptors(obs_params[mask_kwn], obs_objs[mask_kwn])
                    # for feasibility descriptors, we use all data, but we run descriptor generation
                    # only if we have at least 1 infeasible point, otherwise they are all feasible and there is no point
                    # running this. Remember that feasible = 0 and infeasible = 1.
                    if len(obs_params) > 2 and np.sum(obs_feas) > 0.1:
                        self.descriptor_generator_feas.generate_descriptors(obs_params[~mask_kwn], obs_params[~mask_kwn])

                end = time.time()
                time_string = parse_time(start, end)
                self.log(f"Categorical descriptors refined by [italic]Dynamic Gryffin[/italic] in {time_string}",
                         "INFO")

            # extract descriptors and build kernels
            descriptors_kwn = self.descriptor_generator.get_descriptors()
            descriptors_feas = self.descriptor_generator_feas.get_descriptors()

            # ----------------------------------------------
            # get lambda values for exploration/exploitation
            # ----------------------------------------------
            self.sampling_param_values = sampling_strategies * self.bayesian_network.inverse_volume
            dominant_strategy_index = self.iter_counter % len(self.sampling_param_values)
            dominant_strategy_value = np.array([self.sampling_param_values[dominant_strategy_index]])

            # ----------------------------------------------
            # sample bnn to get kernels for all observations
            # ----------------------------------------------
            self.log_chapter('Bayesian Network')
            self.bayesian_network.sample(obs_params)  # infer kernel densities
            # build kernel smoothing/classification surrogates
            self.bayesian_network.build_kernels(descriptors_kwn=descriptors_kwn, descriptors_feas=descriptors_feas,
                                                obs_objs=obs_objs, obs_feas=obs_feas, mask_kwn=mask_kwn)

            # get incumbent
            if len(obs_params[mask_kwn]) > 0:
                # if we have kwn samples ==> pick params with best merit
                best_params = obs_params[mask_kwn][np.argmin(obs_objs[mask_kwn])]
            else:
                # if we have do not have any feasible sample ==> pick any feasible param at random
                best_params_idx = np.random.randint(low=0, high=len(obs_params))
                best_params = obs_params[best_params_idx]

            # ----------------------------------------------
            # optimize acquisition and select samples
            # ----------------------------------------------

            # if there are process constraining parameters, run those first
            if self.config.process_constrained:
                self.proposed_samples = self.acquisition.propose(best_params, self.bayesian_network,
                                                                 self.sampling_param_values, num_samples=200,
                                                                 dominant_samples=None)
                constraining_samples = self.sample_selector.select(self.num_batches, self.proposed_samples,
                                                                   self.acquisition.eval_acquisition,
                                                                   dominant_strategy_value, obs_params)
            else:
                constraining_samples = None

            # then select the remaining proposals
            # note num_samples get multiplied by the number of input variables
            self.log_chapter('Acquisition')
            self.proposed_samples = self.acquisition.propose(best_params=best_params,
                                                             bayesian_network=self.bayesian_network,
                                                             sampling_param_values=self.sampling_param_values,
                                                             num_samples=200, dominant_samples=constraining_samples,
                                                             timings_dict=self.timings)

            self.log_chapter('Sample Selector')
            # note: provide `obs_params` as it contains the params for _all_ samples, including the unfeasible ones
            samples = self.sample_selector.select(num_samples=self.num_batches, proposals=self.proposed_samples,
                                                  eval_acquisition=self.acquisition.eval_acquisition,
                                                  sampling_param_values=self.sampling_param_values,
                                                  obs_params=obs_params)

        # --------------------------------
        # Print overall info for recommend
        # --------------------------------
        self.log_chapter('Summary')
        GB, MB, kB = memory_usage()
        self.log(f'Memory usage: {GB:.0f} GB, {MB:.0f} MB, {kB:.0f} kB', 'INFO')
        end_time = time.time()
        time_string = parse_time(start_time, end_time)
        self.log(f'Overall time required: {time_string}', 'INFO')
        self.log_chapter("End", line='=', style='bold #d9ed92')
        self.log('', 'INFO')

        # -----------------------
        # Return proposed samples
        # -----------------------
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
                d[col] = row[col]
            list_of_dicts.append(d)
        return list_of_dicts

    def get_regression_surrogate(self, params):
        """
        Retrieve the surrogate model.

        Parameters
        ----------
        params : list or DataFrame
            list of dicts with input parameters to evaluate. Alternatively it can also be a pandas DataFrame where
            each column name corresponds to one of the input parameters in Gryffin.

        Returns
        -------
        y_pred : list
            surrogate model evaluated at the locations defined in params.
        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        X = param_dicts_to_vectors(params, param_names=self.config.param_names,
                                   param_options=self.config.param_options, param_types=self.config.param_types)
        y_preds = []
        for x in X:
            y_pred = self.bayesian_network.regression_surrogate(x)
            y_preds.append(y_pred)

        # invert tranform the surrogate according to the chosen transform
        y_preds = np.array(y_preds)
        transform = self.config.get('obj_transform')
        if transform is None:
            pass
        elif transform == 'sqrt':
            # accentuate global minimum
            y_preds = np.square(y_preds)
        elif transform == 'cbrt':
            # accentuate global minimum more than sqrt
            y_preds = np.power(y_preds, 3)
        elif transform == 'square':
            # de-emphasise global minimum
            y_preds = np.sqrt(y_preds)

        # scale the predicted objective back to the original range
        if self.obs_processor.min_obj != self.obs_processor.max_obj:
            y_preds = y_preds * (self.obs_processor.max_obj - self.obs_processor.min_obj) + self.obs_processor.min_obj
        else:
            y_preds = y_preds + self.obs_processor.min_obj

        return y_preds

    def get_feasibility_surrogate(self, params, threshold=None):
        """
        Retrieve the feasibility surrogate model.

        Parameters
        ----------
        params : list or DataFrame
            list of dicts with input parameters to evaluate. Alternatively it can also be a pandas DataFrame where
            each column name corresponds to one of the input parameters in Gryffin.
        threshold : float
            the threshold used to classify whether a set of parameters is feasible or not. If ``None``, the probability
            of feasibility is returned instead of a binary True/False (feasible/infeasible) output. Default is None.

        Returns
        -------
        y_pred : list
            surrogate model evaluated at the locations defined in params.
        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        X = param_dicts_to_vectors(params, param_names=self.config.param_names,
                                   param_options=self.config.param_options, param_types=self.config.param_types)
        y_preds = []
        for x in X:
            if threshold is None:
                y_pred = self.bayesian_network.prob_feasible(x)
            else:
                y_pred = self.bayesian_network.classification_surrogate(x, threshold=threshold)
            y_preds.append(y_pred)
        return np.array(y_preds)

    def get_kernel_density_estimate(self, params, separate_kwn_ukwn=False):
        """
        Retrieve the feasibility surrogate model.

        Parameters
        ----------
        params : list or DataFrame
            list of dicts with input parameters to evaluate. Alternatively it can also be a pandas DataFrame where
            each column name corresponds to one of the input parameters in Gryffin.
        separate_kwn_ukwn : bool
            whether to return the density for all samples, or to separate the density for feasible/infeasible samples.

        Returns
        -------
        y_pred : list
            kernel density estimates.
        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        X = param_dicts_to_vectors(params, param_names=self.config.param_names,
                                   param_options=self.config.param_options, param_types=self.config.param_types)
        y_preds = []
        for x in X:
            log_density_0, log_density_1 = self.bayesian_network.kernel_classification.get_binary_kernel_densities(x)
            density_0 = np.exp(log_density_0)
            density_1 = np.exp(log_density_1)
            if separate_kwn_ukwn is True:
                y_pred = [density_0, density_1]
            else:
                y_pred = density_0 + density_1
            y_preds.append(y_pred)
        return np.array(y_preds)

    def get_acquisition(self, X):
        """
        Retrieve the last acquisition functions for a specific lambda value.
        """
        if isinstance(X, pd.DataFrame):
            X = self._df_to_list_of_dicts(X)
        X_parsed = param_dicts_to_vectors(X, param_names=self.config.param_names,
                                          param_options=self.config.param_options,
                                          param_types=self.config.param_types)

        # collect acquisition values
        acquisition_values = {}
        for batch_index, sampling_param in enumerate(self.sampling_param_values):
            acquisition_values_at_l = []
            for Xi_parsed in X_parsed:
                acq_value = self.acquisition.eval_acquisition(Xi_parsed, batch_index)
                acquisition_values_at_l.append(acq_value)

            lambda_value = self.sampling_strategies[batch_index]
            acquisition_values[lambda_value] = np.array(acquisition_values_at_l)

        return acquisition_values

