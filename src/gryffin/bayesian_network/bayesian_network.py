#!/usr/bin/env python

__author__ = 'Florian Hase'

import time
import numpy as np
from . import CategoryReshaper
from gryffin.utilities import Logger, parse_time
from gryffin.utilities import GryffinUnknownSettingsError
from .kernel_evaluations import KernelEvaluator
from .tfprob_interface import run_tf_network


class BayesianNetwork(Logger):

    def __init__(self, config, classification=False, model_details=None):

        self.COUNTER = 0
        self.has_sampled = False
        self.config = config

        # get domain volume
        self.volume = None
        self.inverse_volume = None
        self._get_volume()

        # variables for kernel density classification
        self.classification = classification
        self.prior_0 = 1  # default prior is all feasible
        self.prior_1 = 0
        self.log_prior_0 = None
        self.log_prior_1 = None
        self.surrogate = lambda x: 0.0  # default to returning constant if surrogate not defined

        # variables for kernel density estimation and regression
        self.kernel_contribution = lambda x: (0.0, self.volume)  # return den=0, inv_dev=volume
        self.cat_reshaper = CategoryReshaper(self.config)

        # verbosity settings
        verbosity = self.config.get('verbosity')
        if 'bayesian_network' in verbosity:
            verbosity = verbosity['bayesian_network']
        Logger.__init__(self, 'BayesianNetwork', verbosity=verbosity)

        # get bnn model details
        if model_details is None:
            self.model_details = self.config.model_details.to_dict()
        else:
            self.model_details = model_details

        # whether to use kernel caching
        self.caching = self.config.get('caching')
        self.cache = None

    def _get_volume(self):
        # get domain volume
        self.volume = 1.
        feature_lengths = self.config.feature_lengths
        feature_ranges = self.config.feature_ranges
        for feature_index, feature_type in enumerate(self.config.feature_types):
            if feature_type == 'continuous':
                self.volume *= feature_ranges[feature_index]
            elif feature_type == 'categorical':
                self.volume *= feature_lengths[feature_index]
            elif feature_type == 'discrete':
                self.volume *= feature_ranges[feature_index]
            else:
                GryffinUnknownSettingsError(
                    'did not understand parameter type: "%s" of variable "%s".\n\t(%s) Please choose from "continuous" or "categorical"' % (
                    feature_type, self.config.feature_names[feature_index], self.template))
        self.inverse_volume = 1 / self.volume

    def sample(self, obs_params, obs_objs):

        trace_kernels, obs_objs = run_tf_network(obs_params, obs_objs, self.config, self.model_details)
        self.trace_kernels = trace_kernels
        self.obs_objs = obs_objs

        # if we are classifying feasible points, update priors
        if self.classification is True:
            # feasible = 0 and infeasible = 1
            self.prior_0 = sum([xi < 0.5 for xi in self.obs_objs]) / len(self.obs_objs)
            self.prior_1 = sum([xi > 0.5 for xi in self.obs_objs]) / len(self.obs_objs)
            assert np.abs((self.prior_0 + self.prior_1) - 1.0) < 10e-5

        # set sampling to true
        self.has_sampled = True

    def build_kernels(self, descriptors):
        assert self.has_sampled
        trace_kernels = self.trace_kernels
        obs_objs      = self.obs_objs

        # shape of the tensors below: (# samples, # obs, # kernels)
        locs       = trace_kernels['locs']
        sqrt_precs = trace_kernels['sqrt_precs']
        probs      = trace_kernels['probs']

        start = time.time()
        try:
            if np.all(np.isfinite(np.array(descriptors, dtype=np.float))):
                probs = self.cat_reshaper.reshape(probs, descriptors)
        except ValueError:
            probs = self.cat_reshaper.reshape(probs, descriptors)
        end = time.time()

        # report cat reshaping time only if we have categorical variables
        if 'categorical' in self.config.kernel_types:
            self.log('[TIME]:  ' + parse_time(start, end) + '  (reshaping categorical space)', 'INFO')

        # write kernel types
        kernel_type_strings = self.config.kernel_types
        kernel_types = []
        for kernel_type_string in kernel_type_strings:
            if kernel_type_string == 'continuous':
                kernel_types.append(0)
            elif kernel_type_string == 'categorical':
                kernel_types.append(1)
            elif kernel_type_string == 'discrete':
                kernel_types.append(1)
        kernel_types = np.array(kernel_types, dtype=np.int32)
        kernel_sizes = self.config.kernel_sizes.astype(np.int32)

        # get lower prob bound
        if self.config.get('boosted'):
            lower_prob_bound = 1e-1
            for size in self.config.feature_ranges:
                lower_prob_bound *= 1. / size
        else:
            lower_prob_bound = 1e-25

        self.kernel_evaluator = KernelEvaluator(locs, sqrt_precs, probs, kernel_types, kernel_sizes, lower_prob_bound, obs_objs, self.inverse_volume)

        # check if we can construct a cache and update option
        if self.caching is True:
            if np.sum(kernel_types) == len(kernel_types):
                self.cache = {}
            else:
                self.caching = False

        # -------------------------------------------------------
        # define functions that use self.kernel_evaluator methods
        # -------------------------------------------------------
        def kernel_contribution(proposed_sample):
            if self.caching is True:
                sample_string = '-'.join([str(int(element)) for element in proposed_sample])
                if sample_string in self.cache:
                    num, inv_den = self.cache[sample_string]
                else:
                    num, inv_den, _ = self.kernel_evaluator.get_kernel(proposed_sample.astype(np.float64))
                    self.cache[sample_string] = (num, inv_den)
            else:
                num, inv_den, _ = self.kernel_evaluator.get_kernel(proposed_sample.astype(np.float64))
            return num, inv_den

        def prob_infeasible(proposed_sample):
            return self.kernel_evaluator.get_probability_of_infeasibility(proposed_sample.astype(np.float64),
                                                                          self.log_prior_0,
                                                                          self.log_prior_1)

        def infeasible_kernel_density(proposed_sample):
            _, log_density_1 = self.kernel_evaluator.get_binary_kernel_densities(proposed_sample.astype(np.float64))
            return np.exp(log_density_1)

        self.kernel_contribution = kernel_contribution

        if self.classification is True:
            # if prior_0 == 1, then prob_infeasible == 0
            if 1.0 - self.prior_0 < 0.01:  # use 1% threshold to speed up computations if prior_0 close to 1
                self.surrogate = lambda x: 0.0
            # if prior_1 == 1, then all infeasible => use kernel density as penalty
            elif 1.0 - self.prior_1 < 0.01:  # use 1% threshold to speed up computations if prior_1 close to 1
                self.surrogate = infeasible_kernel_density
            else:
                # compute log priors and use posterior of kernel density classification model
                self.log_prior_0 = np.log(self.prior_0)
                self.log_prior_1 = np.log(self.prior_1)
                self.surrogate = prob_infeasible
        else:
            self.surrogate = self.regression_surrogate

    def regression_surrogate(self, proposed_sample):
        y_pred = self.kernel_evaluator.get_regression_surrogate(proposed_sample.astype(np.float64))
        return y_pred

    def empty_kernel_contribution(self, proposed_sample):
        num = 0.
        inv_den = self.volume  # = 1/p(x) = 1/inverse_volume
        return num, inv_den










