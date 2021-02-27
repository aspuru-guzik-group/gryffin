#!/usr/bin/env python

__author__ = 'Florian Hase'

import time
import pickle
import numpy as np
from . import CategoryReshaper
from gryffin.utilities import Logger
from gryffin.utilities import GryffinUnknownSettingsError
from .kernel_evaluations import KernelEvaluator
from .tfprob_interface import TfprobNetwork


class BayesianNetwork(Logger):

    def __init__(self, config, model_details=None):

        self.COUNTER = 0
        self.has_sampled = False
        self.config = config
        verbosity = self.config.get('verbosity')
        if 'bayesian_network' in verbosity:
            verbosity = verbosity['bayesian_network']
        Logger.__init__(self, 'BayesianNetwork', verbosity=verbosity)
        self.kernel_contribution = lambda x: (np.sum(x), 1.)
        self.surrogate = lambda x: None  # default to returning None if surrogate not defined
        self.cat_reshaper = CategoryReshaper(self.config)

        # get bnn model detals
        if model_details == None:
            from .model_details import model_details
        self.model_details = model_details

        # get domain volume
        self.volume     = 1.
        feature_lengths = self.config.feature_lengths
        feature_ranges  = self.config.feature_ranges
        for feature_index, feature_type in enumerate(self.config.feature_types):
            if feature_type == 'continuous':
                self.volume *= feature_ranges[feature_index]
            elif feature_type == 'categorical':
                self.volume *= feature_lengths[feature_index]
            elif feature_type == 'discrete':
                self.volume *= feature_ranges[feature_index]
            else:
                GryffinUnknownSettingsError('did not understand parameter type: "%s" of variable "%s".\n\t(%s) Please choose from "continuous" or "categorical"' % (feature_type, self.config.feature_names[feature_index], self.template))
        self.inverse_volume = 1 / self.volume

        # compute sampling parameter values
        if self.config.get('sampling_strategies') == 1:
            self.sampling_param_values = np.zeros(1)
        else:
            self.sampling_param_values = np.linspace(-1.0, 1.0, self.config.get('sampling_strategies'))
            self.sampling_param_values = self.sampling_param_values[::-1]
        self.sampling_param_values *= self.inverse_volume

    def sample(self, obs_params, obs_objs):

        tfprob_network = TfprobNetwork(self.config, self.model_details)
        tfprob_network.declare_training_data(obs_params, obs_objs)
        tfprob_network.construct_model()
        tfprob_network.sample()

        trace_kernels, obs_objs = tfprob_network.get_kernels()
        self.trace_kernels = trace_kernels
        self.obs_objs = obs_objs

        # set sampling to true
        self.has_sampled = True

    def build_kernels(self, descriptors):
        assert self.has_sampled
        trace_kernels = self.trace_kernels
        obs_objs      = self.obs_objs

        # shape of the tensors below: (# samples, # obs, # kernels)
        burnin, thinning = self.model_details['burnin'], self.model_details['thinning']
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
        self.log('ELAPSED TIME (cat reshaping) ' + str(end - start), 'INFO')

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

        # check if we can construct a cache
        self.caching = np.sum(kernel_types) == len(kernel_types)
        self.cache   = {}

        def kernel_contribution(proposed_sample):
            if self.caching:
                sample_string = '-'.join([str(int(element)) for element in proposed_sample])
                if sample_string in self.cache:
                    num, inv_den  = self.cache[sample_string]
                else:
                    num, inv_den, _ = self.kernel_evaluator.get_kernel(proposed_sample.astype(np.float64))
                    self.cache[sample_string] = (num, inv_den)
            else:
                num, inv_den, _ = self.kernel_evaluator.get_kernel(proposed_sample.astype(np.float64))
            return num, inv_den

        def surrogate(proposed_sample):
            y_pred = self.kernel_evaluator.get_surrogate(proposed_sample.astype(np.float64))
            return y_pred

        self.kernel_contribution = kernel_contribution
        self.surrogate = surrogate

    def empty_kernel_contribution(self, proposed_sample):
        num = 0.
        inv_den = self.volume  # = 1/p(x) = 1/inverse_volume
        return num, inv_den










