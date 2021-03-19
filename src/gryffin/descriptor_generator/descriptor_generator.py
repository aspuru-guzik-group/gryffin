#!/usr/bin/env python 

__author__ = 'Florian Hase'

import copy
import time
import numpy as np
import multiprocessing
from gryffin.utilities import Logger, parse_time
from .generation_process import run_generator_network


class DescriptorGenerator(Logger):

    eta = 1e-3
    max_iter = 10**3
    def __init__(self, config):

        self.config        = config
        self.is_generating = False

        # define registers
        self.auto_gen_descs     = {}
        self.comp_corr_coeffs   = {}
        self.gen_descs_cov      = {}
        self.min_corrs          = {}
        self.reduced_gen_descs  = {}
        self.weights            = {}
        self.sufficient_indices = {}

        self.obs_params = None
        self.obs_objs = None
        self.gen_feature_descriptors = None

        verbosity = self.config.get('verbosity')
        Logger.__init__(self, 'DescriptorGenerator', verbosity=verbosity)

        if self.config.get('num_cpus') == 'all':
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get('num_cpus'))

    def generate_single_descriptors(self, feature_index):
        """Parse description generation for a specific parameter, ad indicated by the feature_index"""

        feature_types = self.config.feature_types
        feature_descriptors = self.config.feature_descriptors
        obs_params = self.obs_params
        obs_objs = self.obs_objs

        # if continuous ==> no descriptors, return None
        if feature_types[feature_index] in ['continuous', 'discrete']:
            self.weights[feature_index] = None
            self.reduced_gen_descs[feature_index] = None
            return None, feature_index

        # if None, i.e. naive Gryffin ==> no descriptors, return None
        if feature_descriptors[feature_index] is None:
            self.weights[feature_index] = None
            self.reduced_gen_descs[feature_index] = None
            return None, feature_index

        # if single descriptor ==> cannot get new descriptors, return the same static descriptor
        if feature_descriptors[feature_index].shape[1] == 1:
            self.weights[feature_index] = np.array([[1.]])
            self.reduced_gen_descs[feature_index] = feature_descriptors[feature_index]
            return feature_descriptors[feature_index], feature_index

        # ------------------------------------------------------------------------------------------
        # Else, we have multiple descriptors for a categorical variable and we perform the reshaping
        # ------------------------------------------------------------------------------------------
        params = obs_params[:, feature_index].astype(np.int32)
        descs = feature_descriptors[feature_index][params]
        objs = np.reshape(obs_objs, (len(obs_objs), 1))

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

        # run the generation process
        results = run_generator_network(sim_dict)

        self.min_corrs[feature_index]          = results['min_corrs']
        self.auto_gen_descs[feature_index]     = results['auto_gen_descs']
        self.comp_corr_coeffs[feature_index]   = results['comp_corr_coeffs']
        self.gen_descs_cov[feature_index]      = results['gen_descs_cov']
        self.reduced_gen_descs[feature_index]  = results['reduced_gen_descs']
        self.weights[feature_index]            = results['weights']
        self.sufficient_indices[feature_index] = results['sufficient_indices']

        return results['reduced_gen_descs'], feature_index

    def generate_descriptors(self, obs_params, obs_objs):
        """Generates descriptors for each categorical parameters"""

        start = time.time()

        self.obs_params = obs_params
        self.obs_objs = obs_objs
        result_dict = {}

        feature_indices = range(len(self.config.feature_options))

        # TODO: implement multiprocessing with the code below. The problem is CPU bound and multiprocessing could
        #  speed things up. However, this currently does not work probably because the tf graph is not pickable due
        #  to a lock. We could use multi-threading but not sure this would help much since we do not have I/O issues
        #  here and all threads would run on the same core that is CPU bound anyway.
        #if self.num_cpus >= 2:
            # asynchronous execution across processes
        #    with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
        #        for gen_descriptor, feature_index in executor.map(self.generate_single_descriptors, feature_indices):
        #            result_dict[feature_index] = gen_descriptor
        #else:
        #    for feature_index in feature_indices:
        #        gen_descriptor, _ = self.generate_single_descriptors(feature_index)
        #        result_dict[feature_index] = gen_descriptor

        for feature_index in feature_indices:
            gen_descriptor, _ = self.generate_single_descriptors(feature_index)
            result_dict[feature_index] = gen_descriptor

        # we use this to reorder correctly the descriptors following asynchronous execution
        gen_feature_descriptors = [result_dict[feature_index] for feature_index in range(len(result_dict.keys()))]
        self.gen_feature_descriptors = gen_feature_descriptors

        end = time.time()
        desc_gen_time = parse_time(start, end)
        self.log('[TIME]:  ' + desc_gen_time + '  (descriptor generation)', 'INFO')

    def get_descriptors(self):
        if self.gen_feature_descriptors is not None:
            return self.gen_feature_descriptors
        else:
            return self.config.feature_descriptors

    def get_summary(self):

        summary = {}
        feature_types = self.config.feature_types
        # If we have not generated new descriptors
        if self.gen_feature_descriptors is None:
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


