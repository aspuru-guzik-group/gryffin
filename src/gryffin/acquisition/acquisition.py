#!/usr/bin/env python 

__author__ = 'Florian Hase'

import numpy as np
import time
import multiprocessing
from multiprocessing import Process, Manager

from . import ParameterOptimizer
from gryffin.random_sampler import RandomSampler
from gryffin.utilities      import Logger, parse_time


class Acquisition(Logger):

    def __init__(self, config):

        self.config = config
        Logger.__init__(self, 'Acquisition', self.config.get('verbosity'))
        self.random_sampler = RandomSampler(self.config.general, self.config.parameters)
        self.total_num_vars = len(self.config.feature_names)

        self.kernel_contribution = None
        self.probability_infeasible = None
        self.local_optimizers = None
        self.sampling_param_values = None
        self.frac_infeasible = None
        self.acqs_min_max = None  # expected content is dict where key is batch_index, and dict[batch_index] = [min,max]

        # figure out how many CPUs to use
        if self.config.get('num_cpus') == 'all':
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get('num_cpus'))

        # get sensitivity parameter and do some checks
        self.feas_sensitivity = self.config.get('feas_sensitivity')
        if self.feas_sensitivity < 0.0:
            self.log('Config parameter `feas_sensitivity` should be positive, applying np.abs()', 'WARNING')
            self.feas_sensitivity = np.abs(self.feas_sensitivity)
        elif self.feas_sensitivity == 0.0:
            self.log('Config parameter `feas_sensitivity` cannot be zero, falling back to default value of 1',
                        'WARNING')
            self.feas_sensitivity = 1.0

    def _propose_randomly(self, best_params, num_samples, dominant_samples=None):
        # get uniform samples
        if dominant_samples is None:
            uniform_samples = self.random_sampler.draw(num=self.total_num_vars * num_samples)
            perturb_samples = self.random_sampler.perturb(best_params, num=self.total_num_vars * num_samples)
            samples         = np.concatenate([uniform_samples, perturb_samples])
        else:
            dominant_features = self.config.feature_process_constrained
            for batch_sample in dominant_samples:
                uniform_samples = self.random_sampler.draw(num=self.total_num_vars * num_samples // len(dominant_samples))
                perturb_samples = self.random_sampler.perturb(best_params, num=self.total_num_vars * num_samples)
                samples         = np.concatenate([uniform_samples, perturb_samples])
            samples[:, dominant_features] = batch_sample[dominant_features]
        return samples

    def _proposal_optimization_thread(self, proposals, kernel_contribution, probability_infeasible,
                                      batch_index, return_index, acq_min, acq_max, return_dict=None, dominant_samples=None):
        self.log('starting process for %d' % batch_index, 'INFO')

        # get lambda value
        sampling_param = self.sampling_param_values[batch_index]

        # define the acquisition functions
        def acquisition_standard(x):
            num, inv_den = kernel_contribution(x)  # standard acquisition for samples
            prob_infeas = probability_infeasible(x)  # feasibility acquisition
            acq_samp = (num + sampling_param) * inv_den
            # approximately normalize sample acquisition so it has same scale of prob_infeas
            acq_samp = (acq_samp - acq_min) / (acq_max - acq_min)
            return feasibility_weight * prob_infeas + (1. - feasibility_weight) * acq_samp

        # if all feasible, prob_infeas always zero, so no need to estimate feasibility
        def acquisition_all_feasible(x):
            num, inv_den = kernel_contribution(x)  # standard acquisition for samples
            acq_samp = (num + sampling_param) * inv_den
            return acq_samp

        # if all infeasible, acquisition is flat, so no need to compute it
        def acquisition_all_infeasible(x):
            prob_infeas = probability_infeasible(x)
            return prob_infeas

        # select the relevant acquisition
        if self.frac_infeasible == 0:
            acquisition = acquisition_all_feasible
            feasibility_weight = None  # i.e. not used
        elif self.frac_infeasible == 1:
            acquisition = acquisition_all_infeasible
            feasibility_weight = None  # i.e. not used
        else:
            acquisition = acquisition_standard
            feasibility_weight = self.frac_infeasible ** self.feas_sensitivity

        # get params to be constrained
        if dominant_samples is not None:
            ignore = self.config.feature_process_constrained
        else:
            ignore = np.array([False for _ in range(len(self.config.feature_process_constrained))])

        # get the optimizer instance and set function to be optimized
        local_optimizer = self.local_optimizers[batch_index]
        local_optimizer.set_func(acquisition, ignores=ignore)

        # run acquisition optimization
        optimized = []
        for sample_index, sample in enumerate(proposals):
            opt = local_optimizer.optimize(acquisition, sample, max_iter=10)
            optimized.append(opt)
        optimized = np.array(optimized)

        if return_dict.__class__.__name__ == 'DictProxy':
            return_dict[return_index] = optimized
        else:
            return optimized

    def _get_approx_min_max(self, random_proposals, kernel_contribution, sampling_param):
        """Approximate min and max of sample acquisition to that we can approximately normalize it"""
        acq_values = []
        for proposal in random_proposals:
            num, inv_den = kernel_contribution(proposal)
            acq_samp = (num + sampling_param) * inv_den
            acq_values.append(acq_samp)
        return np.min(acq_values), np.max(acq_values)

    def _optimize_proposals(self, random_proposals, kernel_contribution, probability_infeasible, dominant_samples=None):

        self.acqs_min_max = {}

        # -------------------
        # parallel processing
        # -------------------
        if self.num_cpus > 1:
            result_dict = Manager().dict()

            # get the number of splits
            num_splits = self.num_cpus // len(self.sampling_param_values) + 1
            split_size = len(random_proposals) // num_splits

            processes = []
            for batch_index, sampling_param in enumerate(self.sampling_param_values):
                # get approximate min/max of sample acquisition
                acq_min, acq_max = self._get_approx_min_max(random_proposals, kernel_contribution, sampling_param)
                self.acqs_min_max[batch_index] = [acq_min, acq_max]

                # for all splits
                for split_index in range(num_splits):

                    split_start  = split_size * split_index
                    split_end    = split_size * (split_index + 1)
                    return_index = num_splits * batch_index + split_index
                    # run optimization
                    process = Process(target=self._proposal_optimization_thread, args=(random_proposals[split_start: split_end],
                                                                                       kernel_contribution, probability_infeasible,
                                                                                       batch_index, return_index,
                                                                                       acq_min, acq_max,
                                                                                       result_dict, dominant_samples))
                    processes.append(process)
                    process.start()

            for process_index, process in enumerate(processes):
                process.join()

        # ---------------------
        # sequential processing
        # ---------------------
        else:
            num_splits = 1
            result_dict = {}
            for batch_index, sampling_param in enumerate(self.sampling_param_values):
                # get approximate min/max of sample acquisition
                acq_min, acq_max = self._get_approx_min_max(random_proposals, kernel_contribution, sampling_param)
                self.acqs_min_max[batch_index] = [acq_min, acq_max]

                # run the optimization
                return_index = batch_index
                result_dict[batch_index] = self._proposal_optimization_thread(random_proposals, kernel_contribution,
                                                                              probability_infeasible,
                                                                              batch_index, return_index,
                                                                              acq_min, acq_max,
                                                                              dominant_samples=dominant_samples)

        # -------------------------
        # collect optimized samples
        # -------------------------
        samples = []
        for batch_index in range(len(self.sampling_param_values)):
            batch_samples = []
            for split_index in range(num_splits):
                return_index = num_splits * batch_index + split_index
                batch_samples.append(result_dict[return_index])
            samples.append(np.concatenate(batch_samples))
        samples = np.array(samples)
        return np.array(samples)

    def propose(self, best_params, kernel_contribution, probability_infeasible, frac_infeasible, sampling_param_values,
                num_samples=200, dominant_samples=None):

        # define optimizers
        self.local_optimizers = [ParameterOptimizer(self.config) for _ in range(len(sampling_param_values))]
        assert len(self.local_optimizers) == len(sampling_param_values)

        # define some attributes we'll be using
        self.sampling_param_values = sampling_param_values
        self.frac_infeasible = frac_infeasible
        self.kernel_contribution = kernel_contribution
        self.probability_infeasible = probability_infeasible

        # get random samples
        random_proposals = self._propose_randomly(best_params, num_samples, dominant_samples=dominant_samples)

        # run acquisition optimization starting from random samples
        start = time.time()
        optimized_proposals = self._optimize_proposals(random_proposals, kernel_contribution, probability_infeasible,
                                                       dominant_samples=dominant_samples)
        end = time.time()
        self.log('[TIME]:  ' + parse_time(start, end) + '  (optimizing proposals)', 'INFO')

        extended_proposals = np.array([random_proposals for _ in range(len(sampling_param_values))])
        combined_proposals = np.concatenate((extended_proposals, optimized_proposals), axis=1)

        return combined_proposals

    def eval_acquisition(self, x, batch_index):

        sampling_param = self.sampling_param_values[batch_index]  # lambda value
        feasibility_weight = self.frac_infeasible ** self.feas_sensitivity  # feas weight
        acq_min, acq_max = self.acqs_min_max[batch_index]

        num, inv_den = self.kernel_contribution(x)  # standard acquisition for samples
        prob_infeas = self.probability_infeasible(x)  # feasibility acquisition
        acq_samp = (num + sampling_param) * inv_den

        # approximately normalize sample acquisition so it has same scale of prob_infeas
        acq_samp = (acq_samp - acq_min) / (acq_max - acq_min)

        acq_value = feasibility_weight * prob_infeas + (1. - feasibility_weight) * acq_samp
        return acq_value

