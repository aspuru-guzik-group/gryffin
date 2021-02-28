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
        self.random_sampler   = RandomSampler(self.config.general, self.config.parameters)
        self.total_num_vars   = len(self.config.feature_names)
        self.local_optimizers = None
        # figure out how many CPUs to use
        if self.config.get('num_cpus') == 'all':
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get('num_cpus'))

    def _propose_randomly(self, best_params, num_samples, dominant_samples = None):
        # get uniform samples
        if dominant_samples is None:
            uniform_samples = self.random_sampler.draw(num = self.total_num_vars * num_samples)
            perturb_samples = self.random_sampler.perturb(best_params, num = self.total_num_vars * num_samples)
            samples         = np.concatenate([uniform_samples, perturb_samples])
        else:
            dominant_features = self.config.feature_process_constrained
            for batch_sample in dominant_samples:
                uniform_samples = self.random_sampler.draw(num = self.total_num_vars * num_samples // len(dominant_samples))
                perturb_samples = self.random_sampler.perturb(best_params, num = self.total_num_vars * num_samples)
                samples         = np.concatenate([uniform_samples, perturb_samples])
            samples[:, dominant_features] = batch_sample[dominant_features]
        return samples

    def _proposal_optimization_thread(self, proposals, kernel_contribution, kernel_contribution_feas, unfeas_frac,
                                      batch_index, return_index, return_dict = None, dominant_samples = None):
        self.log('starting process for %d' % batch_index, 'INFO')

        def kernel(x):
            num, inv_den = kernel_contribution(x)  # standard acquisition for samples
            num_feas, inv_den_feas = kernel_contribution_feas(x)  # feasibility acquisition
            acq_samp = (num + self.sampling_param_values[batch_index]) * inv_den
            acq_feas = (num_feas + self.sampling_param_values[batch_index]) * inv_den_feas
            return unfeas_frac * acq_feas + (1. - unfeas_frac) * acq_samp

        if dominant_samples is not None:
            ignore = self.config.feature_process_constrained
        else:
            ignore = np.array([False for _ in range(len(self.config.feature_process_constrained))])

        local_optimizer = self.local_optimizers[batch_index]
        local_optimizer.set_func(kernel, ignores = ignore)

        optimized = []
        for sample_index, sample in enumerate(proposals):
            opt = local_optimizer.optimize(kernel, sample, max_iter = 10)
            optimized.append(opt)
        optimized = np.array(optimized)

        if return_dict.__class__.__name__ == 'DictProxy':
            return_dict[return_index] = optimized
        else:
            return optimized

    def _optimize_proposals(self, random_proposals, kernel_contribution, kernel_contribution_feas, unfeas_frac, dominant_samples=None):

        if self.num_cpus > 1:
            result_dict = Manager().dict()

            # get the number of splits
            num_splits = self.num_cpus // len(self.sampling_param_values) + 1
            split_size = len(random_proposals) // num_splits

            processes   = []
            for batch_index in range(len(self.sampling_param_values)):
                for split_index in range(num_splits):

                    split_start  = split_size * split_index
                    split_end    = split_size * (split_index + 1)
                    return_index = num_splits * batch_index + split_index
                    process = Process(target=self._proposal_optimization_thread, args=(random_proposals[split_start: split_end],
                                                                                       kernel_contribution, kernel_contribution_feas,
                                                                                       unfeas_frac, batch_index, return_index,
                                                                                       result_dict, dominant_samples))
                    processes.append(process)
                    process.start()

            for process_index, process in enumerate(processes):
                process.join()

        else:
            num_splits  = 1
            result_dict = {}
            for batch_index in range(len(self.sampling_param_values)):
                return_index = batch_index
                result_dict[batch_index] = self._proposal_optimization_thread(random_proposals, kernel_contribution,
                                                                              kernel_contribution_feas, unfeas_frac,
                                                                              batch_index, return_index,
                                                                              dominant_samples=dominant_samples)

        # collect optimized samples
        samples = []
        for batch_index in range(len(self.sampling_param_values)):
            batch_samples = []
            for split_index in range(num_splits):
                return_index = num_splits * batch_index + split_index
                batch_samples.append(result_dict[return_index])
            samples.append(np.concatenate(batch_samples))
        samples = np.array(samples)
        return np.array(samples)

    def propose(self, best_params, kernel_contribution, kernel_contribution_feas, unfeas_frac, sampling_param_values,
                num_samples=200, dominant_samples=None):

        self.local_optimizers = [ParameterOptimizer(self.config) for _ in range(len(sampling_param_values))]
        assert len(self.local_optimizers) == len(sampling_param_values)

        self.sampling_param_values = sampling_param_values
        random_proposals = self._propose_randomly(
            best_params, num_samples, dominant_samples=dominant_samples,
        )


        start = time.time()
        optimized_proposals = self._optimize_proposals(random_proposals, kernel_contribution, kernel_contribution_feas,
                                                       unfeas_frac, dominant_samples=dominant_samples)
        end = time.time()
        self.log('[TIME]:  ' + parse_time(start, end) + '  (optimizing proposals)', 'INFO')

        extended_proposals = np.array([random_proposals for _ in range(len(sampling_param_values))])
        combined_proposals = np.concatenate((extended_proposals, optimized_proposals), axis=1)

        return combined_proposals

