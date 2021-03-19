#!/usr/bin/env python 

__author__ = 'Florian Hase'

import numpy as np
import time
import multiprocessing
from multiprocessing import Process, Manager

from gryffin.acquisition import GradientOptimizer
from gryffin.random_sampler import RandomSampler
from gryffin.utilities import Logger, parse_time, GryffinUnknownSettingsError


class Acquisition(Logger):

    def __init__(self, config, known_constraints=None):

        self.config = config
        self.known_constraints = known_constraints
        Logger.__init__(self, 'Acquisition', self.config.get('verbosity'))
        self.random_sampler = RandomSampler(self.config, known_constraints)
        self.total_num_vars = len(self.config.feature_names)
        self.optimizer_type = self.config.get('acquisition_optimizer')

        self.kernel_contribution = None
        self.probability_infeasible = None
        self.local_optimizers = None
        self.sampling_param_values = None
        self.frac_infeasible = None
        self.acqs_min_max = None  # expected content is dict where key is batch_index, and dict[batch_index] = [min,max]
        self.acquisition_functions = {}  # to keep the AcquisitionFunction instances used

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

    def _proposal_optimization_thread(self, proposals, acquisition, batch_index,
                                      return_dict=None, return_index=0, dominant_samples=None):
        if return_dict is not None:
            self.log('running parallel process for lambda strategy number %d' % batch_index, 'INFO')
        else:
            self.log('running serial process for lambda strategy number %d' % batch_index, 'INFO')

        # get params to be constrained
        if dominant_samples is not None:
            ignore = self.config.feature_process_constrained
        else:
            ignore = np.array([False for _ in range(len(self.config.feature_process_constrained))])

        # get the optimizer instance and set function to be optimized
        local_optimizer = self.local_optimizers[batch_index]
        local_optimizer.set_func(acquisition, ignores=ignore)

        # run acquisition optimization
        optimized = local_optimizer.optimize(proposals, max_iter=10)

        if return_dict.__class__.__name__ == 'DictProxy':
            return_dict[return_index] = optimized

        return optimized

    def _get_approx_min_max(self, random_proposals, sampling_param, dominant_samples):
        """Approximate min and max of sample acquisition to that we can approximately normalize it"""

        # If we only have feasible or infeasible points, no need to compute max/min as there is no need to rescale the
        # sample acquisition, because the acquisition will only be for feasible samples or for feasibility search
        if self.frac_infeasible == 0 or self.frac_infeasible == 1:
            return 0.0, 1.0

        acq_values = []
        for proposal in random_proposals:
            num, inv_den = self.kernel_contribution(proposal)
            acq_samp = (num + sampling_param) * inv_den
            acq_values.append(acq_samp)

        acq_values = np.array(acq_values)

        # take top/bottom 5% of samples...
        n = int(round(len(random_proposals) * 0.05, 0))
        indices_top = (-acq_values).argsort()[:n]  # indices of highest n
        indices_bottom = acq_values.argsort()[:n]  # indices of lowest n

        top_params = random_proposals[indices_top, :]  # params of highest n
        bottom_params = random_proposals[indices_bottom, :]  # params of lowest n

        # define acquisition function to be optimized. With frac_infeasible=0 we choose the sample acquisition only,
        # and with acq_min=0, acq_max=1 we are not scaling it
        acquisition = AcquisitionFunction(kernel_contribution=self.kernel_contribution,
                                          probability_infeasible=self.probability_infeasible,
                                          sampling_param=sampling_param, frac_infeasible=0,
                                          acq_min=0, acq_max=1, feas_sensitivity=1.0)

        # get params to be constrained
        if dominant_samples is not None:
            ignore = self.config.feature_process_constrained
        else:
            ignore = np.array([False for _ in range(len(self.config.feature_process_constrained))])

        # ----------------------
        # minimise lowest values
        # ----------------------
        optimizer_bottom = GradientOptimizer(self.config, self.known_constraints)
        optimizer_bottom.set_func(acquisition, ignores=ignore)
        optimized = optimizer_bottom.optimize(bottom_params, max_iter=10)

        bottom_acq_values = np.array([acquisition(x) for x in optimized])
        # concatenate with randomly collected acq values
        bottom_acq_values = np.concatenate((acq_values, bottom_acq_values), axis=0)

        # -----------------------
        # maximise highest values
        # -----------------------
        def inv_acquisition(x):
            """Invert acquisition for its maximisation"""
            return -acquisition(x)

        optimizer_top = GradientOptimizer(self.config, self.known_constraints)
        optimizer_top.set_func(inv_acquisition, ignores=ignore)
        optimized = optimizer_top.optimize(top_params, max_iter=10)

        top_acq_values = np.array([acquisition(x) for x in optimized])
        # concatenate with randomly collected acq values
        top_acq_values = np.concatenate((acq_values, top_acq_values), axis=0)

        return np.min(bottom_acq_values), np.max(top_acq_values)

    def _optimize_proposals(self, random_proposals, dominant_samples=None):

        optimized_samples = []  # all optimized samples, i.e. for all sampling strategies
        self.acqs_min_max = {}

        # ------------------------------------
        # Iterate over all sampling strategies
        # ------------------------------------
        for batch_index, sampling_param in enumerate(self.sampling_param_values):

            # get approximate min/max of sample acquisition
            acq_min, acq_max = self._get_approx_min_max(random_proposals, sampling_param, dominant_samples)
            self.acqs_min_max[batch_index] = [acq_min, acq_max]

            # define acquisition function to be optimized
            acquisition = AcquisitionFunction(kernel_contribution=self.kernel_contribution,
                                              probability_infeasible=self.probability_infeasible,
                                              sampling_param=sampling_param, frac_infeasible=self.frac_infeasible,
                                              acq_min=acq_min, acq_max=acq_max,
                                              feas_sensitivity=self.feas_sensitivity)

            # save acquisition instance for future use
            if batch_index not in self.acquisition_functions.keys():
                self.acquisition_functions[batch_index] = acquisition

            # -------------------
            # parallel processing
            # -------------------
            if self.num_cpus > 1:
                # create shared memory dict that will contain the optimized samples for this batch/sampling strategy
                # keys will correspond to indices so that we can resort the samples afterwards
                return_dict = Manager().dict()

                # split random_proposals into approx equal chunks based on how many CPUs we're using
                random_proposals_splits = np.array_split(random_proposals, self.num_cpus)

                # parallelize over splits
                # -----------------------
                processes = []  # store parallel processes here
                for idx, random_proposals_split in enumerate(random_proposals_splits):
                    # run optimization
                    process = Process(target=self._proposal_optimization_thread, args=(random_proposals_split,
                                                                                       acquisition,
                                                                                       batch_index,
                                                                                       return_dict,
                                                                                       idx,
                                                                                       dominant_samples))
                    processes.append(process)
                    process.start()

                # wait until all processes finished
                for process in processes:
                    process.join()

                # sort results in return_dict to create optimized_batch_samples list with correct sample order
                optimized_batch_samples = []
                for idx in range(len(random_proposals_splits)):
                    optimized_batch_samples.extend(return_dict[idx])

            # ---------------------
            # sequential processing
            # ---------------------
            else:
                # optimized samples for this batch/sampling strategy
                optimized_batch_samples = self._proposal_optimization_thread(proposals=random_proposals,
                                                                             acquisition=acquisition,
                                                                             batch_index=batch_index,
                                                                             return_dict=None,
                                                                             return_index=0,
                                                                             dominant_samples=dominant_samples)

            # append the optimized samples for this sampling strategy to the global list of optimized_samples
            optimized_samples.append(optimized_batch_samples)

        return np.array(optimized_samples)

    def _load_optimizers(self, num):
        if self.optimizer_type == 'adam':
            local_optimizers = [GradientOptimizer(self.config, self.known_constraints) for _ in range(num)]
        elif self.optimizer_type == 'genetic':
            local_optimizers = None
        else:
            GryffinUnknownSettingsError(f'Did not understand optimizer choice {self.optimizer_type}.'
                                        f'\n\tPlease choose "adam" or "genetic"')
        return local_optimizers

    def propose(self, best_params, kernel_contribution, probability_infeasible, frac_infeasible, sampling_param_values,
                num_samples=200, dominant_samples=None):
        """Highest-level method of this class that takes the BNN results, builds the acquisition function, optimises
        it, and returns a number of possible parameter points. These will then be used by the SampleSelector to pick
        the parameters to suggest."""

        # define optimizers
        self.local_optimizers = self._load_optimizers(num=len(sampling_param_values))

        # -------------------------------------------------------------
        # register attributes we'll be using to compute the acquisition
        # -------------------------------------------------------------
        self.acquisition_functions = {}  # reinitialize acquisition functions, otherwise we keep using old ones!
        self.sampling_param_values = sampling_param_values
        self.frac_infeasible = frac_infeasible
        self.kernel_contribution = kernel_contribution
        self.probability_infeasible = probability_infeasible
        # -------------------------------------------------------------

        # get random samples
        random_proposals = self._propose_randomly(best_params, num_samples, dominant_samples=dominant_samples)

        # run acquisition optimization starting from random samples
        start = time.time()
        optimized_proposals = self._optimize_proposals(random_proposals, dominant_samples=dominant_samples)
        end = time.time()
        self.log('[TIME]:  ' + parse_time(start, end) + '  (optimizing proposals)', 'INFO')

        extended_proposals = np.array([random_proposals for _ in range(len(sampling_param_values))])
        combined_proposals = np.concatenate((extended_proposals, optimized_proposals), axis=1)

        return combined_proposals

    def eval_acquisition(self, x, batch_index):
        acquisition = self.acquisition_functions[batch_index]
        return acquisition(x)


class AcquisitionFunction:
    """Acquisition function class that is used to support the class Acquisition. It selects the function to
    be optimized given the situation. It avoids re-defining the same functions multiple times in Acquisition methods"""
    def __init__(self, kernel_contribution, probability_infeasible, sampling_param, frac_infeasible,
                 acq_min=0, acq_max=1, feas_sensitivity=1.0):

        self.kernel_contribution = kernel_contribution
        self.probability_infeasible = probability_infeasible
        self.sampling_param = sampling_param
        self.frac_infeasible = frac_infeasible
        self.acq_min = acq_min
        self.acq_max = acq_max

        # NOTE: splitting the acquisition function into multiple funcs for efficiency when priors == 0/1
        # select the relevant acquisition
        if self.frac_infeasible == 0:
            self.acquisition_function = self._acquisition_all_feasible
            self.feasibility_weight = None  # i.e. not used
        elif self.frac_infeasible == 1:
            self.acquisition_function = self._acquisition_all_infeasible
            self.feasibility_weight = None  # i.e. not used
        else:
            self.acquisition_function = self._acquisition_standard
            self.feasibility_weight = self.frac_infeasible ** feas_sensitivity

    def __call__(self, x):
        return self.acquisition_function(x)

    def _acquisition_standard(self, x):
        num, inv_den = self.kernel_contribution(x)  # standard acquisition for samples
        prob_infeas = self.probability_infeasible(x)  # feasibility acquisition
        acq_samp = (num + self.sampling_param) * inv_den
        # approximately normalize sample acquisition so it has same scale of prob_infeas
        acq_samp = (acq_samp - self.acq_min) / (self.acq_max - self.acq_min)
        return self.feasibility_weight * prob_infeas + (1. - self.feasibility_weight) * acq_samp

    # if all feasible, prob_infeas always zero, so no need to estimate feasibility
    def _acquisition_all_feasible(self, x):
        num, inv_den = self.kernel_contribution(x)  # standard acquisition for samples
        acq_samp = (num + self.sampling_param) * inv_den
        return acq_samp

    # if all infeasible, acquisition is flat, so no need to compute it
    def _acquisition_all_infeasible(self, x):
        prob_infeas = self.probability_infeasible(x)
        return prob_infeas


