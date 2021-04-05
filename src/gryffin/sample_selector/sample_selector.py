#!/usr/bin/env python 

__author__ = 'Florian Hase'

import numpy as np
import multiprocessing
from multiprocessing import Manager, Process
from gryffin.utilities import Logger, parse_time
import time


class SampleSelector(Logger):

    def __init__(self, config):
        self.config = config
        self.verbosity = self.config.get('verbosity')
        Logger.__init__(self, 'SampleSelector', verbosity=self.verbosity)
        # figure out how many CPUs to use
        if self.config.get('num_cpus') == 'all':
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get('num_cpus'))

    @staticmethod
    def compute_exp_objs(proposals, eval_acquisition, batch_index, return_index=0, return_dict=None):
        # batch_index is the index of the sampling_param_values used
        samples = proposals[batch_index]
        exp_objs = np.empty(len(samples))

        for sample_index, sample in enumerate(samples):
            acq = eval_acquisition(sample, batch_index)  # this is a method of the Acquisition instance
            exp_objs[sample_index] = np.exp(-acq)

        if return_dict.__class__.__name__ == 'DictProxy':
            return_dict[return_index] = exp_objs

        return exp_objs

    def select(self, num_samples, proposals, eval_acquisition, sampling_param_values, obs_params):
        """
        num_samples : int
            number of samples to select per sampling strategy (i.e. the ``batches`` argument in the configuration)
        proposals : ndarray
            shape of proposals is (num strategies, num samples, num dimensions).
        """

        start = time.time()

        if self.verbosity > 2.5:  # i.e. INFO or DEBUG
            with self.console.status("Selecting best samples to recommend..."):
                samples = self._select(num_samples, proposals, eval_acquisition, sampling_param_values, obs_params)
        else:
            samples = self._select(num_samples, proposals, eval_acquisition, sampling_param_values, obs_params)

        end = time.time()
        time_string = parse_time(start, end)
        samples_str = 'samples' if len(samples) > 1 else 'sample'
        self.log(f'{len(samples)} {samples_str} selected in {time_string}', 'INFO')

        return samples

    def _select(self, num_samples, proposals, eval_acquisition, sampling_param_values, obs_params):

        num_obs = len(obs_params)
        feature_ranges = self.config.feature_ranges
        char_dists = feature_ranges / float(num_obs)**0.5

        # save all objective values here
        exp_objs = []

        # ---------------------------------
        # compute exp of acquisition values
        # ---------------------------------
        # TODO: this is slightly redundant as we have computed acquisition values already in Acquisition
        for batch_index, sampling_param in enumerate(sampling_param_values):

            # -------------------
            # parallel processing
            # -------------------
            if self.num_cpus > 1:
                return_dict = Manager().dict()

                # split proposals into approx equal chunks based on how many CPUs we're using
                proposals_splits = np.array_split(proposals, self.num_cpus, axis=1)

                # parallelize over splits
                # -----------------------
                processes = []
                for idx, proposals_split in enumerate(proposals_splits):
                    process = Process(target=self.compute_exp_objs, args=(proposals_split, eval_acquisition,
                                                                          batch_index, idx, return_dict))
                    processes.append(process)
                    process.start()

                # wait until all processes finished
                for process in processes:
                    process.join()

                # sort results in return_dict to create batch_exp_objs list with correct sample order
                batch_exp_objs = []
                for idx in range(len(proposals_splits)):
                    batch_exp_objs.extend(return_dict[idx])

            # ---------------------
            # sequential processing
            # ---------------------
            else:
                batch_exp_objs = self.compute_exp_objs(proposals=proposals, eval_acquisition=eval_acquisition,
                                                       batch_index=batch_index, return_index=0, return_dict=None)

            # append the proposed samples for this sampling strategy to the global list of samples
            exp_objs.append(batch_exp_objs)

        # cast to np.array
        exp_objs = np.array(exp_objs)

        # ----------------------------------------
        # compute prior recommendation punishments
        # ----------------------------------------
        for batch_index in range(len(sampling_param_values)):
            batch_proposals = proposals[batch_index, : exp_objs.shape[1]]

            # compute distance to each obs_param
            distances = [np.sum((obs_params - batch_proposal)**2, axis=1) for batch_proposal in batch_proposals]
            distances = np.array(distances)
            min_distances = np.amin(distances, axis=1)
            ident_indices = np.where(min_distances < 1e-8)[0]

            exp_objs[batch_index, ident_indices] = 0.

        # ---------------
        # collect samples
        # ---------------
        samples = []
        for sample_index in range(num_samples):
            new_samples = []

            for batch_index in range(len(sampling_param_values)):
                batch_proposals = proposals[batch_index]

                # compute diversity punishments
                div_crits = np.ones(exp_objs.shape[1])

                for proposal_index, proposal in enumerate(batch_proposals[:exp_objs.shape[1]]):
                    obs_min_distance = np.amin([np.abs(proposal - x) for x in obs_params], axis=0)
                    if len(new_samples) > 0:
                        min_distance = np.amin([np.abs(proposal - x) for x in new_samples], axis=0)
                        min_distance = np.minimum(min_distance, obs_min_distance)
                    else:
                        min_distance = obs_min_distance

                    div_crits[proposal_index] = np.minimum(1., np.mean(np.exp(2. * (min_distance - char_dists) / feature_ranges)))

                # reweight rewards
                reweighted_rewards = exp_objs[batch_index] * div_crits
                largest_reward_index = np.argmax(reweighted_rewards)

                new_sample = batch_proposals[largest_reward_index]
                new_samples.append(new_sample)

                # update reward of selected sample
                exp_objs[batch_index, largest_reward_index] = 0.

            samples.append(new_samples)
        samples = np.concatenate(samples)

        return samples






