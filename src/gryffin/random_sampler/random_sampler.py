#!/usr/bin/env python 

__author__ = 'Florian Hase'


import numpy as np
from gryffin.utilities import Logger
from gryffin.utilities import GryffinUnknownSettingsError, GryffinComputeError
from gryffin.observation_processor import param_vector_to_dict


class RandomSampler(Logger):

    def __init__(self, config, constraints=None):
        """
        known_constraints : list of callable
            List of constraint functions. Each is a function that takes a parameter dict, e.g.
            {'x0':0.1, 'x1':10, 'x2':'A'} and returns a bool indicating
            whether it is in the feasible region or not.
        """

        # register attributes
        self.config = config

        # if constraints not None, and not a list, put into a list
        if constraints is not None and isinstance(constraints, list) is False:
            self.constraints = [constraints]
        else:
            self.constraints = constraints

        # set verbosity
        verbosity = self.config.get('verbosity')
        if 'random_sampler' in self.config.general.verbosity:
            verbosity = self.config.general.verbosity['random_sampler']
        Logger.__init__(self, 'RandomSampler', verbosity)

    def draw(self, num=1):
        # if no constraints, we do not need to do any "rejection sampling"
        if self.constraints is None:
            samples = self._fast_draw(num=num)
        else:
            samples = self._slow_draw(num=num)
        return samples

    def perturb(self, ref_sample, num=1, scale=0.05):
        """Take ref_sample and perturb it num times"""
        # if no constraints, we do not need to do any "rejection sampling"
        if self.constraints is None:
            perturbed_samples = self._fast_perturb(ref_sample, num=num, scale=scale)
        else:
            perturbed_samples = self._slow_perturb(ref_sample, num=num, scale=scale)
        return perturbed_samples

    def _fast_draw(self, num=1):
        samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            param_type = param_settings['type']
            specs = param_settings['specifics']
            param_samples = self._draw_single_parameter(num=num, param_type=param_type, specs=specs)
            samples.append(param_samples)
        samples = np.concatenate(samples, axis=1)
        self.log('generated uniform samples: \n%s' % str(samples), 'DEBUG')
        return samples

    def _slow_draw(self, num=1):
        samples = []
        counter = 0

        # keep trying random samples until we get num samples
        while len(samples) < num:
            sample = []  # we store the random sample used by Gryffin here

            # iterate over each variable and draw at random
            for param_index, param_settings in enumerate(self.config.parameters):
                specs = param_settings['specifics']
                param_type = param_settings['type']
                param_sample = self._draw_single_parameter(num=1, param_type=param_type, specs=specs)[0]
                sample.append(param_sample[0])

            # evaluate whether the sample violates the known constraints
            param = param_vector_to_dict(param_vector=sample, param_names=self.config.param_names,
                                         param_options=self.config.param_options, param_types=self.config.param_types)

            feasible = [constr(param) for constr in self.constraints]
            if all(feasible) is True:
                samples.append(sample)

            counter += 1
            if counter > 100 * num:
                raise GryffinComputeError("the feasible region seems to be less than 1% of the optimization "
                                          "domain - consider redefining the problem")

        samples = np.array(samples)
        return samples

    def _draw_single_parameter(self, num, param_type, specs):
        if param_type == 'continuous':
            sampled_values = self._draw_continuous(low=specs['low'], high=specs['high'], size=(num, 1))
        elif param_type == 'categorical':
            sampled_values = self._draw_categorical(num_options=len(specs['options']), size=(num, 1))
        elif param_type == 'discrete':
            sampled_values = self._draw_discrete(low=specs['low'], high=specs['high'], size=(num, 1))
        else:
            GryffinUnknownSettingsError(f'cannot understand parameter type "{param_type}"')
        return sampled_values

    def _fast_perturb(self, ref_sample, num=1, scale=0.05):
        """Perturbs a reference sample by adding random uniform noise around it"""
        perturbed_samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            param_type = param_settings['type']
            specs = param_settings['specifics']
            ref_value = ref_sample[param_index]
            perturbed_param_samples = self._perturb_single_parameter(ref_value=ref_value, num=num,
                                                                     param_type=param_type,
                                                                     specs=specs, scale=scale)
            perturbed_samples.append(perturbed_param_samples)

        perturbed_samples = np.concatenate(perturbed_samples, axis=1)
        return perturbed_samples

    def _slow_perturb(self, ref_sample, num=1, scale=0.05):
        perturbed_samples = []
        counter = 0

        # keep trying random samples until we get num samples
        while len(perturbed_samples) < num:
            perturbed_sample = []  # we store the samples here

            # iterate over each variable and perturb ref_sample
            for param_index, param_settings in enumerate(self.config.parameters):
                specs = param_settings['specifics']
                param_type = param_settings['type']
                ref_value = ref_sample[param_index]
                perturbed_param = self._perturb_single_parameter(ref_value=ref_value, num=1, param_type=param_type,
                                                                 specs=specs, scale=scale)[0]
                perturbed_sample.append(perturbed_param[0])

            # evaluate whether the sample violates the known constraints
            param = param_vector_to_dict(param_vector=perturbed_sample, param_names=self.config.param_names,
                                         param_options=self.config.param_options, param_types=self.config.param_types)

            feasible = [constr(param) for constr in self.constraints]
            if all(feasible) is True:
                perturbed_samples.append(perturbed_sample)

            counter += 1
            if counter > 100 * num:
                raise GryffinComputeError("we cannot find feasible solutions to perturbations of the incumbent - "
                                          "something is badly wrong with either the setup or the code")

        perturbed_samples = np.array(perturbed_samples)
        return perturbed_samples

    def _perturb_single_parameter(self, ref_value, num, param_type, specs, scale):
        if param_type == 'continuous':
            # draw uniform within unit range
            sampled_values = self._draw_continuous(-scale, scale, (num, 1))
            # scale to actual range
            sampled_values *= specs['high'] - specs['low']
            # add +/- 5% perturbation to sample
            perturbed_sample = ref_value + sampled_values
            # make sure we do not cross optimization boundaries
            perturbed_sample = np.where(perturbed_sample < specs['low'], specs['low'], perturbed_sample)
            perturbed_sample = np.where(perturbed_sample > specs['high'], specs['high'], perturbed_sample)
        elif param_type in ['categorical', 'discrete']:
            # i.e. not perturbing these variables
            perturbed_sample = ref_value * np.ones((num, 1)).astype(np.float32)
        else:
            GryffinUnknownSettingsError('did not understand settings')
        return perturbed_sample

    @staticmethod
    def _draw_categorical(num_options, size):
        return np.random.choice(num_options, size=size).astype(np.float32)

    @staticmethod
    def _draw_continuous(low, high, size):
        return np.random.uniform(low=low, high=high, size=size).astype(np.float32)

    @staticmethod
    def _draw_discrete(low, high, size):
        return np.random.randint(low=0, high=high - low + 1, size=size).astype(np.float32)
