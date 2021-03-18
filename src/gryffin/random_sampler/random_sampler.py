#!/usr/bin/env python 

__author__ = 'Florian Hase'


import numpy as np
from gryffin.utilities import Logger
from gryffin.utilities import GryffinUnknownSettingsError, sample_arrays_to_dicts


class RandomSampler(Logger):

    def __init__(self, config, known_constraints=None):
        """
        known_constraints : callable
            A function that takes a parameter dict, e.g. {'x0':0.1, 'x1':10, 'x2':'A'} and returns a bool indicating
            whether it is in the feasible region or not.
        """

        # register attributes
        self.config = config
        self.known_constraints = known_constraints

        # set verbosity
        verbosity = self.config.get('verbosity')
        if 'random_sampler' in self.config.general.verbosity:
            verbosity = self.config.general.verbosity['random_sampler']
        Logger.__init__(self, 'RandomSampler', verbosity)

    def draw(self, num=1):
        # if no constraints, we do not need to do any "rejection sampling"
        if self.known_constraints is None:
            samples = self._fast_draw(num=num)
        else:
            samples = self._slow_draw(num=num)
        return samples

    def _slow_draw(self, num=1):
        samples = []

        # keep trying random samples until we get num samples
        while len(samples) < num:
            sample = []  # we store the random sample used by Gryffin here

            # iterate over each variable and draw at random
            for param_index, param_settings in enumerate(self.config.parameters):
                specs = param_settings['specifics']
                param_type = param_settings['type']

                if param_type == 'continuous':
                    sample_array = self._draw_continuous(low=specs['low'], high=specs['high'], size=(1,))
                elif param_type == 'categorical':
                    sample_array = self._draw_categorical(num_options=len(specs['options']), size=(1,))
                elif param_type == 'discrete':
                    sample_array = self._draw_discrete(low=specs['low'], high=specs['high'], size=(1,))
                else:
                    GryffinUnknownSettingsError(f'cannot understand parameter type "{param_type}"')

                sample.append(sample_array[0])

            # evaluate whether the sample violates the known constraints
            param = sample_arrays_to_dicts(samples=sample, param_names=self.config.param_names,
                                           param_options=self.config.param_options, param_types=self.config.param_types)
            feasible = self.known_constraints(param)
            if feasible is True:
                samples.append(sample)

        samples = np.array(samples)
        return samples

    def _fast_draw(self, num=1):
        samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            param_type = param_settings['type']
            specs = param_settings['specifics']
            if param_type == 'continuous':
                sampled_values = self._draw_continuous(low=specs['low'], high=specs['high'], size=(num, 1))
            elif param_type == 'categorical':
                sampled_values = self._draw_categorical(num_options=len(specs['options']), size=(num, 1))
            elif param_type == 'discrete':
                sampled_values = self._draw_discrete(low=specs['low'], high=specs['high'], size=(num, 1))
            else:
                GryffinUnknownSettingsError(f'cannot understand parameter type "{param_type}"')
            samples.append(sampled_values)
        samples = np.concatenate(samples, axis=1)
        self.log('generated uniform samples: \n%s' % str(samples), 'DEBUG')
        return samples

    def perturb(self, pos, num=1, scale=0.05):
        samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            specs = param_settings['specifics']
            if param_settings['type'] == 'continuous':
                sampled_values  = self.continuous_sampler.draw(-scale, scale, (num, 1))
                sampled_values *= specs['high'] - specs['low']
                close_samples   = pos[param_index] + sampled_values
                close_samples   = np.where(close_samples < specs['low'],  specs['low'],  close_samples)
                close_samples   = np.where(close_samples > specs['high'], specs['high'], close_samples)
                samples.append(close_samples)
            elif param_settings['type'] in ['categorical', 'discrete']:
                sampled_values = pos[param_index] * np.ones((num, 1)).astype(np.float32)
                samples.append(sampled_values)
            else:
                GryffinUnknownSettingsError('did not understand settings')
        samples = np.concatenate(samples, axis = 1)
        return samples

    def normal_samples(self, loc=0., scale=1., num=1):
        samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            param_type = param_settings['type']
            specs = param_settings['specifics']
            if param_type == 'continuous':
                param_range = specs['high'] - specs['low']
                sampled_values = np.random.normal(0., scale * param_range, (num, 1)) + loc[param_index]
            elif param_type == 'categorical':
                sampled_values = self.categorical_sampler.draw(len(specs['options']), (num, 1))
            elif param_type == 'discrete':
                sampled_values = self.discrete_sampler.draw(specs['low'], specs['high'], (num, 1))
            else:
                GryffinUnknownSettingsError(f'cannot understand parameter type "{param_type}"')
            samples.append(sampled_values)
        samples = np.concatenate(samples, axis=1)
        return samples

    @staticmethod
    def _draw_categorical(num_options, size):
        return np.random.choice(num_options, size=size).astype(np.float32)

    @staticmethod
    def _draw_continuous(low, high, size):
        return np.random.uniform(low=low, high=high, size=size).astype(np.float32)

    @staticmethod
    def _draw_discrete(low, high, size):
        return np.random.randint(low=0, high=high - low, size=size).astype(np.float32)
