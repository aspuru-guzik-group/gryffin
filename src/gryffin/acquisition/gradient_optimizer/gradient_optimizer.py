#!/usr/bin/env python 

__author__ = 'Florian Hase'


import numpy as np
from gryffin.utilities import Logger, GryffinUnknownSettingsError
from gryffin.observation_processor import param_vector_to_dict
from . import AdamOptimizer, NaiveDiscreteOptimizer, NaiveCategoricalOptimizer


class GradientOptimizer(Logger):

    def __init__(self, config, known_constraints=None):
        self.config = config
        self.known_constraints = known_constraints
        Logger.__init__(self, 'GradientOptimizer', verbosity=self.config.get('verbosity'))

        # parse positions
        self.pos_continuous = np.full(self.config.num_features, False, dtype=bool)
        self.pos_categories = np.full(self.config.num_features, False, dtype=bool)
        self.pos_discrete   = np.full(self.config.num_features, False, dtype=bool)
        for feature_index, feature_type in enumerate(self.config.feature_types):
            if feature_type == 'continuous':
                self.pos_continuous[feature_index] = True
            elif feature_type == 'categorical':
                self.pos_categories[feature_index] = True
            elif feature_type == 'discrete':
                self.pos_discrete[feature_index] = True
            else:
                feature_name = self.config.feature_names[feature_index]
                GryffinUnknownSettingsError('did not understand parameter type "%s" for parameter "%s".\n\t(%s) Please choose from "continuous" or "categorical"' % (feature_type, feature_name, self.template))

        # instantiate optimizers for all variable types
        self.opt_con = AdamOptimizer()
        self.opt_dis = NaiveDiscreteOptimizer()
        self.opt_cat = NaiveCategoricalOptimizer()

    def _within_bounds(self, sample):
        return not (np.any(sample < self.config.feature_lowers) or np.any(sample > self.config.feature_uppers))

    def _optimize_continuous(self, sample):
        proposal = self.opt_con.get_update(sample)
        if self._within_bounds(proposal):
            return proposal
        else:
            return sample

    def _optimize_discrete(self, sample):
        proposal = self.opt_dis.get_update(sample)
        return proposal

    def _optimize_categorical(self, sample):
        proposal = self.opt_cat.get_update(sample)
        return proposal

    def set_func(self, kernel, ignores=None):
        pos_continuous = self.pos_continuous.copy()
        pos_discrete   = self.pos_discrete.copy()
        pos_categories = self.pos_categories.copy()
        if ignores is not None:
            for ignore_index, ignore in enumerate(ignores):
                if ignore:
                    pos_continuous[ignore_index] = False
                    pos_discrete[ignore_index]   = False
                    pos_categories[ignore_index] = False

        self.opt_con.set_func(kernel, pos=np.arange(self.config.num_features)[pos_continuous])
        self.opt_dis.set_func(kernel, pos=np.arange(self.config.num_features)[pos_discrete],   highest=self.config.feature_sizes[self.pos_discrete])
        self.opt_cat.set_func(kernel, pos=np.arange(self.config.num_features)[pos_categories], highest=self.config.feature_sizes[self.pos_categories])

    def optimize(self, sample, max_iter=10):
        # project sample onto opt boundaries if needed
        # ...though it should not be needed, as RandomSampler should do this already?
        if not self._within_bounds(sample):
            sample = np.where(sample < self.config.feature_lowers, self.config.feature_lowers, sample)
            sample = np.where(sample > self.config.feature_uppers, self.config.feature_uppers, sample)
            sample = sample.astype(np.float32)

        # if no constraints, we do not need to do any checks on feasibility
        if self.known_constraints is None:
            optimized = self._fast_optimize(sample=sample, max_iter=max_iter)
        else:
            optimized = self._slow_optimize(sample=sample, max_iter=max_iter)
        return optimized

    def _single_opt_iteration(self, optimized):
        # one step of continuous
        if np.any(self.pos_continuous):
            optimized = self._optimize_continuous(optimized)

        # one step of categorical perturbation
        if np.any(self.pos_categories):
            optimized = self._optimize_categorical(optimized)

        # one step of discrete optimization
        if np.any(self.pos_discrete):
            optimized = self._optimize_discrete(optimized)

        return optimized

    def _fast_optimize(self, sample, max_iter=10):

        # copy sample
        sample_copy = sample.copy()
        optimized = sample.copy()
        # optimize
        for num_iter in range(max_iter):
            # one step of optimization
            optimized = self._single_opt_iteration(optimized)
            # check for convergence
            if np.any(self.pos_continuous) and np.linalg.norm(sample_copy - optimized) < 1e-7:
                break
            else:
                sample_copy = optimized.copy()
        return optimized

    def _slow_optimize(self, sample, max_iter=10):

        # copy sample
        sample_copy = sample.copy()
        optimized = sample.copy()

        # --------
        # optimize
        # --------
        for num_iter in range(max_iter):
            # one step of optimization
            optimized = self._single_opt_iteration(optimized)

            # evaluate whether the optimized sample violates the known constraints
            param = param_vector_to_dict(param_vector=optimized, param_names=self.config.param_names,
                                         param_options=self.config.param_options, param_types=self.config.param_types)
            feasible = self.known_constraints(param)
            if feasible is False:
                # stop optimization and return last feasible point
                optimized = sample_copy.copy()
                break

            # check for convergence
            if np.any(self.pos_continuous) and np.linalg.norm(sample_copy - optimized) < 1e-7:
                break
            else:
                sample_copy = optimized.copy()

        return optimized

