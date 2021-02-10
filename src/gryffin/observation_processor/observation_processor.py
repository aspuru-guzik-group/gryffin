#!/usr/bin/env python

__author__ = 'Florian Hase'

import numpy as np

from chimera import Chimera
from gryffin.utilities import Logger
from gryffin.utilities import GryffinUnknownSettingsError


class ObservationProcessor(Logger):

    def __init__(self, config):
        self.config = config
        self.chimera = Chimera(tolerances=self.config.obj_tolerances,
                               absolutes=self.config.obj_absolutes,
                               goals=self.config.obj_goals,
                               softness=self.config.get('softness'))
        Logger.__init__(self, 'ObservationProcessor', verbosity=self.config.get('verbosity'))

        # compute some boundaries
        self.feature_lowers = self.config.feature_lowers
        self.feature_uppers = self.config.feature_uppers
        self.soft_lower     = self.feature_lowers + 0.1 * (self.feature_uppers - self.feature_lowers)
        self.soft_upper     = self.feature_uppers - 0.1 * (self.feature_uppers - self.feature_lowers)

    def mirror_parameters(self, param_vector):
        # get indices
        lower_indices_prelim = np.where(param_vector < self.soft_lower)[0]
        upper_indices_prelim = np.where(param_vector > self.soft_upper)[0]

        lower_indices, upper_indices = [], []
        for feature_index, feature_type in enumerate(self.config.feature_types):
            if feature_type != 'continuous': continue
            if feature_index in lower_indices_prelim:
                lower_indices.append(feature_index)
            if feature_index in upper_indices_prelim:
                upper_indices.append(feature_index)

        index_dict    = {index: 'lower' for index in lower_indices}
        for index in upper_indices:
            index_dict[index] = 'upper'

        # mirror param
        params = []
        index_dict_keys, index_dict_values = list(index_dict.keys()), list(index_dict.values())
        for index in range(2**len(index_dict)):
            param_copy = param_vector.copy()
            for jndex in range(len(index_dict)):
                if (index // 2**jndex) % 2 == 1:
                    param_index = index_dict_keys[jndex]
                    if index_dict_values[jndex] == 'lower':
                        param_copy[param_index] = self.feature_lowers[param_index] - (param_vector[param_index] - self.feature_lowers[param_index])
                    elif index_dict_values[jndex] == 'upper':
                        param_copy[param_index] = self.feature_uppers[param_index] + (self.feature_uppers[param_index] - param_vector[param_index])
            params.append(param_copy)
        if len(params) == 0:
            params.append(param_vector.copy())
        return params

    def scalarize_objectives(self, objs):
        # MA: this is technically not needed because Chimera should already normalize its output
        #  we still keep it however to really make sure the objectives are normalized, and because
        #  Gryffin takes the sqrt of the scalarized objective and I do not want to affect the behavior,
        #  although I don't know the reason of the sqrt
        scalarized = self.chimera.scalarize(objs)
        min_obj, max_obj = np.amin(scalarized), np.amax(scalarized)
        if min_obj != max_obj:
            scaled_obj = (scalarized - min_obj) / (max_obj - min_obj)
            scaled_obj = np.sqrt(scaled_obj)
        else:
            scaled_obj = scalarized - min_obj
        return scaled_obj

    def process(self, obs_dicts):

        param_names   = self.config.param_names
        param_options = self.config.param_options
        param_types   = self.config.param_types
        mirror_mask_kwn = []
        mirror_mask_ukwn = []

        # get raw results
        raw_params_kwn = []  # known result = feasible
        raw_objs_kwn = []
        raw_params_ukwn = []  # unknown result = unfeasible
        raw_objs_ukwn = []

        for obs_dict in obs_dicts:

            # get param-vector
            param_vector = []
            for param_index, param_name in enumerate(param_names):

                param_type = param_types[param_index]
                if param_type == 'continuous':
                    obs_param = obs_dict[param_name]
                elif param_type == 'categorical':
                    obs_param = np.array([param_options[param_index].index(element) for element in obs_dict[param_name]])
                elif param_type == 'discrete':
                    obs_param = np.array([list(param_options[param_index]).index(element) for element in obs_dict[param_name]])
                param_vector.extend(obs_param)

            mirrored_params = self.mirror_parameters(param_vector)

            # get obj-vector
            obj_vector = np.array([obs_dict[obj_name] for obj_name in self.config.obj_names])

            # --------------------
            # add processed params
            # --------------------
            if any(np.isnan(obj_vector)) is False:
                # add to known if there is no nan (note: we expect all objs to either feasible or unfeasible)
                for i, param in enumerate(mirrored_params):
                    raw_params_kwn.append(param)
                    raw_objs_kwn.append(obj_vector)

                    # add feasibility info to ukwn lists (i.e. all feasible)
                    raw_params_ukwn.append(param)
                    raw_objs_ukwn.append([0.])  # single objective

                    # keep track of mirrored params
                    if i == 0:
                        mirror_mask_kwn.append(True)
                        mirror_mask_ukwn.append(True)
                    else:
                        mirror_mask_kwn.append(False)
                        mirror_mask_ukwn.append(False)

            # if we have nan ==> unfeasible, add only to ukwn list
            else:
                for i, param in enumerate(mirrored_params):
                    raw_params_ukwn.append(param)
                    raw_objs_ukwn.append([1.])  # single objective

                    # keep track of mirrored params
                    if i == 0:
                        mirror_mask_ukwn.append(True)
                    else:
                        mirror_mask_ukwn.append(False)

        # process standard params/objs
        # ----------------------------
        raw_objs_kwn = np.array(raw_objs_kwn)
        raw_params_kwn = np.array(raw_params_kwn)
        params_kwn = raw_params_kwn
        if raw_objs_kwn.shape[0] > 0:  # guard against empty objs, which can happen if first sample is nan
            # Note that Chimera takes care of adjusting the objectives based on whether we are
            #  minimizing vs maximizing
            objs_kwn = self.scalarize_objectives(raw_objs_kwn)
        else:
            objs_kwn = raw_objs_kwn

        # process feasibility space
        # -------------------------
        raw_objs_ukwn = np.array(raw_objs_ukwn)
        raw_params_ukwn = np.array(raw_params_ukwn)
        params_ukwn = raw_params_ukwn
        # no need to adjust objs_ukwn: we assume we are minimizing and 0 = feasible, 1 = infeasible
        # we also do not scalarize because:
        # (i) we consider all objectives to be known/unknown at the same time
        # (ii) the threshold are defined for the objectives, not for the feasibility. If feasibility
        #  itself is an objective, it should be defined as such in the config.
        objs_ukwn = raw_objs_ukwn.flatten()

        return params_kwn, objs_kwn, mirror_mask_kwn, params_ukwn, objs_ukwn, mirror_mask_ukwn


