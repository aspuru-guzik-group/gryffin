#!/usr/bin/env python

import numpy as np

#===============================================================================

from gryffin.utilities import Transformation
from gryffin.utilities import Logger

#===============================================================================

class TrainingSetGenerator(Logger):
    def __init__(self, config):
        self.config = config
        Logger.__init__(self, 'TrainingSetGenerator', verbosity=self.config.get('verbosity'))


    def set_descriptors(self, descriptors):
        if not hasattr(self, 'descriptors'):
            setattr(self, 'descriptors', descriptors)
        else:
            pass

    def has_descriptors(self):
        return hasattr(self, 'descriptors')


    def get_arrays(self, observations):
        ''' Generate arrays of features and targets from parameters of observations
        '''
        params = []
        objs   = []
        for obs in observations:
            p = [obs[name][0] for name in self.config.param_names]
            o = [obs[name] for name in self.config.obj_names]
            params.append(p)
            objs.append(o)

        params = np.array(params)
        objs   = np.array(objs)

        return params, objs

    def construct_context_set(self, obs_params_kwn, obs_objs_kwn):
        ''' Construct context set for a predictive model

        Args:
            observations (list): list of the observations of the objective
                                being optimized
            proxy_observations (list): list of auxilliary observations

        Returns:
            context_x (np.ndarray):
            context_y (np.ndarray):
        '''
        # reshape the targets if needed
        if len(obs_objs_kwn.shape) == 1:
            obs_objs_kwn = obs_objs_kwn.reshape(-1,  1)
        # generate the features
        context_x = self.construct_features(obs_params_kwn)
        # rename the targets
        context_y = obs_objs_kwn

        return {'context_x': context_x, 'context_y': context_y}

    def construct_training_set(self, obs_params_kwn, obs_objs_kwn,
                               proxy_obs_params_kwn, proxy_obs_objs_kwn):
        ''' Contruct training set for predictive model

        Args:
            observations (list): list of the observations of the objective
                                being optimized
            proxy_observations (list): list of auxilliary observations

        Returns:
            features (np.ndarray):
            proxy_features (np.ndarray):
            targets (np.ndarray):
            proxy_targets (np.ndarray)

        '''
        # reshape targets if needed
        if len(obs_objs_kwn.shape) == 1 and len(proxy_obs_objs_kwn.shape) == 1:
            obs_objs_kwn = obs_objs_kwn.reshape(-1, 1)
            proxy_obs_objs_kwn = proxy_obs_objs_kwn.reshape(-1, 1)
        # generate the features
        train_features = self.construct_features(obs_params_kwn)
        proxy_train_features = self.construct_features(proxy_obs_params_kwn)
        # rename the objecitves as targets
        train_targets       = obs_objs_kwn
        proxy_train_targets = proxy_obs_objs_kwn
        # check if the shapes of the features and targets match
        if train_features.shape[1] != proxy_train_features.shape[1]:
            self.log('Proxy observations features shape does not match', '[FATAL]')
        elif train_targets.shape[1] != proxy_train_targets.shape[1]:
            self.log('Proxy observations targets shape does not match', '[FATAL]')

        return {'train_features': train_features, 'train_targets': train_targets,
                'proxy_train_features': proxy_train_features, 'proxy_train_targets': proxy_train_targets}


    def construct_features(self, params_kwn):
            ''' Generate a feature vector from a list of parameters
            Args:
                params_kwn (np.ndarray): the known parameter values

            Returns:
                features (np.ndarray): input features for the predictive model
            '''
            # reshape if needed --> evaluating acquisition or selector
            if len(params_kwn.shape)==1:
                params_kwn = params_kwn.reshape((1, len(params_kwn)))
            features = []
            for example_ix, example in enumerate(params_kwn):
                vec = []
                for param_ix, (param_type, param_opts, param_desc) in enumerate(zip(self.config.param_types, self.config.param_options, self.descriptors)):
                    if param_type in ['continuous', 'discrete']:
                        # add continuous/discrete parameter
                        # TODO: perform transformation if needed
                        vec.extend([example[param_ix]])
                    elif param_type == 'categorical':
                        if isinstance(param_desc, type(None)):
                            # naive formulation - make one-hot-encoded vector
                            one_hot = np.zeros(len(param_opts), dtype=float)
                            one_hot[example[param_ix]] = 1.
                            vec.extend(one_hot)
                        else:
                            # descriptors provided - add categorical descriptor
                            vec.extend(param_desc[int(example[param_ix]), :])
                features.append(np.array(vec))
            features = np.array(features)

            return features


class PredictiveModel(Logger):
    ''' Built-in predictive model in Gryffin

    Currently supported:
        - Gemini
        -

    Args:
        config (dict): Gryffin configuration
        pred_model_config (dict): additional hyperparameters for the predictive
                                  model in key:value format
    '''

    def __init__(self, config, pred_model_config={}):

        Logger.__init__(self, 'PredictiveModel', verbosity=0)
        self.config = config
        self.kind = self.config.get_pred('model_kind')
        self.pred_model_config = pred_model_config
        self.is_trained = False
        if self.kind is not None:
            self.log(f'Initializing predictive model {self.kind}', 'INFO')
            # initialize model here
            self._initialize_model()
            self.transformation = Transformation(self.config.get_pred('transformation'))
            setattr(self, 'is_internal', True)
        else:
            setattr(self, 'is_internal', False)


    def _initialize_model(self):
        ''' Initialize model and set training regime
        '''
        if self.kind == 'gemini':
            from gemini import GeminiOpt
            self.model = GeminiOpt()
        else:
            raise NotImplementedError


    def train(self, training_set):
        ''' Train the built in predictive model
        '''
        self.model.train(training_set['train_features'], training_set['train_targets'],
                         training_set['proxy_train_features'], training_set['proxy_train_targets'],
                         num_folds=3)

        # set is trained
        self.is_trained = True


    def predict(self, features):
        ''' Make predictions
        '''
        mean_pred = self.model.predict(features)
        return mean_pred


    def get_pearson_coeff(self):
        return self.model.get_pearson_coeff()
