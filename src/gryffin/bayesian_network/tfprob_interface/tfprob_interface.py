#!/usr/bin/env python

__author__ = 'Florian Hase, Matteo Aldeghi'

import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow_probability import distributions as tfd

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .numpy_graph import NumpyGraph
from gryffin.utilities.decorators import processify
from gryffin.utilities import Logger, GryffinUnknownSettingsError, GryffinComputeError


class TfprobNetwork(Logger):

    def __init__(self, config, model_details, frac_feas):
        self.config = config
        self.frac_feas = frac_feas
        self.numpy_graph = NumpyGraph(self.config, model_details)

        # set logger verbosity
        Logger.__init__(self, 'TfprobNetwork', verbosity=self.config.get('verbosity'))

        # model hyperparams
        self.num_epochs = model_details['num_epochs']
        self.learning_rate = model_details['learning_rate']
        self.num_draws = model_details['num_draws']
        self.num_layers = model_details['num_layers']
        self.hidden_shape = model_details['hidden_shape']
        self.weight_loc = model_details['weight_loc']
        self.weight_scale = model_details['weight_scale']
        self.bias_loc = model_details['bias_loc']
        self.bias_scale = model_details['bias_scale']

        self.feature_size = len(self.config.kernel_names)
        self.bnn_output_size = len(self.config.kernel_names)
        self.target_size = len(self.config.kernel_names)

        self.trace = {}

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.InteractiveSession()

    def declare_training_data(self, obs_params):

        # categorical features are encoded as single numbers, bnn output is one-hot encoded
        self.num_obs = len(obs_params)

        # initialize training features and targets
        self.features = np.zeros((self.num_obs, self.feature_size))
        self.targets = np.zeros((self.num_obs, self.target_size))

        # construct training features
        feature_begin = 0
        feature_sizes = self.config.feature_sizes
        for feature_index, feature_type in enumerate(self.config.feature_types):
            feature_size = feature_sizes[feature_index]
            if feature_type == 'categorical':
                for obs_param_index, obs_param in enumerate(obs_params):
                    self.features[obs_param_index, feature_begin + int(obs_param[feature_index])] += 1
            elif feature_type == 'discrete':
                for obs_param_index, obs_param in enumerate(obs_params):
                    self.features[obs_param_index, feature_begin + int(obs_param[feature_index])] += 1
            elif feature_type == 'continuous':
                self.features[:, feature_begin] = obs_params[:, feature_index]
            else:
                raise NotImplementedError
            feature_begin += feature_size
        self.targets = self.features.copy()

        # rescale features
        lower_rescalings = np.empty(self.feature_size)
        upper_rescalings = np.empty(self.feature_size)
        kernel_uppers, kernel_lowers = self.config.kernel_uppers, self.config.kernel_lowers
        for kernel_index, kernel_name in enumerate(self.config.kernel_names):
            low = kernel_lowers[kernel_index]
            up  = kernel_uppers[kernel_index]
            lower_rescalings[kernel_index] = low  # - 0.1 * (up - low)
            upper_rescalings[kernel_index] = up   # + 0.1 * (up - low)

        self.lower_rescalings = lower_rescalings
        self.upper_rescalings = upper_rescalings

        self.rescaled_features = (self.features - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings)
        self.rescaled_targets = (self.targets - self.lower_rescalings) / (self.upper_rescalings - self.lower_rescalings)

        self.numpy_graph.declare_training_data(self.rescaled_features)

    def construct_model(self, learning_rate=None):

        if learning_rate is None:
            learning_rate = self.learning_rate

        with self.graph.as_default():

            self.sess.close()
            self.sess = tf.compat.v1.InteractiveSession()
            self.sess.as_default()

            self.x = tf.convert_to_tensor(self.rescaled_features, dtype=tf.float32)
            self.y = tf.convert_to_tensor(self.targets, dtype=tf.float32)

            # construct precisness
            self.tau_rescaling = np.zeros((self.num_obs, self.bnn_output_size))
            kernel_ranges      = self.config.kernel_ranges
            for obs_index in range(self.num_obs):
                self.tau_rescaling[obs_index] += kernel_ranges
            self.tau_rescaling = self.tau_rescaling**2

            # construct weight and bias shapes
            activations = [tf.nn.tanh]
            weight_shapes, bias_shapes = [[self.feature_size, self.hidden_shape]], [[self.hidden_shape]]
            for _ in range(1, self.num_layers - 1):
                activations.append(tf.nn.tanh)
                weight_shapes.append([self.hidden_shape, self.hidden_shape])
                bias_shapes.append([self.hidden_shape])
            activations.append(lambda x: x)
            weight_shapes.append([self.hidden_shape, self.bnn_output_size])
            bias_shapes.append([self.bnn_output_size])

            # ---------------
            # construct prior
            # ---------------
            self.prior_layer_outputs = [self.x]
            self.priors = {}
            for layer_index in range(self.num_layers):
                weight_shape, bias_shape = weight_shapes[layer_index], bias_shapes[layer_index]
                activation = activations[layer_index]

                weight = tfd.Normal(loc=tf.zeros(weight_shape) + self.weight_loc, scale=tf.zeros(weight_shape) + self.weight_scale)
                bias = tfd.Normal(loc=tf.zeros(bias_shape) + self.bias_loc, scale=tf.zeros(bias_shape) + self.bias_scale)
                self.priors['weight_%d' % layer_index] = weight
                self.priors['bias_%d' % layer_index] = bias

                prior_layer_output = activation(tf.matmul(self.prior_layer_outputs[-1], weight.sample()) + bias.sample())
                self.prior_layer_outputs.append(prior_layer_output)

            self.prior_bnn_output = self.prior_layer_outputs[-1]
            # draw precisions from gamma distribution
            self.prior_tau_normed = tfd.Gamma(
                            (self.num_obs/self.frac_feas)**2 + tf.zeros((self.num_obs, self.bnn_output_size)),
                            tf.ones((self.num_obs, self.bnn_output_size)),
                        )
            self.prior_tau        = self.prior_tau_normed.sample() / self.tau_rescaling
            self.prior_scale      = tfd.Deterministic(1. / tf.sqrt(self.prior_tau))

            # -------------------
            # construct posterior
            # -------------------
            self.post_layer_outputs = [self.x]
            self.posteriors = {}
            for layer_index in range(self.num_layers):
                weight_shape, bias_shape = weight_shapes[layer_index], bias_shapes[layer_index]
                activation = activations[layer_index]

                weight = tfd.Normal(loc=tf.Variable(tf.random.normal(weight_shape)), scale=tf.nn.softplus(tf.Variable(tf.zeros(weight_shape))))
                bias = tfd.Normal(loc=tf.Variable(tf.random.normal(bias_shape)), scale=tf.nn.softplus(tf.Variable(tf.zeros(bias_shape))))

                self.posteriors['weight_%d' % layer_index] = weight
                self.posteriors['bias_%d' % layer_index] = bias

                post_layer_output = activation(tf.matmul(self.post_layer_outputs[-1], weight.sample()) + bias.sample())
                self.post_layer_outputs.append(post_layer_output)

            self.post_bnn_output = self.post_layer_outputs[-1]
            self.post_tau_normed = tfd.Gamma(
                                (self.num_obs/self.frac_feas)**2+ tf.Variable(tf.zeros((self.num_obs, self.bnn_output_size))),
                                tf.nn.softplus(tf.Variable(tf.ones((self.num_obs, self.bnn_output_size)))),
                            )
            self.post_tau        = self.post_tau_normed.sample() / self.tau_rescaling
            self.post_sqrt_tau   = tf.sqrt(self.post_tau)
            self.post_scale	     = tfd.Deterministic(1. / self.post_sqrt_tau)

            # map bnn output to prediction
            post_kernels = {}
            targets_dict = {}
            inferences = []

            target_element_index = 0
            kernel_element_index = 0

            while kernel_element_index < len(self.config.kernel_names):

                kernel_type = self.config.kernel_types[kernel_element_index]
                kernel_size = self.config.kernel_sizes[kernel_element_index]

                feature_begin, feature_end = target_element_index, target_element_index + 1
                kernel_begin, kernel_end   = kernel_element_index, kernel_element_index + kernel_size

                prior_relevant = self.prior_bnn_output[:, kernel_begin: kernel_end]
                post_relevant  = self.post_bnn_output[:,  kernel_begin: kernel_end]

                if kernel_type == 'continuous':

                    target = self.y[:, kernel_begin: kernel_end]
                    lowers, uppers = self.config.kernel_lowers[kernel_begin: kernel_end], self.config.kernel_uppers[kernel_begin : kernel_end]

                    prior_support = (uppers - lowers) * (1.2 * tf.nn.sigmoid(prior_relevant) - 0.1) + lowers
                    post_support = (uppers - lowers) * (1.2 * tf.nn.sigmoid(post_relevant) - 0.1) + lowers

                    prior_predict = tfd.Normal(prior_support, self.prior_scale[:, kernel_begin: kernel_end].sample())
                    post_predict = tfd.Normal(post_support,  self.post_scale[:,  kernel_begin: kernel_end].sample())

                    targets_dict[prior_predict] = target
                    post_kernels['param_%d' % target_element_index] = {
                        'loc':       tfd.Deterministic(post_support),
                        'sqrt_prec': tfd.Deterministic(self.post_sqrt_tau[:, kernel_begin: kernel_end]),
                        'scale':     tfd.Deterministic(self.post_scale[:, kernel_begin: kernel_end].sample())}

                    inference = {'pred': post_predict, 'target': target}
                    inferences.append(inference)

                elif kernel_type in ['categorical', 'discrete']:
                    target = tf.cast(self.y[:, kernel_begin: kernel_end], tf.int32)

                    prior_temperature = 0.5 + 10.0 / self.num_obs
                    #prior_temperature = 1.0
                    post_temperature = prior_temperature

                    prior_support = prior_relevant
                    post_support = post_relevant

                    prior_predict_relaxed = tfd.RelaxedOneHotCategorical(prior_temperature, prior_support)
                    prior_predict = tfd.OneHotCategorical(probs=prior_predict_relaxed.sample())

                    post_predict_relaxed = tfd.RelaxedOneHotCategorical(post_temperature, post_support)
                    post_predict = tfd.OneHotCategorical(probs=post_predict_relaxed.sample())

                    targets_dict[prior_predict] = target
                    post_kernels['param_%d' % target_element_index] = {'probs': post_predict_relaxed}

                    inference = {'pred': post_predict, 'target': target}
                    inferences.append(inference)

                    '''
                        Temperature annealing schedule:
                            - temperature of 100   yields 1e-2 deviation from uniform
                            - temperature of  10   yields 1e-1 deviation from uniform
                            - temperature of   1   yields *almost* perfect agreement with expectation
                            - temperature of   0.1 yields perfect agreement with expectation
                    '''

                else:
                    GryffinUnknownSettingsError(f'did not understand kernel type: {kernel_type}')

                target_element_index += 1
                kernel_element_index += kernel_size

            self.post_kernels = post_kernels
            self.targets_dict = targets_dict

            self.loss = 0.
            for inference in inferences:
                self.loss += - tf.reduce_sum(inference['pred'].log_prob(inference['target']))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            tf.compat.v1.global_variables_initializer().run()

    def sample(self, num_epochs=None, num_draws=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        if num_draws is None:
            num_draws = self.num_draws

        with self.graph.as_default():

            # run inference
            for num_epoch in range(num_epochs):
                self.sess.run(self.train_op)

            # sample posterior
            posterior_samples = {}
            for key, kernel_parent in self.posteriors.items():
                parent_samples = kernel_parent.sample(num_draws).eval()
                posterior_samples[key] = parent_samples

            # sample the posterior of the Gamma distribution
            gamma_posterior_samples = self.post_tau_normed.sample(num_draws).eval()
            posterior_samples['gamma'] = gamma_posterior_samples


            post_kernels = self.numpy_graph.compute_kernels(posterior_samples)

            self.trace = {}
            for key in post_kernels.keys():
                self.trace[key] = {}
                kernel_dict = post_kernels[key]
                for kernel_name, kernel_values in kernel_dict.items():
                    self.trace[key][kernel_name] = kernel_values


    def get_kernels(self):
        trace_kernels = {'locs': [], 'sqrt_precs': [], 'probs': []}
        for param_index in range(len(self.config.param_names)):
            post_kernel = self.trace['param_%d' % param_index]

            # ------------------
            # continuous kernels
            # ------------------
            if 'loc' in post_kernel and 'sqrt_prec' in post_kernel:
                trace_kernels['locs'].append(post_kernel['loc'].astype(np.float64))
                trace_kernels['sqrt_precs'].append(post_kernel['sqrt_prec'].astype(np.float64))
                # for continuous variables, key "probs" contains all zeros
                trace_kernels['probs'].append(np.zeros(post_kernel['loc'].shape, dtype=np.float64))

            # ------------------
            # categorical kernels
            # ------------------
            elif 'probs' in post_kernel:
                # for categorical variables, keys "locs" and "precs" are all zeros
                trace_kernels['locs'].append(np.zeros(post_kernel['probs'].shape, dtype=np.float64))
                trace_kernels['sqrt_precs'].append(np.zeros(post_kernel['probs'].shape, dtype=np.float64))
                trace_kernels['probs'].append(post_kernel['probs'].astype(np.float64))
            else:
                raise NotImplementedError

        for key, kernel in trace_kernels.items():
            trace_kernels[key] = np.concatenate(kernel, axis=2)

        return trace_kernels


def _check_trace_kernels(trace_kernels):
    # check locations
    if np.isnan(trace_kernels['locs']).any():
        return False
    # check precisions
    if np.isnan(trace_kernels['sqrt_precs']).any():
        return False
    # check probabilities
    if np.isnan(trace_kernels['probs']).any():
        return False
    return True


@processify
def run_tf_network(observed_params, frac_feas, config, model_details):
    """Run network in a function that gets run in a temporary process. Important to keep the @processify decorator,
    otherwise TensorFlow keeps a bunch of global variables that do not get garbage collected and memory usage
    keeps increasing when Gryffin is run in a loop, until we run out of memory.
    """
    check_passed = False
    counter = 0
    learning_rate = model_details['learning_rate']
    # we do this because it can sometimes happen that the training does not converge and we get NaN losses, resulting
    # in the network returning NaN for all probs. Very practically this can be fixed by reducing the learning rate.
    while not check_passed:
        # instantiate and sample BNN
        tfprob_network = TfprobNetwork(config, model_details, frac_feas)
        tfprob_network.declare_training_data(observed_params)
        tfprob_network.construct_model(learning_rate=learning_rate)
        tfprob_network.sample()
        trace_kernels = tfprob_network.get_kernels()
        # checks and logs
        check_passed = _check_trace_kernels(trace_kernels)
        counter += 1
        tfprob_network.log(f'\nTfprobNetwork run attempt number {counter} '
                           f'with learning rate {learning_rate}', 'DEBUG')

        # after two failures, half learning rate at each iteration
        if counter > 1:
            learning_rate = learning_rate * 0.5
        # after ten failures, give up
        if counter > 10:
            raise GryffinComputeError("TfprobNetwork keeps returning (at least some) NaN values")
    return trace_kernels
