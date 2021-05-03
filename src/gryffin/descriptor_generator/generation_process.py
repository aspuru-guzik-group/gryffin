#!/usr/bin/env python 

__author__ = 'Florian Hase, Matteo Aldeghi'

import numpy as np
import tensorflow as tf
from gryffin.utilities.decorators import processify
from gryffin.utilities import GryffinComputeError


class Generator:

    def __init__(self, descs, objs, grid_descs, max_epochs=1000, learning_rate=0.001):

        self.descs = descs
        self.objs = objs
        self.grid_descs = grid_descs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        self.num_samples = descs.shape[0]
        self.num_descs = descs.shape[1]

        # attributes that will be assigned later on
        self.tf_descs = None
        self.tf_objs = None
        self.weights_0 = None
        self.biases_0 = None
        self.gen_descs = None
        self.corr_coeffs = None
        self.cov_gen_descs = None
        self.sess = None

        self.construct_comp_graph()

    def construct_comp_graph(self):

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        self.tf_descs = tf.compat.v1.placeholder(tf.float32, [None, self.num_descs])
        self.tf_objs = tf.compat.v1.placeholder(tf.float32, [None, 1])

        with tf.name_scope('auto_desc_gen'):

            self.weights_0 = tf.compat.v1.get_variable('weights_0', [self.num_descs, self.num_descs], initializer=tf.initializers.identity())
            self.biases_0 = tf.compat.v1.get_variable('biases_0',  [self.num_descs], initializer=tf.initializers.zeros())

            self.weights_0 = self.weights_0 + tf.random.normal([self.num_descs, self.num_descs], 0., 1e-5)
            self.biases_0 = self.biases_0 + tf.random.normal([self.num_descs], 0., 1e-5)

            activation = lambda x: tf.nn.softsign(x)
            regressor = lambda x: activation(tf.matmul(x, self.weights_0) + self.biases_0)

            gen_descs = regressor(self.tf_descs)
            self.gen_descs = gen_descs

            # compute correlation coefficients between descriptors and objectives
            gen_descs_mean, gen_descs_var = tf.nn.moments(gen_descs, axes=0)
            objs_mean, objs_var = tf.nn.moments(self.tf_objs, axes=0)

            gen_descs_var += 1e-6
            objs_var += 1e-6

            numerator = tf.reduce_mean((self.tf_objs - objs_mean) * (gen_descs - gen_descs_mean), axis=0)
            denominator = tf.sqrt(gen_descs_var * objs_var)
            corr_coeffs = numerator / denominator
            self.corr_coeffs = corr_coeffs

            # compute correlation coefficients among descriptors
            gen_descs_expand = tf.expand_dims(gen_descs - gen_descs_mean, -1)
            gen_descs_transpose = tf.transpose(gen_descs_expand, perm=[0, 2, 1])

            gen_descs_var_expand = tf.expand_dims(gen_descs_var, -1)
            gen_descs_var_transpose = tf.transpose(gen_descs_var_expand, perm=[1, 0])

            cov_gen_descs  = tf.reduce_mean(tf.matmul(gen_descs_expand, gen_descs_transpose), axis=0)
            cov_gen_descs /= tf.sqrt(tf.matmul(gen_descs_var_expand, gen_descs_var_transpose))
            self.cov_gen_descs = cov_gen_descs

            # compute loss for deviating from target binary matrix
            min_corr = 1. / np.sqrt(self.num_samples - 2)
            self.min_corr = min_corr
            norm_corr_coeffs = tf.nn.leaky_relu((tf.abs(corr_coeffs) - min_corr) / (1. - min_corr), 0.01)

            loss_0 = tf.reduce_mean(tf.square(tf.sin(np.pi * norm_corr_coeffs)))
            loss_1 = (1. - tf.reduce_max(tf.abs(norm_corr_coeffs)))

            # compute loss for non-zero correlations in generated descriptors
            norm_cov_x = tf.nn.leaky_relu((tf.abs(cov_gen_descs) - min_corr) / (1. - min_corr), 0.01)
            loss_2 = tf.reduce_sum(tf.square(tf.sin(np.pi * norm_cov_x / 2.))) / (self.num_descs**2 - self.num_descs)

            # weight regularization
            loss_3 = 1e-2 * tf.reduce_mean(tf.abs(self.weights_0))

            self.loss_0 = loss_0
            self.loss_1 = loss_1
            self.loss_2 = loss_2
            self.loss_3 = loss_3

            # register training operation
            self.loss = loss_0 + loss_1 + loss_2 + loss_3
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(init_op)

    def generate_descriptors(self):

        network_results = {}

        for epoch in range(self.max_epochs):
            current_loss, loss_0, loss_1, loss_2, loss_3, gen_descs, _ = self.sess.run(
                [self.loss, self.loss_0, self.loss_1, self.loss_2, self.loss_3, self.weights_0, self.train_op],
                feed_dict={self.tf_descs: self.descs, self.tf_objs: self.objs}
            )

        corr_coeffs, gen_descs_cov = self.sess.run([self.corr_coeffs, self.cov_gen_descs],
                                                   feed_dict={self.tf_descs: self.descs, self.tf_objs: self.objs})

        # register results
        network_results['min_corrs'] = self.min_corr
        network_results['comp_corr_coeffs'] = corr_coeffs
        network_results['gen_descs_cov'] = gen_descs_cov
        network_results['weights'] = self.weights_0.eval(session=self.sess)

        # run prediction
        auto_gen_descs = self.sess.run(self.gen_descs, feed_dict={self.tf_descs: self.grid_descs})
        network_results['auto_gen_descs'] = auto_gen_descs.astype(np.float64)

        # compute reduced descriptors
        sufficient_desc_indices = np.where(np.abs(corr_coeffs) > self.min_corr)[0]
        if len(sufficient_desc_indices) == 0:
            sufficient_desc_indices = np.array([0])
        reduced_gen_descs = auto_gen_descs[:, sufficient_desc_indices]
        network_results['reduced_gen_descs'] = reduced_gen_descs.astype(np.float64)
        network_results['sufficient_indices'] = sufficient_desc_indices

        return network_results


def _check_results(results):
    if np.isnan(results['reduced_gen_descs']).any():
        return False
    return True


@processify
def run_generator_network(descs, objs, grid_descs):
    generator = Generator(descs, objs, grid_descs)
    results = generator.generate_descriptors()
    check_passed = _check_results(results)
    if check_passed is False:
        d = results['reduced_gen_descs']
        raise GryffinComputeError(f"Generator returned NaN descriptors:\n{d}")
    return results





