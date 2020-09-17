#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os, sys
import pickle
import numpy      as np
import tensorflow as tf

#========================================================================

class Generator(object):

	eta      = 1e-3
	max_iter = 10**3

	def __init__(self, config_file):
		self.config_file = config_file
		with open(self.config_file, 'rb') as content:
			self.config = pickle.load(content)
		for key, value in self.config.items():
			setattr(self, key, value)

	def construct_comp_graph(self):

		tf.compat.v1.reset_default_graph()
		tf.compat.v1.disable_eager_execution()
		self.tf_descs = tf.compat.v1.placeholder(tf.float32, [None, self.num_descs])
		self.tf_objs  = tf.compat.v1.placeholder(tf.float32, [None, 1])
		
		with tf.name_scope('auto_desc_gen'):

			self.weights_0  = tf.compat.v1.get_variable('weights_0', [self.num_descs, self.num_descs], initializer = tf.initializers.identity())
			self.biases_0   = tf.compat.v1.get_variable('biases_0',  [self.num_descs],                 initializer = tf.initializers.zeros())

			self.weights_0  = self.weights_0 + tf.random.normal([self.num_descs, self.num_descs], 0., 1e-5)
			self.biases_0   = self.biases_0  + tf.random.normal([self.num_descs], 0., 1e-5)

			activation = lambda x: tf.nn.softsign(x)
			regressor  = lambda x: activation( tf.matmul(x, self.weights_0) + self.biases_0 )

			gen_descs      = regressor(self.tf_descs)
			self.gen_descs = gen_descs

			# compute correlation coefficients between descriptors and objectives
			gen_descs_mean, gen_descs_var = tf.nn.moments(gen_descs,    axes = 0)
			objs_mean,      objs_var      = tf.nn.moments(self.tf_objs, axes = 0)

			gen_descs_var += 1e-6
			objs_var      += 1e-6

			numerator   = tf.reduce_mean( (self.tf_objs - objs_mean) * (gen_descs - gen_descs_mean), axis = 0 )
			denominator = tf.sqrt( gen_descs_var * objs_var )
			corr_coeffs = numerator / denominator
			self.corr_coeffs = corr_coeffs

			# compute correlation coefficients among descriptors
			gen_descs_expand    = tf.expand_dims(gen_descs - gen_descs_mean, -1)
			gen_descs_transpose = tf.transpose(gen_descs_expand, perm = [0, 2, 1])

			gen_descs_var_expand    = tf.expand_dims(gen_descs_var, -1)
			gen_descs_var_transpose = tf.transpose(gen_descs_var_expand, perm = [1, 0])

			cov_gen_descs  = tf.reduce_mean( tf.matmul(gen_descs_expand, gen_descs_transpose), axis = 0 )
			cov_gen_descs /= tf.sqrt( tf.matmul(gen_descs_var_expand, gen_descs_var_transpose) )
			self.cov_gen_descs = cov_gen_descs

			# compute loss for deviating from target binary matrix 
#			min_corr         = 2. * 2. / (3. * np.sqrt(self.num_samples - 1) )    # corresponds to 95 % confidence interval
			min_corr         = 1. / np.sqrt(self.num_samples - 2)
			self.min_corr    = min_corr
			norm_corr_coeffs = tf.nn.leaky_relu( (tf.abs(corr_coeffs) - min_corr) / (1. - min_corr), 0.01 )
		
			loss_0  = tf.reduce_mean( tf.square( tf.sin(np.pi * norm_corr_coeffs) ) )
			loss_1  = (1. - tf.reduce_max(tf.abs(norm_corr_coeffs)))
#			loss_1  = tf.square( tf.cos( np.pi * norm_corr_coeffs[0] ) ) + tf.square( tf.cos( corr_coeffs[0] * np.pi / 2. ) )
#           loss_1 /= self.num_descs

			# compute loss for non-zero correlations in generated descriptors
			norm_cov_x = tf.nn.leaky_relu( (tf.abs(cov_gen_descs) - min_corr) / (1. - min_corr), 0.01 )
			loss_2     = tf.reduce_sum( tf.square( tf.sin(np.pi * norm_cov_x / 2.) ) ) / (self.num_descs**2 - self.num_descs)
			
			# weight regularization
			loss_3 = 1e-2 * tf.reduce_mean(tf.abs(self.weights_0))		
	
			self.loss_0 = loss_0
			self.loss_1 = loss_1
			self.loss_2 = loss_2
			self.loss_3 = loss_3

			# register training operation
			self.loss     = loss_0 + loss_1 + loss_2 + loss_3
			optimizer     = tf.compat.v1.train.AdamOptimizer(learning_rate = self.eta)
			self.train_op = optimizer.minimize(self.loss)

#		print('initializing graph variables')
		init_op   = tf.group( tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer() )
		config    = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1)
		self.sess = tf.compat.v1.Session(config = config)
#		print('running init_op')
		self.sess.run(init_op)	
#		print('completed init_op')



	def generate_descriptors(self):
		
#		print('OBJS',  self.objs)
#		print('DESCS', self.descs)

		self.construct_comp_graph()
		for epoch in range(self.max_iter):
			current_loss, loss_0, loss_1, loss_2, loss_3, gen_descs, _ = self.sess.run(
					[self.loss, self.loss_0, self.loss_1, self.loss_2, self.loss_3, self.weights_0, self.train_op], 
					feed_dict = {self.tf_descs: self.descs, self.tf_objs: self.objs}
				)

		corr_coeffs, gen_descs_cov = self.sess.run([self.corr_coeffs, self.cov_gen_descs], 
												    feed_dict = {self.tf_descs: self.descs, self.tf_objs: self.objs})

		# register results
		self.config['min_corrs']        = self.min_corr
		self.config['comp_corr_coeffs'] = corr_coeffs
		self.config['gen_descs_cov']    = gen_descs_cov
#		print('EVALUATING WEIGHTS')
#		with self.sess.as_default():
		self.config['weights']          = self.weights_0.eval(session = self.sess)
#		print('EVALUATED WEIGHTS')

		# run prediction
		auto_gen_descs                = self.sess.run(self.gen_descs, feed_dict = {self.tf_descs: self.grid_descs})
		self.config['auto_gen_descs'] = auto_gen_descs.astype(np.float64)
	
		# compute reduced descriptors
		sufficient_desc_indices = np.where(np.abs(corr_coeffs) > self.min_corr)[0]
		if len(sufficient_desc_indices) == 0:
			sufficient_desc_indices = np.array([0])
		reduced_gen_descs = auto_gen_descs[:, sufficient_desc_indices]
		self.config['reduced_gen_descs'] = reduced_gen_descs.astype(np.float64)


		# write pickle file
		results_path     = self.config_file.split('/')
		results_path[-1] = 'completed_%s' % results_path[-1]
		results_file     = '/'.join(results_path)

		with open(results_file, 'wb') as content:
			pickle.dump(self.config, content)


#========================================================================

if __name__ == '__main__':

	import time 
	time.sleep(0.2)
	config    = sys.argv[1]
	generator = Generator(config)
	generator.generate_descriptors()


