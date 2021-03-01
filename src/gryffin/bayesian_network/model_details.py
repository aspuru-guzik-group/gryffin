#!/usr/bin/env python

__author__ = 'Florian Hase'


model_details = {
    'num_epochs':  2 * 10**3, 'learning_rate': 0.01,
    'burnin': 0, 'thinning': 1, 'num_draws': 10**3,

    'num_layers': 3, 'hidden_shape': 6,
    'weight_loc': 0., 'weight_scale': 1., 'bias_loc': 0., 'bias_scale': 1.
}

