#!/usr/bin/env python

import numpy as np
import sobol_seq


def estimate_feas_fraction(known_constraints, config, resolution=50):
    ''' Produces an estimate of the fraction of the domain which
    is feasible. For continuous valued parameters, we build a grid with
    "resolution" number of points in each dimensions. We measure each of
    possible categorical options

    Args:
        known_constraints (callable): callable function which retuns the
            feasibility mask
        config (): gryffin config
        resolution (int): the number of points to query per continuous dimension
    '''
    samples = []
    for param_ix, param in enumerate(config.parameters):
        if param['type'] == 'continuous':
            sample = np.linspace(param['specifics']['low'], param['specifics']['high'], resolution)
        elif param['type'] == 'discrete':
            num_options = int((param['specifics']['low']-param['specifics']['high'])/param['specific']['stride']+1)
            sample = np.linspace(param['specifics']['low'], param['specifics']['high'], num_options)
        elif param['type'] == 'categorical':
            sample = param['options']
        else:
            quit()
        samples.append(sample)
    # make meshgrid
    meshgrid = np.stack(np.meshgrid(*samples), len(samples))
    num_samples = np.prod(np.shape(meshgrid)[:-1])
    # reshape into 2d array
    samples = np.reshape(meshgrid, newshape=(num_samples, len(samples)))

    samples_dict = []
    for sample in samples:
        s = {f'{name}': element for element, name in zip(sample, config.parameters.name)}
        samples_dict.append(s)

    num_feas = 0.
    num_total = len(samples_dict)
    for sample in samples_dict:
        if known_constraints(sample):
            num_feas += 1
        else:
            pass
    frac_feas = num_feas/num_total
    assert  0. <= frac_feas <= 1.

    return frac_feas
