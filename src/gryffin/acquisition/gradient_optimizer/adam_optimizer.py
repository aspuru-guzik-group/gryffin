#!/usr/bin/env python 

__author__ = 'Florian Hase, Matteo Aldeghi'

import numpy as np


class AdamOptimizer:

    def __init__(self, func=None, pos=None, eta=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.func = func
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

        # init Adam parameters if pos is provided
        if pos is not None:
            self.init_params(pos)
        else:
            self.pos = None
            self.num_pos = None
            self.ms = None
            self.vs = None

        # step used to estimate gradients numerically
        self.dx = 1e-6

    def init_params(self, pos):
        """
        pos : list
            list of indices corresponding to the continuous variables in the vectors that will be passed to update.
        """
        self.pos = np.array(pos)
        self.num_pos = len(pos)
        self.ms = np.zeros(self.num_pos)  # moment vector
        self.vs = np.zeros(self.num_pos)  # exponentially weighted infinity norm

    def reset(self):
        self.iterations = 0
        self.ms = np.zeros(self.num_pos)
        self.vs = np.zeros(self.num_pos)

    def set_func(self, func, pos=None):
        """
        func : callable
            function to be optimized.
        pos : list
            list of indices corresponding to the continuous variables in the vectors that will be passed to update.
        """
        self.func = func
        self.reset()
        if pos is not None:
            self.init_params(pos)

    def grad(self, sample):
        """Estimate the gradients"""
        gradients = np.zeros(len(sample), dtype=np.float32)
        perturb = np.zeros(len(sample), dtype=np.float32)

        for i in self.pos:
            if i is None:
                continue
            perturb[i] += self.dx
            gradient = (self.func(sample + perturb) - self.func(sample - perturb)) / (2. * self.dx)
            gradients[i] = gradient
            perturb[i] -= self.dx

        return gradients

    def get_update(self, sample):
        """
        vector : nd.array
        """
        # get gradients: g
        grads = self.grad(sample)
        # get iteration: t
        self.iterations += 1

        # eta(t) = eta * sqrt(1 – beta2(t)) / (1 – beta1(t))
        # where: beta(t) = beta^t
        eta_next = self.eta * (np.sqrt(1. - np.power(self.beta_2, self.iterations)) /
                               (1. - np.power(self.beta_1, self.iterations)))
        # m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
        ms_next = (self.beta_1 * self.ms) + (1. - self.beta_1) * grads
        # v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
        vs_next = (self.beta_2 * self.vs) + (1. - self.beta_2) * np.square(grads)

        # update sample: x(t) = x(t-1) – eta(t) * m(t) / (sqrt(v(t)) + eps)
        sample_next = sample - eta_next * ms_next / (np.sqrt(vs_next) + self.epsilon)

        # update params
        self.ms = ms_next
        self.vs = vs_next

        return sample_next


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn as sns

    adam = AdamOptimizer()

    def func(x):
        return (x - 1)**2

    adam.set_func(func, pos=np.arange(1))

    domain = np.linspace(-1, 3, 200)
    values = func(domain)

    start = np.zeros(1) - 0.8

    plt.ion()

    for _ in range(10**3):

        plt.clf()
        plt.plot(domain, values)
        plt.plot(start, func(start), marker='o', color='k')

        start = adam.get_update(start)

        plt.pause(0.05)





