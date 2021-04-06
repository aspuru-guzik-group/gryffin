#!/usr/bin/env python 

__author__ = 'Florian Hase'


import numpy as np


class NaiveCategoricalOptimizer:

    def __init__(self, func=None):
        self.func = func

    def _set_func(self, func, pos=None):
        self.func = func
        if pos is not None:
            self.pos = pos
            self.num_pos = len(pos)

    def set_func(self, func, pos=None, highest=None):
        # set function
        self.highest = highest
        self._set_func(func, pos)

    def get_update(self, vector):
        func_best = self.func(vector)
        for pos_index, pos in enumerate(self.pos):
            if pos is None:
                continue

            current = vector[pos]
            perturb = np.random.choice(self.highest[pos_index])
            vector[pos] = perturb

            func_cand = self.func(vector)

            if func_cand < func_best:
                func_best = func_cand
            else:
                vector[pos] = current
        return vector


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.color_palette('YlOrRd', 256)

    cat_opt = NaiveCategoricalOptimizer()
    highest = 20
    max_loss = np.sum(np.square(np.zeros(2) - highest / 2.))
    def func(array):
        loss = np.sum( np.square( array - highest / 2. ) ) / max_loss
        return loss

    cat_opt.set_func(func, highest = [highest, highest], pos = np.arange(2))

    lines = np.arange(highest + 2) - 0.5

    start = np.zeros(2)

    fig = plt.figure(figsize=(6, 6))

    for _ in range(10**3):

        plt.clf()

        plt.plot(start[0], start[1], marker = 'o', color = 'k', markersize = 10)

        for x_line in lines:
            plt.plot([lines[0], lines[-1]], [x_line, x_line], color = 'k', lw = 0.5)
            plt.plot([x_line, x_line], [lines[0], lines[-1]], color = 'k', lw = 0.5)

        for x_element in np.arange(highest + 1):
            for y_element in np.arange(highest + 1):
                array = np.array([x_element, y_element])
                loss  = func(array)
                color_index = int( 255 * loss  )
                plt.plot(x_element, y_element, color = colors[color_index], marker = 's', markersize = 10, alpha = 0.5)

        start = cat_opt.get_update(start)

        plt.pause(0.05)


