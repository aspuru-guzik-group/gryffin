#!/usr/bin/env python

#==========================================================================

import os
import copy
import pickle
import numpy as np

#=========================================================================

class CategoryWriter(object):

    def __init__(self, num_opts, num_dims):
        self.num_opts = num_opts
        self.num_dims = num_dims


    def write_categories(self, home_dir, num_descs, with_descriptors = True):

        param_names = ['param_%d' % dim for dim in range(self.num_dims)]

        for param_name in param_names:

            opt_list = []
            for opt_index in range(self.num_opts):

                if with_descriptors:
                    descriptors = np.array([float(opt_index) for _ in range(num_descs)])
                    opt_dict    = {'name': 'x_%d' % opt_index, 'descriptors': descriptors}
                else:
                    opt_dict = {'name': 'x_%d' % opt_index}
                opt_list.append(copy.deepcopy(opt_dict))

            # create cat_details dir if necessary
            if not os.path.isdir('%s/CatDetails' % home_dir):
                os.mkdir('%s/CatDetails' % home_dir)

            cat_details_file = '%s/CatDetails/cat_details_%s.pkl' % (home_dir, param_name)
            pickle.dump(opt_list, open(cat_details_file, 'wb'))

#=========================================================================
