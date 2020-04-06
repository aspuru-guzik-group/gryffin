
#!/usr/bin/env python 
  
__author__ = 'Florian Hase'

#========================================================================

import numpy as np 

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Build        import cythonize

ext_modules = [
	Extension(
			'kernel_evaluations', ['BayesianNetwork/kernel_evaluations.pyx'],
			 include_dirs       = [np.get_include(), '.'],
			 extra_compile_args = ['-fopenmp'], extra_link_args = ['-fopenmp'],
		),
	Extension(
			'kernel_prob_reshaper', ['BayesianNetwork/kernel_prob_reshaping.pyx'],
			 include_dirs       = [np.get_include(), '.'],
			 extra_compile_args = ['-fopenmp'], extra_link_args = ['-fopenmp'],
		),
]

setup(
	name = 'Phoenics',
	ext_modules  = cythonize(ext_modules)
)
