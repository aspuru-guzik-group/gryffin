#!/usr/bin/env python 

import versioneer
import numpy as np

from setuptools import setup, find_packages
from distutils.extension import Extension

#===============================================================================

def readme():
	with open('README.md', 'r') as content:
		return content.read()

#===============================================================================

try: 
	from Cython.Distutils import build_ext
	from Cython.Build     import cythonize
except ImportError:
	use_cython = False
else:
	use_cython = True

cmdclass    = {}
ext_modules = []

if use_cython:
	ext_modules += [
		Extension('gryffin.bayesian_network.kernel_evaluations',
			['src/gryffin/bayesian_network/kernel_evaluations.pyx']),
		Extension('gryffin.bayesian_network.kernel_prob_reshaping',
			['src/gryffin/bayesian_network/kernel_prob_reshaping.pyx']),]
	ext_modules = cythonize(ext_modules)
	cmdclass.update({'build_ext': build_ext})
else:
	ext_modules += [
		Extension('gryffin.bayesian_network.kernel_evaluations',
			['src/gryffin/bayesian_network/kernel_evaluations.c']),
		Extension('gryffin.bayesian_network.kernel_prob_reshaping.pyx',
			['src/gryffin/bayesian_network/kernel_prob_reshaping.c']),]

#===============================================================================

setup(name='gryffin',
	#version=versioneer.get_version(),
	version='0.1.0',
        #cmdclass=versioneer.get_cmdclass(),
	description='Bayesian optimization for categorical variables', 
	long_description=readme(),
	long_description_content_type='text/markdown',
	classifiers=[
		'Intended Audience :: Science/Research',
		'Operating System :: Unix', 
		'Programming Language :: Python',
		'Topic :: Scientific/Engineering', 
	],
	url='https://github.com/aspuru-guzik-group/gryffin',
	author='Florian Hase',
	packages=find_packages('./src/'),
	package_dir={'': 'src'},
	zip_safe=False,
	ext_modules=ext_modules,
	tests_require=['pytest'],
	install_requires=[
		'numpy',
		'sqlalchemy',
		],
	include_dirs=np.get_include(),
	python_requires='>=3.6',
)
