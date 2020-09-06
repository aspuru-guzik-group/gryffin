#!/usr/bin/env python 

# import versioneer
# import numpy as np

from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.build import build as build_orig

#===============================================================================

def readme():
	with open('README.md', 'r') as content:
		return content.read()

#===============================================================================

ext_modules = [
	Extension('gryffin.bayesian_network.kernel_evaluations',
		['src/gryffin/bayesian_network/kernel_evaluations.c']),
	Extension('gryffin.bayesian_network.kernel_prob_reshaping',
		['src/gryffin/bayesian_network/kernel_prob_reshaping.c']),]


class build(build_orig):

    def finalize_options(self):
        super().finalize_options()
        # I stole this line from ead's answer:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        # or just modify my_c_lib_ext directly here, ext_modules should contain a reference anyway
        extension = next(m for m in self.distribution.ext_modules if m == ext_modules[0])
        extension.include_dirs.append(numpy.get_include())


#===============================================================================

setup(name='gryffin',
	#version=versioneer.get_version(),
	version='0.1.0',
    # cmdclass=versioneer.get_cmdclass(),
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
	cmdclass={'build': build},
	tests_require=['pytest'],
	install_requires=[
		'numpy',
		'sqlalchemy',
		],
	python_requires='>=3.6',
)
