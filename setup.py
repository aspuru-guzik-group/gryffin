#!/usr/bin/env python

from setuptools import setup, find_packages
from distutils.extension import Extension


def readme():
    with open('README.md', 'r') as content:
        return content.read()


def requirements():
    with open('requirements.txt', 'r') as content:
        return content.readlines()


ext_modules = [
    Extension('gryffin.bayesian_network.kernel_evaluations',
              ['src/gryffin/bayesian_network/kernel_evaluations.c']),
    Extension('gryffin.bayesian_network.kernel_prob_reshaping',
              ['src/gryffin/bayesian_network/kernel_prob_reshaping.c']),]

# Preinstall numpy
from setuptools import dist
dist.Distribution().fetch_build_eggs(['numpy>=1.10'])
import numpy as np


setup(name='gryffin',
      #version=versioneer.get_version(),
      version='0.1.1',
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
      packages=find_packages('./src'),
      package_dir={'': 'src'},
      zip_safe=False,
      ext_modules=ext_modules,
      tests_require=['pytest'],
      include_dirs=np.get_include(),
      install_requires=requirements(),
      python_requires='>=3.6',
      )
