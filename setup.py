"""Gryffin: An algorithm for Bayesian optimization of categorical variables informed by expert knowledge
"""

__author__ = 'Florian Hase, Matteo Aldeghi'

import versioneer
from setuptools import setup
from distutils.extension import Extension
import numpy as np


# readme file
def readme():
    with open('README.md') as f:
        return f.read()


# ----------
# Extensions
# ----------
ext_modules = [Extension('gryffin.bayesian_network.kernel_evaluations',
                         ['src/gryffin/bayesian_network/kernel_evaluations.c'],
                         include_dirs=[np.get_include()]),
               Extension('gryffin.bayesian_network.kernel_prob_reshaping',
                         ['src/gryffin/bayesian_network/kernel_prob_reshaping.c'],
                         include_dirs=[np.get_include()])]

# -----
# Setup
# -----
setup(name='matter-gryffin',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Bayesian optimization for continuous and categorical variables',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
      ],
      url='https://github.com/aspuru-guzik-group/gryffin',
      author='Florian Hase, Matteo Aldeghi',
      author_email='matteo.aldeghi@vectorinstitute.ai',
      license='MIT',
      packages=['gryffin'],
      package_dir={'': 'src'},
      zip_safe=False,
      tests_require=['pytest'],
      install_requires=['numpy', 'sqlalchemy', 'rich', 'pandas', 'matter-chimera',
                        'tensorflow>=2.3.0', 'tensorflow-probability>=0.11'],
      python_requires=">=3.6",
      ext_modules=ext_modules,
      entry_points={"console_scripts": ["gryffin = gryffin.cli:entry_point"]}
      )
