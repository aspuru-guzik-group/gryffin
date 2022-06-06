[![build](https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml)
[![Documentation Status](https://readthedocs.org/projects/gryffin/badge/?version=latest)](http://gryffin.readthedocs.io/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Gryffin: Bayesian Optimization of Continuous and Categorical Variables
======================================================================

.. image:: https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml/badge.svg
    :target: https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml
.. image:: https://codecov.io/gh/aspuru-guzik-group/gryffin/branch/master/graph/badge.svg?token=pHQ8Z50qf8
    :target: https://codecov.io/gh/aspuru-guzik-group/gryffin

Welcome to **Gryffin**!

Designing functional molecules and advanced materials requires complex design choices: tuning
continuous process parameters such as temperatures or flow rates, while simultaneously selecting
catalysts or solvents. 

To date, the development of data-driven experiment planning strategies for
autonomous experimentation has largely focused on continuous process parameters despite the urge
to devise efficient strategies for the selection of categorical variables. Here, we introduce Gryffin,
a general purpose optimization framework for the autonomous selection of categorical variables
driven by expert knowledge.

Features
--------

* Gryffin extends the ideas of the `Phoenics <https://pubs.acs.org/doi/10.1021/acscentsci.8b00307>`_ optimizer to categorical variables. Phoenics is a linear-scaling Bayesian optimizer for continuous spaces which uses a kernel regression surrogate. Gryffin extends this approach to categorical and mixed continuous-categorical spaces. 
* Gryffin is linear-scaling appraoch to Bayesian optimization, whose acquisition function natively supports batched optimization. Gryffin's acquisition function uses an intuitive sampling parameter to bias its behaviour between exploitation and exploration. 
* Gryffin is capable of leveraging expert knowledge in the form of physicochemical descriptors to enhance its optimization performance (static formulation). Also, Gryffin can refine the provided descriptors to further accelerate the optimization (dynamic formulation) and foster scientific understanding. 

Use cases of Gryffin/Phoenics
-----------------------------

* `Self-driving lab to optimize multicomponet organic photovoltaic systems <https://onlinelibrary.wiley.com/doi/full/10.1002/adma.201907801>`_
* `Self-driving laboratory for accelerated discovery of thin-film materials <https://www.science.org/doi/10.1126/sciadv.aaz8867>`_
* `Data-science driven autonomous process optimization <https://www.nature.com/articles/s42004-021-00550-x>`_ 
* `Self-driving platform for metal nanoparticle synthesis <https://onlinelibrary.wiley.com/doi/full/10.1002/adfm.202106725>`_
* `Optimization of photophyscial properties of organic dye laser molecules <https://pubs.acs.org/doi/10.1021/acscentsci.1c01002>`_


## Gryffin: An algorithm for Bayesian optimization for categorical variables informed by physical intuition with applications to chemistry

Gryffin is an open source algorithm which implements Bayesian optimization for categorical variables and mixed categorical-continuous parameter domains [1].

... this repository is under construction ... please stay tuned! 


## Installation

##### From source 

Gryffin can be installed from source. 
```
git clone https://github.com/aspuru-guzik-group/gryffin.git
cd gryffin 
pip install .
```

### Reference 

[1] Häse, F., Roch, L.M. and Aspuru-Guzik, A., 2020. Gryffin: An algorithm for Bayesian optimization for categorical variables informed by physical intuition with applications to chemistry. arXiv preprint arXiv:2003.12127.

