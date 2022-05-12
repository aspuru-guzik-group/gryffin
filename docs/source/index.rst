Gryffin: Bayesian Optimization of Continuous and Categorical Variables
======================================================================

.. image:: https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml/badge.svg
    :target: https://github.com/aspuru-guzik-group/gryffin/actions/workflows/continuous-integration.yml
.. image:: https://codecov.io/gh/aspuru-guzik-group/gryffin/branch/master/graph/badge.svg?token=pHQ8Z50qf8
    :target: https://codecov.io/gh/aspuru-guzik-group/gryffin

**Gryffin** is a Python tool that ...

.. toctree::
   :maxdepth: 2
   :caption: Contents

   gryffin_api
   tutorial

Installation
--------

Citation
--------
If you use **Gryffin** in scientific publications, please cite the following papers depending on which aspects of the
code you used.

If you optimized **continuous variables**, please cite `this publication <https://pubs.acs.org/doi/abs/10.1021/acscentsci.8b00307>`_:

::

    @article{phoenics,
      title = {Phoenics: A Bayesian Optimizer for Chemistry},
      author = {Florian Häse and Loïc M. Roch and Christoph Kreisbeck and Alán Aspuru-Guzik},
      year = {2018}
      journal = {ACS Central Science},
      number = {9},
      volume = {4},
      pages = {1134--1145}
      }


If you optimized **categorical variables**, please cite `this publication <link URL>`_:

::

    @article{gryffin,
      title = {Gryffin},
      author = {},
      year = {2021},
      journal = {2103.03716},
      number = {arXiv},
      volume = {math.OC}.
      pages = {1134--1145}
      }

If you performed a **multi-objective optimization**, or used **periodic variables**, please cite
`this publication <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c8sc02239a#!divAbstract>`_:

::

    @article{chimera,
      title = {Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories},
      author = {Florian Häse and Loïc M. Roch and Alán Aspuru-Guzik},
      year = {2018},
      journal = {Chemical Science},
      number = {9},
      pages = {7642--7655}
      }

If you performed an optimization with **known or unknown feasibility constraints**, or used ``genetic`` as the
optimization algorithm for the acquisition, please cite `this publication <link URL>`_:

::

    @article{gryffin_feasibility,
      title={},
      author={},
      year={},
      journal = {},
      number = {},
      pages = {}
      }


License
-------
**Gryffin** is distributed under an Apache Licence 2.0.
