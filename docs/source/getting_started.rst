Getting Started
===============

Integrate Gryffin into your project quickly!


Requirements
------------

* Python version >= 3.7


Installation
------------

To install ``gryffin`` from `PyPI <https://pypi.org/project/gryffin/>`_:

.. code-block:: console

    $ pip install gryffin

To install ``gryffin`` from source:

.. code-block:: console

    $ git clone git@github.com:aspuru-guzik-group/gryffin.git
    $ cd gryffin
    $ pip install .

Example Usage 
-------------

This is a minimalist example of Gryffin in action.


.. code-block:: python

    from gryffin import Gryffin
    import experiment

    # load config
    config = {
    
    }

    # initialize gryffin
    gryffin = Gryffin(
        config_dict=config
    )

    observations = [] 
    for iter in range(ITER_BUDGET):

        # query gryffin for new hyper-parameters
    	hyper_parameters  = gryffin.recommend(observations=observations)

        # evaluate the proposed parameters
        merit = experiment.run(hyper_paramters)
	    hyper_parameters['obj'] = merit

        observations.append(hyper_parameters)


















