Configuration
=============

Gryffin provides a flexible configuration interface, it accepts either a path to a config.json file or a python dict. 


.. code-block:: python
    
    gryffin = Gryffin(config_file='/path/to/your/config.json')

.. code-block:: python

    gryffin = Gryffin(config_dict={})


Gryffin exposes 5 configurable modules, `general`, `database`, `model`, `parameters` and `objectives`.

.. code-block:: JSON   

    {
        "general": {},
        "database": {},
        "model": {},
        "parameters": [],
        "objectives": []
    }

.. code-block:: python

    config = {
        "general": {},
        "database": {},
        "model": {},
        "parameters": [],
        "objectives": []  
        
    }

General Configuration
---------------------

.. list-table::

    * - Parameter
      - Definition
      - Example
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 
    * - format
      - 
      - 

Database Configuration
----------------------

.. list-table::

    * - Parameter
      - Definition
      - Example
    * - format
      - 
      - 
    * - path
      - 
      - 
    * - log_observations
      - 
      - 
    * - log_runtimes
      - 
      - 

Model Configuration
-------------------

.. list-table::
    :header-rows: 1

    * - Parameter
      - Definition
      - Example
    * - num_epochs
      - 
      - 
    * - learning_rate
      - 
      - 
    * - num_draws
      - 
      - 
    * - num_layers
      - 
      - 
    * - hidden_shape
      - 
      - 
    * - weight_loc
      - 
      - 
    * - weight_scale
      - 
      - 
    * - bias_loc
      - 
      - 
    * - bias_scale
      - 
      -    
    

Parameters Configuration
------------------------

Gryffin supports 3 parameter types, `continuous`, `discrete` and `categorical`. Each parameter is configured as elements of the root level parameters list:

.. code-block:: JSON
    {
        "parameters": [
                {},      
        ]
    }

Continuous Parameters:

.. list-table::
    :header-rows: 1

    * - Parameter
      - Definition
      - Example [type]
    * - name 
      - Human-readable parameter name 
      - "Your-parameter-name" [string]
    * - type 
      - Selects parameter type, either 'continuous', 'discrete' or 'categorical'
      - "continuous" [string]
    * - low
      - Lower bound of continuous parameter
      - [float]
    * - high
      - Upper bound of continuous parameter. Note: high must be larger than low.
      - [float]
    * - periodic 
      - Boolean flag indicating that the parameter is periodic
      - [bool]

Discrete Parameters:

.. list-table::
    :header-rows: 1

    * - Parameter
      - Definition
      - Example [type]
    * - name 
      - Human-readable parameter name 
      - "Your-parameter-name" [string]
    * - type 
      - Selects parameter type, either 'continuous', 'discrete' or 'categorical'
      - "discrete" [string]
    * - low
      - Lower bound of discrete parameter
      - [float]
    * - high
      - Upper bound of discrete parameter. Note: high must be larger than low.
      - [float]
    * - options 
      - ToDo: Need explanation of options
      - [List[]]
    * - descriptors 
      - ToDo: Need explanation of descriptors
      - [List[]]

Categorical Parameters:

.. list-table::
    :header-rows: 1

    * - Parameter
      - Definition
      - Example [type]
    * - name 
      - Human-readable parameter name 
      - "Your-parameter-name" [string]
    * - type 
      - Selects parameter type, either 'continuous', 'discrete' or 'categorical'
      - "categorical" [string]
    * - options 
      - ToDo: Need explanation of options
      - [List[]]
    * - descriptors 
      - ToDo: Need explanation of descriptors
      - [List[]]
    * - category_details
      - ToDo: Need explanation of category_details
      - [List[]]


Objective Configuration
-----------------------

Each objective is configured as elements of the root level objective list:

.. code-block:: JSON
    {
        "objectives": [
                {},      
        ]
    }

.. list-table::
    :header-rows: 1

    * - Parameter
      - Definition
      - Example [type]
    * - name 
      - Human-readable objective name 
      - "Your-parameter-name" [string]
    * - goal 
      - Optimization objective
      - min/max [string]
    * - tolerance
      - Termination tolerance on parameter changes
      - [float]
    * - absolute
      - Boolean flag indicating if objective is absolute
      - [bool]



