Configuration
=============

Gryffin provides a flexible configuration interface, it accepts either a path to a config.json file or a python dict. 

.. example-code::

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
        "parameters": {},
        "objectives": {}
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
      - Example

Database Configuration
----------------------

.. list-table::

    * - Parameter
      - Example

Model Configuration
-------------------

.. list-table::

    * - Parameter
      - Example

Parameters Configuration
------------------------

Gryffin supports 3 parameter types, `continuous`, `discrete` and `categorical`. Each parameters is configured as elements of the root level parameters list:

.. code-black:: JSON
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
      - Human-readable parmater name 
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


Objective Configuration
-----------------------

.. list-table::
    :header-rows: 1

    * - Parameter
      - Example



