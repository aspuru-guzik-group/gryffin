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

    * - Parameter
      - Example
    * - name
      - Explanation and example
    * - type 
      - Enum, categorical, continuous, 

Discrete Parameters:

.. list-table::

    * - Parameter
      - Example
    * - name
      - Explanation and example
    * - type 
      - Enum, categorical, continuous, 

Categorical Parameters:

.. list-table::

    * - Parameter
      - Example
    * - name
      - Explanation and example
    * - type 
      - Enum, categorical, continuous, 


Objective Configuration
-----------------------

.. list-table::

    * - Parameter
      - Example



