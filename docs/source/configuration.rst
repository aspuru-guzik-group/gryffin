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
        "parameters": {},
        "objectives": {}
    }

.. code-block:: python

    config = {
        
    }


