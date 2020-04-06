#!/usr/bin/env 
  
__author__ = 'Florian Hase'

#========================================================================

import sys

from utilities import PhoenicsModuleError, PhoenicsVersionError

#========================================================================

try:
    import tensorflow as tf
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = '\n\tTry installing the tensorflow package or use a different backend instead.'
    PhoenicsModuleError(str(error_message) + extension)

if not tf.__version__ in ['1.4.0', '1.4.1']:
    PhoenicsVersionError('cannot operate with tensorflow version: "%s".\n\tPlease install version 1.4.1' % tf.__version__)

try: 
	import edward as ed
except ModuleNotFoundError:
	_, error_message, _ = sys.exc_info()
	extension = '\n\tTry installing the edward package or use a different backend instead.'
	PhoenicsModuleError(str(error_message) + extension)

if not ed.__version__ in ['1.3.5']:
	PhoenicsVersionError('cannot operate with edward version: "%s".\n\tPlease install version 1.3.5' % ed.__version__)

#========================================================================


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

from BayesianNetwork.EdwardInterface.numpy_graph      import NumpyGraph
from BayesianNetwork.EdwardInterface.edward_interface import EdwardNetwork

