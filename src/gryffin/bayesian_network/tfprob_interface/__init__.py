#!/usr/bin/env 
  
__author__ = 'Florian Hase'

#========================================================================

import sys

from gryffin.utilities import GryffinModuleError, GryffinVersionError

#========================================================================

try:
    import tensorflow as tf
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = '\n\tTry installing the tensorflow package or use a different backend instead.'
    GryffinModuleError(str(error_message) + extension)
#========================================================================


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .numpy_graph      import NumpyGraph
from .tfprob_interface import TfprobNetwork, run_tf_network

