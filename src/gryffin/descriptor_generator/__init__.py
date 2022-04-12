#!/usr/bin/env 

__author__ = 'Florian Hase'

import sys
from gryffin.utilities import GryffinModuleError

try:
    import tensorflow as tf
except ModuleNotFoundError:
    _, error_message, _ = sys.exc_info()
    extension = '\n\tAutomatic descriptor generation requires the tensorflow package. Please install tensorflow or disable automatic descriptor generation by setting auto_desc_gen="False".'
    GryffinModuleError(str(error_message) + extension)

from .descriptor_generator import DescriptorGenerator
from .generation_process import Generator, run_generator_network
