#!/usr/bin/env 

__author__ = 'Florian Hase'

#========================================================================

import sys
import threading

#========================================================================

def safe_execute(error):
	def decorator_wrapper(function):
		def wrapper(*args, **kwargs):
			try:
				function(*args, **kwargs)
			except:
				error_type, error_message, traceback = sys.exc_info()
				error(error_message)
		return wrapper
	return decorator_wrapper



def thread(function):
	def wrapper(*args, **kwargs):
		background_thread = threading.Thread(target = function, args = args, kwargs = kwargs)
		background_thread.start()
	return wrapper


