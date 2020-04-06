#!/usr/bin/env 

__author__ = 'Florian Hase'

#========================================================================

import numpy as np

from . import ParserJSON, ParserPickle

#========================================================================

class CategoryParser(object):

	def __init__(self, file_name = None):
		self.file_name     = file_name
		self.json_parser   = ParserJSON()
		self.pickle_parser = ParserPickle()
		self.parsers       = [self.json_parser, self.pickle_parser]

	def parse(self, file_name = None):
		# update file name
		if not file_name is None:
			self.file_name = file_name

		suffix = self.file_name.split('.')[-1]
		if suffix == 'json':
			cat_details = self.json_parser.parse(self.file_name)
		elif suffix == 'pkl':
			cat_details = self.pickle_parser.parse(self.file_name)
		elif suffix == 'csv':
			raise NotImplementedError
		else:
			# try different parsers
			cat_details = None
			for parser in self.parsers:
				try:
					cat_details = parser.parse(self.file_name)
				except:
					pass
				if cat_details is not None:
					break
			else:
				PhoenicsUnkown

		# reformat information
		options     = [cat_dict['name'] for cat_dict in cat_details]
	
		try:
			descriptors = np.array([cat_dict['descriptors'] for cat_dict in cat_details])
	
			# TODO: rescale the descriptors!!
			min_descriptors, max_descriptors = np.amin(descriptors, axis = 0), np.amax(descriptors, axis = 0)
			descriptors = (descriptors - min_descriptors) / (max_descriptors - min_descriptors)

			descs = []
			for descriptor in descriptors:
				desc = descriptor[np.where(np.isfinite(descriptor))[0]]
				descs.append(desc)
			descs = np.array(descs)
			descriptors = descs

		except KeyError:
			descriptors = None
			
		return options, descriptors

