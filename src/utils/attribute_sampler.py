import numpy as np

class AttributeSampler():

	def sample_attributes(self, attributes):
		raise NotImplementedError

	@classmethod
	def Create(cls, sampler_type):
		if sampler_type == 'n':
			return NAttributeSampler()
		elif sampler_type == 'logn':
			return LogNAttributeSampler()
		else:
			raise NotImplementedError(sampler_type)


class NAttributeSampler(AttributeSampler):

	def sample_attributes(self, attributes):
		return attributes

class LogNAttributeSampler(AttributeSampler):

	def sample_attributes(self, attributes):
		attr_len = len(attributes)
		if attr_len== 1:
			return attributes
		else:
			attr_log2 = int(np.ceil(np.log2(attr_len)))
			return np.random.choice(attributes, size=attr_log2, replace=False)