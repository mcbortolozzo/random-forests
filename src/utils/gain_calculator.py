import pandas as pd
import numpy as np

class GainCalculator():

	@classmethod
	def Create(cls, criterion):
		if criterion == 'entropy':
			return EntropyGainCalculator()
		else:
			raise NotImplementedError(criterion)

	def get_partitions_length(self, partitions):
		return sum([len(x) for _, x in partitions])

	def get_partition_values(self, partitions):
		return [x for _, x in partitions]

	def calculate_gain(self, partitions, target_attribute):
		raise NotImplementedError

class EntropyGainCalculator(GainCalculator):

	def get_class_entropy(self, data, target_attribute):
		total_entries = len(data)
		class_entropy = 0
		for v in data[target_attribute].unique():
			v_count = len(data[data[target_attribute] == v])
			prob = float(v_count)/total_entries
			class_entropy -= prob*np.log2(prob)

		return class_entropy

	def calculate_gain(self, partitions, target_attribute):
		data_length = self.get_partitions_length(partitions)
		data_entropy = self.get_class_entropy(pd.concat(self.get_partition_values(partitions)), target_attribute)

		attribute_entropy = 0
		for _, data_part in partitions:
			part_class_entropy = self.get_class_entropy(data_part, target_attribute)
			attribute_entropy += float(len(data_part))/data_length*part_class_entropy

		return data_entropy - attribute_entropy