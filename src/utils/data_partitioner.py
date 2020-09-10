from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class DataPartitioner():

	def __init__(self, numeric_partition, string_partition = 'unique'):
		self.string_partitioner = DataPartitioner.CreateStringPartitioner(string_partition)
		self.numeric_partitioner = DataPartitioner.CreateNumericPartitioner(numeric_partition)

	def partition_attribute(self, data, attribute):
		is_numeric = is_numeric_dtype(data[attribute])
		splitting_point = None

		if is_numeric:
			partitions, splitting_point = self.numeric_partitioner.split(data, attribute)
			return partitions, is_numeric, splitting_point
		elif is_string_dtype(data[attribute]):
			return self.string_partitioner.split(data, attribute), is_numeric, splitting_point
		else:
			raise Exception("Invalid data type for decision tree feature")

	@classmethod
	def CreateStringPartitioner(cls, string_partition):
		if string_partition == 'unique':
			return UniqueStringDataPartitioner()
		else:
			raise NotImplementedError(string_partition)

	@classmethod
	def CreateNumericPartitioner(cls, numeric_partition):
		if numeric_partition == 'mean':
			return MeanNumericDataPartitioner()
		else:
			raise NotImplementedError

class DataTypePartitioner():

	def split(self, data, attribute):
		raise NotImplementedError

class UniqueStringDataPartitioner(DataTypePartitioner):

	def split(self, data, attribute):
		return data.groupby(attribute)

class MeanNumericDataPartitioner(DataTypePartitioner):

	def split(self, data, attribute):
		splitting_point = data[attr].mean()
		# Using split point +/- 1 as key for correct insertion on tree TODO find better way
		return { 
			splitting_point-1: data[data[attr] < splitting_point],
			splitting_point+1: data[data[attr] >= splitting_point]
		}, splitting_point