from .gain_calculator import GainCalculator
from .data_partitioner import DataPartitioner
from .attribute_sampler import AttributeSampler


class AttributeSelector():

	def __init__(self, attribute_sampler, data_partitioner, gain_calculator):
		self.attribute_sampler = attribute_sampler
		self.data_partitioner = data_partitioner
		self.gain_calculator = gain_calculator

	def select_next_attribute(self, data, feature_attributes, target_attribute):
		selected_attributes = self.attribute_sampler.sample_attributes(feature_attributes)

		highest_gain = 0
		best_attribute = None
		best_data_partition = None
		best_splitting_point = None
		for attr in selected_attributes:
			data_partitions, is_numeric, splitting_point = self.data_partitioner.partition_attribute(data, attr)
			gain = self.gain_calculator.calculate_gain(data_partitions, target_attribute)
			if gain >= highest_gain:
				highest_gain = gain
				best_attribute = attr
				best_data_partition = data_partitions
				best_splitting_point = splitting_point

		assert best_attribute is not None

		return best_attribute, highest_gain, best_data_partition, is_numeric, best_splitting_point

	@classmethod
	def Create(cls, criterion, numeric_partition, attribute_sampling):
		gain_calculator = GainCalculator.Create(criterion)
		data_partitioner = DataPartitioner(numeric_partition)
		attribute_sampler = AttributeSampler.Create(attribute_sampling)
		return AttributeSelector(attribute_sampler, data_partitioner, gain_calculator)

