import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def is_data_of_same_class(data, class_attribute):
	return len(data[class_attribute].unique()) == 1

def get_majority_class(data, class_attribute):
	return data[class_attribute].mode()[0]

def get_class_entropy(data, target_attribute):
	total_entries = len(data)
	class_entropy = 0
	for v in data[target_attribute].unique():
		v_count = len(data[data[target_attribute] == v])
		prob = float(v_count)/total_entries
		class_entropy -= prob*np.log2(prob)

	return class_entropy

def get_mean_attribute_partition(data, attr):
	splitting_point = data[attr].mean()
	# Using split point +/- 1 as key for correct insertion on tree TODO find better way
	return { 
		splitting_point-1: data[data[attr] < splitting_point],
		splitting_point+1: data[data[attr] >= splitting_point]
	}, splitting_point

def get_numeric_attribute_partition(data, attr, numeric_partition):
	if numeric_partition == 'mean':
		return get_mean_attribute_partition(data, attr)
	else:
		raise Exception("not implemented")

def get_attribute_partitions(data, attr, numeric_partition):
	is_numeric = is_numeric_dtype(data[attr])
	splitting_point = None

	if is_numeric:
		partitions, splitting_point = get_numeric_attribute_partition(data, attr)
		return partitions, is_numeric, splitting_point
	elif is_string_dtype(data[attr]):
		return data.groupby(attr), is_numeric, splitting_point
	else:
		raise Exception("Invalid data type for decision tree feature")

def get_attribute_entropy(partitions, target_attribute, data_length):
	attribute_entropy = 0
	for _, data_part in partitions:
		part_class_entropy = get_class_entropy(data_part, target_attribute)
		attribute_entropy += float(len(data_part))/data_length*part_class_entropy

	return attribute_entropy	


def get_best_split_attribute(data, attributes, target_attribute, criterion='entropy', numeric_partition='mean'):
	data_length = len(data)
	if criterion == 'entropy':
		class_entropy = get_class_entropy(data, target_attribute)

	best_attr = None
	highest_gain = 0
	for attr in attributes:
		partitions, is_numeric, splitting_point = get_attribute_partitions(data, attr, numeric_partition)

		if criterion == 'entropy':
			attr_entropy = get_attribute_entropy(partitions, target_attribute, data_length)
			attr_gain = class_entropy - attr_entropy
			if attr_gain >= highest_gain:
				best_attr = attr
				highest_gain = attr_gain

	assert best_attr is not None

	return best_attr, highest_gain, is_numeric, splitting_point
