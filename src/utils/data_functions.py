import pandas as pd
import numpy as np

def is_data_of_same_class(data, class_attribute):
	return len(data[class_attribute].unique()) == 1

def get_majority_class(data, class_attribute):
	return data[class_attribute].mode()[0]

def get_class_entropy(data, target_attribute):
	total_entries = len(data)
	class_entropy = 0
	for v in data[target_attribute].unique().values:
		v_count = len(data[data[target_attribute] == v])
		prob = v_count/total_entries
		class_entropy -= prob*np.log2(prob)

	return class_entropy

def get_attribute_entropy(data, feature, target_attribute):
	data_len = len(data)
	attribute_entropy = 0
	for v in data[feature].unique().values:
		data_part = data[data[feature] == v]
		part_class_entropy = get_class_entropy(data_part, target_attribute)
		attribute_entropy += len(data_part)/data_len*part_class_entropy

	return attribute_entropy


def get_best_split_attribute(data, attributes, target_attribute, criterion='entropy'):
	if criterion == 'entropy':
		class_entropy = get_class_entropy(data, target_attribute)

	best_attr = None
	highest_gain = 0
	for attr in attributes:
		if criterion == 'entropy':
			attr_entropy = get_attribute_entropy(data[[attr, target_attribute]], attr, target_attribute)
			attr_gain = class_entropy - attr_entropy
			if attr_gain > highest_gain:
				best_attr = attr
				highest_gain = attr_gain

	return best_attr
