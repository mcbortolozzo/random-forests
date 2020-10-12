import numpy as np
import pandas as pd

from utils import is_data_of_same_class, get_majority_class, get_possible_values
from utils import AttributeSelector

from .decision_tree_node import LeafDecisionTreeNode, StringDecisionTreeNode, NumericDecisionTreeNode

class DecisionTree():

	def __init__(self, criterion = 'entropy', numeric_partition='mean', attribute_sampler='logn'):
		self.attribute_selector = AttributeSelector.Create(criterion, numeric_partition, attribute_sampler)

	def train(self, data, possible_values, feature_attributes, target_attribute):
		self.root = self._build_tree(data, possible_values, feature_attributes, target_attribute)

	def predict(self, features):
		current_node = self.root
		while not current_node.is_leaf:
			data = features[current_node.attribute[0]]
			current_node = current_node.get_child(data)

		return current_node.class_id

	def _create_leaf_node(self, data, target_attribute):
		return LeafDecisionTreeNode(get_majority_class(data, target_attribute))

	def _build_tree(self, data, possible_values, feature_attributes, target_attribute):
		node = None
		if is_data_of_same_class(data, target_attribute) or len(feature_attributes) == 0:
			node = self._create_leaf_node(data, target_attribute)
		else:
			selected_attribute, gain, data_partitions, is_numeric, splitting_point = self.attribute_selector.select_next_attribute(data, feature_attributes, target_attribute)

			# check there are entries for all possible values of the categorical attribute
			if not is_numeric and any([value not in data_partitions.groups or len(data_partitions.groups[value]) == 0 for value in possible_values[selected_attribute]]):
				node = self._create_leaf_node(data, target_attribute)
			else:
				if not is_numeric:
					node = StringDecisionTreeNode(selected_attribute, gain)
				else:
					node = NumericDecisionTreeNode(selected_attribute, gain, splitting_point)
				updated_attributes = list(feature_attributes)
				updated_attributes.remove(selected_attribute)
			
				resulting_nodes = self._get_children_nodes(data_partitions, possible_values, updated_attributes, target_attribute)
				for key, n in resulting_nodes.items():
					node.add_child(key, n)

		return node

	def _get_children_nodes(self, data_partitions, possible_values, updated_attributes, target_attribute):
		nodes = {}
		for part_key, data_part in data_partitions:
			nodes[part_key] = self._build_tree(data_part, possible_values, updated_attributes, target_attribute)

		return nodes





