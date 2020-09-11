import numpy as np
import pandas as pd

from utils import is_data_of_same_class, get_majority_class
from utils import AttributeSelector

from .decision_tree_node import LeafDecisionTreeNode, StringDecisionTreeNode, NumericDecisionTreeNode

class DecisionTree():

	def __init__(self, criterion = 'entropy', numeric_partition='mean', attribute_sampler='logn'):
		self.attribute_selector = AttributeSelector.Create(criterion, numeric_partition, attribute_sampler)

	def train(self, data, feature_attributes, target_attribute):
		self.root = self._build_tree(data, feature_attributes, target_attribute)

	def _create_leaf_node(self, data, target_attribute):
		return LeafDecisionTreeNode(get_majority_class(data, target_attribute))

	def _build_tree(self, data, feature_attributes, target_attribute):
		node = None
		if is_data_of_same_class(data, target_attribute) or len(feature_attributes) == 0:
			node = self._create_leaf_node(data, target_attribute)
		else:
			selected_attribute, gain, data_partitions, is_numeric, splitting_point = self.attribute_selector.select_next_attribute(data, feature_attributes, target_attribute)

			if any([len(x) == 0 for x in data_partitions]):
				node = self._create_leaf_node(data, target_attribute)
			else:
				if not is_numeric:
					node = StringDecisionTreeNode(selected_attribute, gain)
				else:
					node = NumericDecisionTreeNode(selected_attribute, gain, splitting_point)
				updated_attributes = list(feature_attributes)
				updated_attributes.remove(selected_attribute)
			
				resulting_nodes = self._get_children_nodes(data_partitions, updated_attributes, target_attribute)
				for key, n in resulting_nodes.items():
					node.add_child(key, n)

		return node

	def _get_children_nodes(self, data_partitions, updated_attributes, target_attribute):
		nodes = {}
		for part_key, data_part in data_partitions:
			if len(data_part) == 0:
				node = LeafDecisionTreeNode(get_majority_class(data_part, target_attribute))
				nodes[part_key] = node
			else:
				nodes[part_key] = self._build_tree(data_part, updated_attributes, target_attribute)

		return nodes





