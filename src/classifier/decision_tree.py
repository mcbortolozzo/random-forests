import numpy as np
import pandas as pd

from utils import is_data_of_same_class, get_majority_class, get_best_split_attribute, get_attribute_partitions

class DecisionTreeNode():

	def __init__(self, is_leaf):
		super().__init__()
		self.is_leaf = is_leaf

	def get_child(self, attribute_value):
		raise Exception("Not Implemented")

	def add_child(self, key, value):
		raise Exception("Not implemented")

	def get_children_count(self):
		raise Exception("Not implemented")	

class LeafDecisionTreeNode(DecisionTreeNode):

	def __init__(self, class_id):
		super().__init__(is_leaf=True)
		self.class_id = class_id

	def get_child(self, attribute_value):
		raise Exception("Leaf node has no children")

class AttributeDecisionTreeNode(DecisionTreeNode):

	def __init__(self, attribute, gain):
		super().__init__(is_leaf=False)
		self.attribute = attribute,
		self.gain = gain


class NumericDecisionTreeNode(AttributeDecisionTreeNode):

	def __init__(self, attribute, gain, splitting_point):
		super().__init__(attribute, gain)
		self.splitting_point = splitting_point
		self.above_child = None
		self.below_child = None

	def add_child(self, key, value):
		if key > self.splitting_point:
			self.above_child = value
		else:
			self.below_child = value

	def get_child(self, attribute_value):
		if attribute_value < self.splitting_point:
			return self.below_child
		else:
			return self.above_child	

	def get_children_count(self):
		return int(self.below_child is not None) + int(self.above_child is not None)


class StringDecisionTreeNode(AttributeDecisionTreeNode):

	def __init__(self, attribute, gain):
		super().__init__(attribute, gain)
		self.children = {}

	def get_child(self, attribute_value):
		return self.children[attribute_value]

	def add_child(self, key, value):
		self.children[key] = value

	def get_children_count(self):
		return len(self.children)

class DecisionTree():

	def __init__(self, criterion = 'entropy', numeric_partition='mean'):
		self.criterion = criterion
		self.numeric_partition = numeric_partition

	def train(self, data, feature_attributes, target_attribute):
		self.root = self._build_tree(data, feature_attributes, target_attribute)

	def _build_tree(self, data, feature_attributes, target_attribute):
		node = None
		if is_data_of_same_class(data, target_attribute) or len(feature_attributes) == 0:
			node = LeafDecisionTreeNode(get_majority_class(data, target_attribute))
		else:
			selected_attribute, gain, is_numeric, splitting_point = get_best_split_attribute(data, feature_attributes, target_attribute, self.criterion, self.numeric_partition)
			if not is_numeric:
				node = StringDecisionTreeNode(selected_attribute, gain)
			else:
				node = NumericDecisionTreeNode(selected_attribute, gain, splitting_point)
			updated_attributes = list(feature_attributes)
			updated_attributes.remove(selected_attribute)

			data_partitions, _, _ = get_attribute_partitions(data, selected_attribute, self.numeric_partition)

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





