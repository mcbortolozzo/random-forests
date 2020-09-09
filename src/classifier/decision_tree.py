import numpy as np
import pandas as pd

from utils import Node, is_data_of_same_class, get_majority_class, get_best_split_attribute

class DecisionTreeNode(Node):

	def set_leaf_class(self, class_id):
		self.is_leaf = True
		self.class_id = class_id

	def set_split_attribute(self, attribute, gain):
		self.is_leaf = False
		self.gain = gain
		self.attribute = attribute


class DecisionTree():

	def __init__(self, data, feature_attributes, target_attribute, criterion = 'entropy'):
		self.criterion = criterion
		self.clf = self._train(data, feature_attributes, target_attribute)

	def _train(self, data, feature_attributes, target_attribute):
		self.root = self._build_tree(data, feature_attributes, target_attribute)

	def _build_tree(self, data, feature_attributes, target_attribute):
		node = DecisionTreeNode(Node)
		if is_data_of_same_class(data, target_attribute) or len(feature_attributes) == 0:
			node.set_leaf_class(get_majority_class(data, target_attribute))
		else:
			selected_attribute, gain = get_best_split_attribute(data, feature_attributes, target_attribute, self.criterion)
			node.set_split_attribute(selected_attribute, gain)
			updated_attributes = list(feature_attributes)
			updated_attributes.remove(selected_attribute)

			resulting_nodes = self._get_children_nodes(data, selected_attribute, updated_attributes, target_attribute)
			for n in resulting_nodes:
				node.add_child(n)

		return node

	def _get_children_nodes(self, data, selected_attribute, updated_attributes, target_attribute):
		nodes = []
		for v in data[selected_attribute].unique():
			data_part = data[data[selected_attribute] == v]
			if len(data_part) == 0:
				node = Node()
				node.set_leaf_class(get_majority_class(data_part, target_attribute))
				nodes.append(node)
			else:
				nodes.append(self._build_tree(data_part, updated_attributes, target_attribute))

		return nodes

	def output_gains(self):
		self._output_gains(self.root)

	def _output_gains(self, node):
		if not node.is_leaf:
			print(node.attribute, node.gain)
			for n in node.children:
				self._output_gains(n)





