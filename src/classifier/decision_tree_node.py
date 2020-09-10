
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
