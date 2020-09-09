
class Node():

	def __init__(self, data = None):
		self.data = data
		self.children = []

	def add_child(self, node):
		self.children.append(node)