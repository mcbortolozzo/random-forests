import pandas as pd

from classifier import DecisionTree, StringDecisionTreeNode, LeafDecisionTreeNode
from utils import get_possible_values

BENCHMARK_DATASET = './data/dadosBenchmark_validacaoAlgoritmoAD.csv'

TARGET_COLUMN = 'Joga'

def compare_decision_tree(expected_node, actual_node):
	assert expected_node.is_leaf == actual_node.is_leaf
	if expected_node.is_leaf:
		assert expected_node.class_id == actual_node.class_id
	else:
		assert expected_node.attribute == actual_node.attribute
		assert expected_node.gain - actual_node.gain < 0.001
		assert expected_node.get_children_count() == actual_node.get_children_count()
		for k in expected_node.children.keys():
			expected_child = actual_node.get_child(k)
			assert expected_child, "Expected %s to be in node children but found %s" % (k, actual_node.children.keys())
			compare_decision_tree(expected_node.children[k], actual_node.children[k])

def print_tree(node, tab_level):

	if(node.is_leaf):
		print("%s Leaf Node - Class: %s" % ("".join(["\t"]*tab_level), node.class_id))
	else:
		print("%s Decision Node - Attribute: %s, Gain: %.3f"  % ("".join(["\t"]*tab_level), node.attribute[0], node.gain))

		for k,v in node.children.items():
			print("%s Case: %s" % ("".join(["\t"]*tab_level), k))
			print_tree(v, tab_level+1)



print('Loading Benchmark Data')
df = pd.read_csv(BENCHMARK_DATASET, delimiter=';', dtype=str)

feature_columns = list(df.columns.values)
feature_columns.remove(TARGET_COLUMN)

print('Training Decision Tree')
actual_tree = DecisionTree(attribute_sampler='n')
possible_values = get_possible_values(df)
actual_tree.train(df, possible_values, feature_columns, TARGET_COLUMN)

print('Validating Nodes')
# Build correct tree manually
expected_tree = DecisionTree()
expected_tree.root = StringDecisionTreeNode('Tempo', 0.247)

node = StringDecisionTreeNode('Umidade', 0.947)

child_node = LeafDecisionTreeNode('Nao')
node.add_child('Alta', child_node)

child_node = LeafDecisionTreeNode('Sim')
node.add_child('Normal', child_node)

expected_tree.root.add_child('Ensolarado', node)

node = LeafDecisionTreeNode('Sim')
expected_tree.root.add_child('Nublado', node)

node = StringDecisionTreeNode('Ventoso', 0.971)

child_node = LeafDecisionTreeNode('Sim')
node.add_child('Falso', child_node)

child_node = LeafDecisionTreeNode('Nao')
node.add_child('Verdadeiro', child_node)

expected_tree.root.add_child('Chuvoso', node)

#Compare Actual and Expected Tree

compare_decision_tree(expected_tree.root, actual_tree.root)

print("Benchmark Passed")

print("Tree Output:")

print_tree(actual_tree.root, 0)