from .decision_tree import DecisionTree
from evaluation import Bootstrap

class RandomForest():

	def __init__(self, ntree, criterion='entropy', numeric_partition='mean', attribute_sampler='logn'):
		self.ntree = ntree
		self.trees = self._create_trees(criterion, numeric_partition, attribute_sampler)

	def train(self, data, possible_values, feature_attributes, target_attribute):
		bootstrap_generator = Bootstrap(data)
		for t in self.trees:
			bootstrap_data = bootstrap_generator.get_data_sample()
			t.train(bootstrap_data, possible_values, feature_attributes, target_attribute)

	def predict(self, features):
		votes = []
		for t in self.trees:
			votes.append(t.predict(features))

		return self._get_vote_majority(votes)
			
	def _create_trees(self, criterion, numeric_partition, attribute_sampler):
		trees = []
		for i in range(self.ntree):
			trees.append(DecisionTree(criterion, numeric_partition, attribute_sampler))

		return trees

	def _get_vote_majority(self, votes):
		return max(set(votes), key=votes.count)