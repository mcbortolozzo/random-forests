import pandas as pd
import numpy as np
from evaluation import Bootstrap, StratifiedKFolds
from classifier import DecisionTree
from utils import get_possible_values

DATA = './data/dadosBenchmark_validacaoAlgoritmoAD.csv'
LARGE_DATA = './data/house_votes_84.tsv'
test_data = pd.read_csv(DATA, delimiter=';')

large_test_data = pd.read_csv(LARGE_DATA, delimiter='\t')

def test_bootstrap():
	np.random.seed(1)
	bootstrap = Bootstrap(test_data)
	sample_idx = bootstrap.get_data_sample().index
	assert len(sample_idx) == len(test_data)
	assert (sample_idx == [5,11,12,8,9,11,5,0,0,1,12,7,13,12]).all()

def test_kfolds_3():
	np.random.seed(1)
	fold_generator = StratifiedKFolds(test_data, 3, "Joga")
	f1, f2, f3 = fold_generator.folds
	assert abs(len(f1) - len(f2)) <= 1
	assert abs(len(f2) - len(f3)) <= 1

	total_class_1 = len(test_data[test_data["Joga"] == "Nao"])
	total_class_2 = len(test_data[test_data["Joga"] == "Sim"])

	expected_class_1 = total_class_1/3
	expected_class_2 = total_class_2/3

	fold_indexes = set()

	for f in [f1,f2,f3]:
		len_class_1, len_class_2 = [len(x) for _, x in f.groupby("Joga")]
		assert abs(len_class_1 - expected_class_1) <= 1
		assert abs(len_class_2 - expected_class_2) <= 1

		fold_indexes.update(f.index)

	assert len(fold_indexes) == len(test_data)


def test_kfolds_5_large_data():
	np.random.seed(1)
	fold_generator = StratifiedKFolds(large_test_data, 5, "target")
	folds = fold_generator.folds

	total_class_1 = len(large_test_data[large_test_data["target"] == 0])
	total_class_2 = len(large_test_data[large_test_data["target"] == 1])

	expected_class_1 = total_class_1/5
	expected_class_2 = total_class_2/5

	fold_indexes = set()

	for f in folds:
		len_class_1, len_class_2 = [len(x) for _, x in f.groupby("target")]
		assert abs(len_class_1 - expected_class_1) <= 1
		assert abs(len_class_2 - expected_class_2) <= 1

		fold_indexes.update(f.index)

	assert len(fold_indexes) == len(large_test_data)

def test_prediction():
	expected_prediction = 'Sim'
	df = pd.read_csv(DATA, delimiter=';', dtype=str)
	feature_columns = ['Tempo', 'Temperatura', 'Umidade', 'Ventoso']
	actual_tree = DecisionTree()
	possible_values = get_possible_values(df)
	actual_tree.train(df, possible_values, feature_columns, 'Joga')

	prediction_data = pd.DataFrame.from_dict({'Tempo': ['Nublado'], 'Temperatura': ['Quente'], 'Umidade': ['Alta'], 'Ventoso': ['Falso']}).iloc[0]

	actual_prediction = actual_tree.predict(prediction_data)

	assert actual_prediction == expected_prediction


print('Validating Bootstrap')
test_bootstrap()
print('Validating Folds')
test_kfolds_3()
test_kfolds_5_large_data()
print('Validating Prediction')
test_prediction()

print("Everything seems to be ok")