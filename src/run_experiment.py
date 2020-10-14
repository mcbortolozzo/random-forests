import pandas as pd
import json
import time

from evaluation import StratifiedKFolds
from classifier import RandomForest
from utils import get_possible_values

FOLDS = 10

HOUSE_VOTES_DATASET = 'data/house_votes_84.tsv'
HOUSE_VOTES_TARGET = 'target'
HOUSE_VOTES_DTYPE = str

WINE_RECOGNITION_DATASET = 'data/wine_recognition.tsv'
WINE_RECOGNITION_TARGET = 'target'
WINE_RECOGNITION_DTYPE = float

OUTPUT_FILE = 'output.json'

POSSIBLE_N_TREES = [1, 3, 7, 15, 31, 65, 127, 255]


def read_data(dataset_file, target_column, data_type):
	df = pd.read_csv(dataset_file, '\t', dtype = data_type)
	feature_columns = list(df.columns)
	feature_columns.remove(target_column)
	possible_values = get_possible_values(df)
	return df, feature_columns, possible_values


def get_test_results(clf, test_data, target_column, train_time):
	correct = 0
	test_input = test_data.drop(target_column, axis=1)
	test_target = test_data[target_column]
	prediction_time = 0
	for features, target in zip(test_input.iterrows(), test_target):
		start = time.clock()
		prediction = clf.predict(features[1])
		end = time.clock()
		prediction_time += (end - start)
		if prediction == target:
			correct += 1

	return {
		'acc': correct/len(test_data),
		'train_time': train_time,
		'pred_time': prediction_time/len(test_data)
	}


def run_experiment(dataset_file, target_column, data_type, n_tree, kfolds=10):
	print("Running Experiment for %s with %d trees and %d folds" % (dataset_file, n_tree, kfolds))
	print("Loading data...")
	# Load data
	df, feature_columns, possible_values = read_data(dataset_file, target_column, data_type)

	print("Preparing folds...")
	#Prepare folds
	kfold_generator = StratifiedKFolds(df,  kfolds, target_column)

	results = {'folds': {}}
	# Get kfolds and merge training folds into one
	for idx, (training, test) in enumerate(kfold_generator.get_folds(join_train=True)):
		print("Running fold", str(idx + 1))
		clf = RandomForest(n_tree)
		start = time.clock()
		clf.train(training, possible_values, feature_columns, target_column)
		end = time.clock()
		test_results = get_test_results(clf, test, target_column, end - start)
		results['folds'][idx] = test_results

	return results

def log_results(results):
	best_n_tree = None
	best_n_tree_acc = 0
	print('---------------------')
	for n_tree in results.keys():
		average_acc = 0
		average_train_time = 0
		average_pred_time = 0
		for fold in results[n_tree]['folds']:
			average_acc += results[n_tree]['folds'][fold]['acc']
			average_train_time += results[n_tree]['folds'][fold]['train_time']
			average_pred_time += results[n_tree]['folds'][fold]['pred_time']

		average_acc /= len(results[n_tree]['folds'])
		average_train_time /= len(results[n_tree]['folds'])
		average_pred_time /= len(results[n_tree]['folds'])

		results[n_tree]['acc'] = average_acc
		results[n_tree]['train_time'] = average_train_time
		results[n_tree]['pred_time'] = average_pred_time
		print('Average accuracy for %d trees: %.2f' %(n_tree, average_acc))

		if average_acc >= best_n_tree_acc:
			best_n_tree = n_tree
			best_n_tree_acc = average_acc

	print('Best number of trees: %d with %.2f accuracy' % (best_n_tree, best_n_tree_acc))


def optimize_tree_count(dataset_file, target_column, data_type, kfolds=10):
	results = {}
	for n_tree in POSSIBLE_N_TREES:
		out_log = run_experiment(dataset_file, target_column, data_type, n_tree, kfolds)
		results[n_tree] = out_log

	log_results(results)
	return results

final_output = {}
final_output['house_votes'] = optimize_tree_count(HOUSE_VOTES_DATASET, HOUSE_VOTES_TARGET, HOUSE_VOTES_DTYPE, FOLDS)
final_output['wine_recognition'] = optimize_tree_count(WINE_RECOGNITION_DATASET, WINE_RECOGNITION_TARGET, WINE_RECOGNITION_DTYPE, FOLDS)

with open(OUTPUT_FILE, 'w') as f:
	json.dump(final_output, f)