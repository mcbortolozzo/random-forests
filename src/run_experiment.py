import pandas as pd
import json
import time

from evaluation import StratifiedKFolds
from classifier import RandomForest
from utils import get_possible_values, Analyzer

FOLDS = 10

HOUSE_VOTES_DATASET = 'data/house_votes_84.tsv'
HOUSE_VOTES_TARGET = 'target'
HOUSE_VOTES_DTYPE = str

WINE_RECOGNITION_DATASET = 'data/wine_recognition.tsv'
WINE_RECOGNITION_TARGET = 'target'
WINE_RECOGNITION_DTYPE = float

OUTPUT_FILE = 'output.json'
QTY_TEST_REPETITIONS = 10

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

def get_test_results_newer(clf, test_data, target_column, train_time):
	analyzer = Analyzer(list(test_data[target_column].unique()))
	test_input = test_data.drop(target_column, axis=1)
	test_target = test_data[target_column]
	prediction_time = 0
	for features, target in zip(test_input.iterrows(), test_target):
		start = time.process_time()
		prediction = clf.predict(features[1])
		end = time.process_time()
		prediction_time += (end - start)
		analyzer.addValueInConfusionMatrix(prediction,target)

	return {
		'acc': analyzer.calcAccuracy(),
		'fMeasure_micro': analyzer.calcFBethaMeasure(1,"micro"),
		'fMeasure_macro': analyzer.calcFBethaMeasure(1,"macro"),
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
		start = time.process_time()
		clf.train(training, possible_values, feature_columns, target_column)
		end = time.process_time()
		#test_results = get_test_results(clf, test, target_column, end - start)
		test_results = get_test_results_newer(clf, test, target_column, end - start)
		results['folds'][idx] = test_results

	return results

def log_results(results):
	best_n_tree = {'acc':None,
			'train_time':None,
			'pred_time':None,
			'fMeasure_micro':None,
			'fMeasure_macro':None}

	print('\n---------------------')

	for n_tree in results.keys():
		allValues = {'acc':list(),
			'train_time':list(),
			'pred_time':list(),
			'fMeasure_micro':list(),
			'fMeasure_macro':list()}
		average = 0.0
		median = 0.0
		percentile_75 = 0.0
		percentile_90 = 0.0
		percentile_99 = 0.0
		standarDeviation = 0.0

		for fold in results[n_tree]['folds']:
			allValues['acc'].append(results[n_tree]['folds'][fold]['acc'])
			allValues['train_time'].append(results[n_tree]['folds'][fold]['train_time'])
			allValues['pred_time'].append(results[n_tree]['folds'][fold]['pred_time'])
			allValues['fMeasure_micro'].append(results[n_tree]['folds'][fold]['fMeasure_micro'])
			allValues['fMeasure_macro'].append(results[n_tree]['folds'][fold]['fMeasure_macro'])
		for metric in allValues.keys():
			average = Analyzer.calcAverage(allValues[metric])
			standarDeviation = Analyzer.calcStandarDeviation(allValues[metric])
			percentile_75 = Analyzer.calcPercentile(allValues[metric],75)
			percentile_90 = Analyzer.calcPercentile(allValues[metric],90)
			percentile_99 = Analyzer.calcPercentile(allValues[metric],99)
			median = Analyzer.calcMedian(allValues[metric])

			results[n_tree][metric+'_avg'] = average
			results[n_tree][metric+'_sd'] = standarDeviation
			results[n_tree][metric+'_perc75'] = percentile_75
			results[n_tree][metric+'_perc90'] = percentile_90
			results[n_tree][metric+'_perc99'] = percentile_99
			results[n_tree][metric+'_median'] = median

			#print('Average %s for %d trees: %.2f' %(metric, n_tree, average))
			#print('Standart Deviation %s for %d trees: %.2f' %(metric, n_tree, standarDeviation))
			#print('Perc 75 %s for %d trees: %.2f' %(metric, n_tree, percentile_75))
			#print('Perc 90 %s for %d trees: %.2f' %(metric, n_tree, percentile_90))
			#print('Perc 99 %s for %d trees: %.2f' %(metric, n_tree, percentile_99))
			#print('Median %s for %d trees: %.2f' %(metric, n_tree, median))

			if best_n_tree[metric] == None or median >= best_n_tree[metric]['median']:
				best_n_tree[metric]= {'tree':n_tree, 'median': median}

	for metric in allValues.keys():
		print('Best number of trees: %d with %.2f %s' % (best_n_tree[metric]['tree'], best_n_tree[metric]['median'],metric))
	print('---------------------\n')


def optimize_tree_count(dataset_file, target_column, data_type, kfolds=10):
	results = {}
	for n_tree in POSSIBLE_N_TREES:
		out_log = run_experiment(dataset_file, target_column, data_type, n_tree, kfolds)
		results[n_tree] = out_log

	log_results(results)
	return results

final_output = {}
for fileID in range(QTY_TEST_REPETITIONS):
	print(f'\n\nTesting number: {fileID}')
	print('---------------------')
	final_output = {}
	final_output[f'house_votes'] = optimize_tree_count(HOUSE_VOTES_DATASET, HOUSE_VOTES_TARGET, HOUSE_VOTES_DTYPE, FOLDS)
	final_output[f'wine_recognition'] = optimize_tree_count(WINE_RECOGNITION_DATASET, WINE_RECOGNITION_TARGET, WINE_RECOGNITION_DTYPE, FOLDS)

	with open(f'{fileID}_'+OUTPUT_FILE, 'w') as f:
		json.dump(final_output, f)