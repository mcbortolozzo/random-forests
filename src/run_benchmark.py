import pandas as pd

from classifier import DecisionTree

BENCHMARK_DATASET = './data/dadosBenchmark_validacaoAlgoritmoAD.csv'

TARGET_COLUMN = 'Joga'

print('Loading Benchmark Data')
df = pd.read_csv(BENCHMARK_DATASET, delimiter=';')

feature_columns = list(df.columns.values)
feature_columns.remove(TARGET_COLUMN)

print('Training Decision Tree')
clf = DecisionTree(df, feature_columns, TARGET_COLUMN)

print('Node Gains:')
clf.output_gains()


