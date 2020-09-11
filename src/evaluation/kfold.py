import numpy as np
import pandas as pd

class StratifiedKFolds():

    def __init__(self, data, k, target_attribute):
        self.k = k
        self.folds = self._split_folds(data, target_attribute)

    def _split_fold_group(self, data_group):
        split_idx = [int(len(data_group)*float(i)/self.k) for i in range(1, self.k)]
        return np.split(data_group.sample(frac=1), split_idx)

    def _split_folds(self, data, target_attribute):     
        split_data = data.groupby(target_attribute).apply(self._split_fold_group)

        class_count = len(data[target_attribute].unique())
        folds = []
        for j in range(self.k):
            current_fold = []
            for i in range(class_count):
                current_fold.append(split_data.iloc[i][j])
            folds.append(pd.concat(current_fold))
        return folds

    def get_folds(self, join_train=True):
        for i in range(self.k):
            training = pd.concat([x for j,x in enumerate(self.folds) if j != i])
            test = self.folds[i]
            yield training, test 

