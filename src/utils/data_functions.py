import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def is_data_of_same_class(data, class_attribute):
	return len(data[class_attribute].unique()) == 1

def get_majority_class(data, class_attribute):
	return data[class_attribute].mode()[0]

def get_possible_values(df):
	possible_values = {}
	for key in df.columns:
		if not is_numeric_dtype(df[key]):
			possible_values[key] = df[key].unique()

	return possible_values