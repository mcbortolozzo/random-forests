# Random Forests Implementation

This repository contains an implementation of the random forests algorithm, based on categorical and numerical decision trees for the Machine Learning class at UFRGS.

## Installation

Install the requirements

```
pip install -r requirements.txt
```

Download the data:

```
make download-data
```

**Important:** The data for the validation benchmark has to be downloaded manually and the `.csv` file placed in the `data` folder

## Validation

Run the validation benchmark:

```
make run-benchmark
```

Run additional tests:

```
make run-tests
```

## Usage

In order to run the experiment which attempts to optimize the number of trees, run the following command

```
make run-experiment
```
