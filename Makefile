
download-data:
	curl https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/house_votes_84/house_votes_84.tsv.gz | gunzip - > data/house_votes_84.tsv
	curl https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/wine_recognition/wine_recognition.tsv.gz | gunzip - > data/wine_recognition.tsv

run-benchmark:
	python src/run_benchmark.py

run-tests:
	python src/test.py

run-experiment:
	python src/run_experiment.py

