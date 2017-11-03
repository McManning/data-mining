# data-mining
Fiddling with some introductory data mining

## Requirements
* Python 3.5+
* scikit-learn

## Usage

`./main.py --k=4 --limit=100 samples/income_tr.csv`
* Generate a similarities table of the 4 closest records for the first 100 records in `income_tr.csv` using the default algorithm (composition of per-attribute or per-group similarities)

`./main.py --k=4 --out results.csv --alt samples/income_tr.csv`
* Generate a similarities table using the alternative algorithm (cosine similarity of a vectorization of each row) and write the resulting table to `results.csv`

`./main.py --k=4 --knn --test samples/income_te.csv --out results.csv samples/income_tr.csv`
* Run a k-Nearest Neighbor classifier against the test dataset (income_te) using the training dataset (income_tr) and outputting the results to `results.csv`. Uses majority votes for classification by default.

`./main.py --k=4 --knn --weighted --test samples/income_te.csv --out results.csv samples/income_tr.csv`
* Run a kNN classifier using weighted distances instead of majority votes against income_te and output the results to results.csv

Additional options available via `--help`
