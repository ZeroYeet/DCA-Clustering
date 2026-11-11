# DCA Clustering

This project clusters tabular bank-customer data with an implementation of
IP-DCA (inertial proximal difference-of-convex algorithm). It includes a
preprocessing pipeline and a command-line script for evaluating CSV datasets.
The clustering approach is based on the minimum sum-of-squares clustering
(MSSC) formulation and DC programming/DCA methods described in the references
below.

## Datasets

Place CSV files in `data\`. The repository includes:

| File | Source |
| --- | --- |
| `bankCustomers.csv` | [BankCustomers dataset](https://gist.github.com/jrgamez/c9ea9e8a5d6d000619b31b8499af6a83#file-bankcustomers-csv) |
| `bankVN80k.csv` | [Vietnam Bank Churn Dataset 2025](https://www.kaggle.com/datasets/tranhuunhan/vietnam-bank-churn-dataset-2025) |
| `CC.csv` | [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |

The loader detects common CSV delimiters and, by default, infers whether the
first row is a header. It treats the last column as the target label, excludes
the target and columns recognized as identifiers from the features, imputes
missing numeric and categorical values, one-hot encodes categorical features,
and scales features to the range `[0, 1]`.

## Requirements

Use Python 3.10 or later. Install the packages used by the evaluator:

```powershell
python -m pip install numpy scikit-learn matplotlib
```

## Run an evaluation

From the project directory, evaluate all CSV files in `data\`:

```powershell
python evaluate_dca.py
```

Evaluate one dataset by its filename without the `.csv` extension:

```powershell
python evaluate_dca.py --dataset bankVN80K
```

By default, the evaluator uses the last column as the target, infers the
number of clusters from its distinct labels, and runs 10 initializations. To
choose a target column by its header name or zero-based index:

```powershell
python evaluate_dca.py --dataset churn --target-column cluster_group
python evaluate_dca.py --dataset churn --target-column 24
```

To treat a dataset as unlabeled, use every column as a feature and specify
the number of clusters. In this mode, no target column is removed:

```powershell
python evaluate_dca.py --dataset churn --no-target --n-clusters 3
```

## Estimate the number of clusters with the elbow method

Run the separate elbow script on one dataset:

```powershell
python elbow_dca.py --dataset bankVN80k_2 --target-column cluster_group
```

By default, it tests `K=2` through `K=10` on one reproducible sample of up to
5,000 rows, with 3 DCA initializations per K. It saves the MSSC values to
`results\<dataset>_elbow.csv` and an elbow chart to
`results\<dataset>_elbow.png`. Override the search range, sample size, or
initializations as needed:

```powershell
python elbow_dca.py --dataset bankVN80k_2 --target-column cluster_group --min-k 2 --max-k 8 --sample-size 10000 --n-init 5
```

The suggested K is the point farthest from the straight line joining the
first and last points of the MSSC curve. This is a heuristic; inspect the
curve and validate the resulting clusters for the dataset and use case.
For unlabeled data, add `--no-target` to use every column as a feature.

Useful options:

| Option | Description |
| --- | --- |
| `--dataset NAME` | Evaluate one dataset or `all` (the default). |
| `--target-column NAME_OR_INDEX` | Select the target column; defaults to the last column. |
| `--no-target` | Do not use a target column. Requires `--n-clusters`. |
| `--n-clusters N` | Set the number of clusters explicitly. |
| `--no-header` | Treat the CSV as having no header row. |
| `--t VALUE` | Set the DCA penalty parameter (default: `10`). |
| `--n-init N` | Number of initializations (default: `10`). |
| `--max-iter N` | Maximum iterations per initialization (default: `1000`). |
| `--tol VALUE` | Convergence tolerance (default: `1e-6`). |
| `--random-state N` | Set the random seed (default: `42`). |
| `--output PATH` | CSV results path; defaults to `results\<dataset>_dca_results.csv` for one dataset, or `results\dca_results.csv` for all datasets. |
| `--plot PATH` | Chart path; defaults to `results\<dataset>_dca_results.png` for one dataset, or `results\dca_results.png` for all datasets. |

## Evaluation metrics and outputs

The evaluator prints a summary, writes a CSV file, and saves a bar chart.
Results include the number of samples, features, and predicted clusters,
runtime, iterations, and these metrics:

| Metric | Meaning | Better |
| --- | --- | --- |
| Silhouette | Cohesion and separation of the predicted clusters. | Higher |
| Adjusted Rand Index (ARI) | Agreement between predicted clusters and the provided target labels, adjusted for chance. | Higher |

Silhouette is reported on its standard `[-1, 1]` scale. ARI is also reported
on its standard scale and is only available when a target column is included.
Silhouette is calculated on up to 2,000 randomly selected samples when the
dataset is larger; the subset is reproducible for a given `--random-state` and
includes at least one sample from every predicted cluster when there are at
most 2,000 clusters. If there are more clusters than the metric sample limit,
Silhouette is left blank rather than calculated on a subset that omits
clusters. Silhouette is also left blank when the evaluated subset has fewer
than two clusters or assigns every sample to its own cluster. The reported
cluster count always describes the full clustering.

The chart plots available Silhouette and ARI values. Results are overwritten
when the evaluator is run again with the same output paths.

The evaluator also saves a separate PCA 2D visualization for each dataset to
`results\<dataset>_pca.png`. When target labels are available, the figure shows
two panels: one colored by the DCA clusters and one by the true labels. PCA is
fit on a reproducible sample of up to 5,000 rows for large datasets; the plot
is for visualization only and does not affect clustering or metric values.

## References

1. Le Thi Hoai An, Le Hoai Minh, and Pham Dinh Tao. “New and efficient DCA
   based algorithms for minimum sum-of-squares clustering.” *Pattern
   Recognition*, 47(1), 388–401, 2014.
   [https://doi.org/10.1016/j.patcog.2013.07.012](https://doi.org/10.1016/j.patcog.2013.07.012)
2. Le Thi Hoai An and Pham Dinh Tao. “Minimum Sum-of-Squares Clustering by DC
   Programming and DCA.” In *ICIC 2009*, Lecture Notes in Computer Science,
   vol. 5755, pp. 327–340, Springer, 2009.
