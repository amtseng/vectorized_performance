## Vectorized computation of various performance metrics for machine learning

### The goal of this repository
This repository contains code to compute various performance metrics in a vectorized way. Oftentimes, there are certain metrics we want to compute over many data points. The functions here aim to provide a fast way to compute these metrics over many points at the same time (rather than iteratively).

The focus here is on commonly used metrics for machine learning that _do not_ already have a vectorized implementation in NumPy, SciPy, or Scikit Learn.

### Dependencies
- Python 3.7
- NumPy 1.17

### Supported functionalities
- Jensen-Shannon divergence
- auPRC (area under precision-recall curve)
- Pearson correlation
- Spearman correlation

### API
- `jensen_shannon_distance(probs1, probs2)`
Computes the Jesnsen-Shannon distance in the last dimension of `probs1` and `probs2`. `probs1` and `probs2` must be the same shape. For example, if they are both A x B x L arrays, then the KL divergence of corresponding L-arrays will be computed and returned in an A x B array. This will renormalize the arrays so that each subarray sums to 1. If the sum of a subarray is 0, then the resulting JSD will be NaN.

- `auprc_score(true_vals, pred_vals)`
Computes the auPRC in the last dimension of `arr1` and `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are both A x B x L arrays, then the auPRC of corresponding L-arrays will be computed and returned in an A x B array. `true_vals` should contain binary values; any values other than 0 or 1 will be ignored when computing auPRC. `pred_vals` should contain prediction values in the range [0, 1]. The behavior of this function is meant to match `sklearn.metrics.average_precision_score` in its calculation with regards to thresholding. If there are no true positives, the auPRC returned will be NaN.

- `pearson_corr(arr1, arr2)`
Computes the Pearson correlation in the last dimension of `arr1` and `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are both A x B x L arrays, then the correlation of corresponding L-arrays will be computed and returned in an A x B array.

- `spearman_corr(arr1, arr2)`
Computes the Spearman correlation in the last dimension of `arr1` and `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are both A x B x L arrays, then the correlation of corresponding L-arrays will be computed and returned in an A x B array. The Spearman correlation is computed by taking the Pearson correlation of the ranks of the input. For ties, the average rank is used (this is the behavior of `scipy.stats.spearmanr`).

### Memory warning
In order to vectorize some of these functions, there is a larger memory footprint (cest la vie). If you need to compute these metrics over a very large number of data points, it may be needed to batch your calls to these functions.

Batches that are too small won't take advantage of vectorization. Batches that are too large will take up extra time due to swapping/paging. There is a sweet spot to aim for.

### Tests and benchmarks
There are tests for the correctness of these functions (compared to their non-vectorized counterparts in NumPy/SciPy/Scikit Learn), and benchmarks for the speed boost achieved.

On my machine, I got the following times using the benchmarking script, without tuning for batch size.

Jensen-Shannon distance

        Time to compute (SciPy): 120ms
        Time to compute (vec): 107ms

auPRC

        Time to compute (SKLearn): 887ms
        Time to compute (vec): 272ms

Pearson correlation

        Time to compute (SciPy): 85ms
        Time to compute (vec): 13ms

Spearman correlation

        Time to compute (SciPy): 427ms
        Time to compute (vec): 130ms
