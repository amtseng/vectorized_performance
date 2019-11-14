import numpy as np

def _kl_divergence(probs1, probs2):
    """
    Computes the KL divergence in the last dimension of `probs1` and `probs2`
    as KL(P1 || P2). `probs1` and `probs2` must be the same shape. For example,
    if they are both A x B x L arrays, then the KL divergence of corresponding
    L-arrays will be computed and returned in an A x B array. Does not
    renormalize the arrays. If probs2[i] is 0, that value contributes 0.
    """
    quot = np.divide(
        probs1, probs2, out=np.ones_like(probs1),
        where=((probs1 != 0) & (probs2 != 0))
        # No contribution if P1 = 0 or P2 = 0
    )
    return np.sum(probs1 * np.log(quot), axis=-1)


def jensen_shannon_distance(probs1, probs2):
    """
    Computes the Jesnsen-Shannon distance in the last dimension of `probs1` and
    `probs2`. `probs1` and `probs2` must be the same shape. For example, if they
    are both A x B x L arrays, then the KL divergence of corresponding L-arrays
    will be computed and returned in an A x B array. This will renormalize the
    arrays so that each subarray sums to 1. If the sum of a subarray is 0, then
    the resulting JSD will be NaN.
    """
    # Renormalize both distributions, and if the sum is NaN, put NaNs all around
    probs1_sum = np.sum(probs1, axis=-1, keepdims=True)
    probs1 = np.divide(
        probs1, probs1_sum, out=np.full_like(probs1, np.nan),
        where=(probs1_sum != 0)
    )
    probs2_sum = np.sum(probs2, axis=-1, keepdims=True)
    probs2 = np.divide(
        probs2, probs2_sum, out=np.full_like(probs2, np.nan),
        where=(probs2_sum != 0)
    )

    mid = 0.5 * (probs1 + probs2)
    return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))


def auprc_score(true_vals, pred_vals):
    """
    Computes the auPRC in the last dimension of `arr1` and `arr2`. `arr1` and
    `arr2` must be the same shape. For example, if they are both A x B x L
    arrays, then the auPRC of corresponding L-arrays will be computed and
    returned in an A x B array. `true_vals` should contain binary values; any
    values other than 0 or 1 will be ignored when computing auPRC. `pred_vals`
    should contain prediction values in the range [0, 1]. The behavior of this
    function is meant to match `sklearn.metrics.average_precision_score` in its
    calculation with regards to thresholding. If there are no true positives,
    the auPRC returned will be NaN.
    """
    # Sort true and predicted values in descending order
    sorted_inds = np.flip(np.argsort(pred_vals, axis=-1), axis=-1)
    pred_vals = np.take_along_axis(pred_vals, sorted_inds, -1)
    true_vals = np.take_along_axis(true_vals, sorted_inds, -1)

    # Compute the indices where a run of identical predicted values stops
    # In `thresh_inds`, there is a 1 wherever a run ends, and 0 otherwise
    diff = np.diff(pred_vals, axis=-1)
    diff[diff != 0] = 1  # Assign 1 to every nonzero diff
    thresh_inds = np.pad(
        diff, ([(0, 0)] * (diff.ndim - 1)) + [(0, 1)], constant_values=1
    ).astype(int)
    thresh_mask = thresh_inds == 1

    # Compute true positives and false positives at each location; this will
    # eventually be subsetted to only the threshold indices
    # Assign a weight of zero wherever the true value is not binary
    weight_mask = (true_vals == 0) | (true_vals == 1)
    true_pos = np.cumsum(true_vals * weight_mask, axis=-1)
    false_pos = np.cumsum((1 - true_vals) * weight_mask, axis=-1)

    # Compute precision array, but keep 0s wherever there isn't a threshold
    # index
    precis_denom = true_pos + false_pos
    precis = np.divide(
        true_pos, precis_denom,
        out=np.zeros(true_pos.shape),
        where=((precis_denom != 0) & thresh_mask)
    )

    # Compute recall array, but if there are no true positives, it's nan for the
    # entire subarray
    recall_denom = true_pos[..., -1:]
    recall = np.divide(
        true_pos, recall_denom,
        out=np.full(true_pos.shape, np.nan),
        where=(recall_denom != 0)
    )

    # Concatenate an initial value of 0 for recall; adjust `thresh_inds`, too
    thresh_inds = np.pad(
        thresh_inds, ([(0, 0)] * (thresh_inds.ndim - 1)) + [(1, 0)],
        constant_values=1
    )
    recall = np.pad(
        recall, ([(0, 0)] * (recall.ndim - 1)) + [(1, 0)], constant_values=0
    )
    # Concatenate an initial value of 1 for precision; technically, this initial
    # value won't be used for auPRC calculation, but it will be easier for later
    # steps to do this anyway
    precis = np.pad(
        precis, ([(0, 0)] * (precis.ndim - 1)) + [(1, 0)], constant_values=1
    )

    # We want the difference of the recalls, but only in buckets marked by
    # threshold indices; since the number of buckets can be different for each
    # subarray, we create a set of bucketed recalls and precisions for each
    # Each entry in `thresh_buckets` is an index mapping the thresholds to
    # consecutive buckets
    thresh_buckets = np.cumsum(thresh_inds, axis=-1) - 1
    # Set unused buckets to -1; won't happen if there are no unused buckets
    thresh_buckets[thresh_inds == 0] = -1
    # Place the recall values into the buckets into consecutive locations; any
    # unused recall values get placed (and may clobber) the last index
    recall_buckets = np.zeros_like(recall)
    np.put_along_axis(recall_buckets, thresh_buckets, recall, -1)
    # Do the same for precision
    precis_buckets = np.zeros_like(precis)
    np.put_along_axis(precis_buckets, thresh_buckets, precis, -1)

    # Compute the auPRC/average precision by computing the recall bucket diffs
    # and weighting by bucketed precision; note that when `precis` was made,
    # it is 0 wherever there is no threshold index, so all locations in
    # `precis_buckets` which aren't used (even the last index) have a 0
    recall_diffs = np.diff(recall_buckets, axis=-1)
    return np.sum(recall_diffs * precis_buckets[..., 1:], axis=-1)


def pearson_corr(arr1, arr2):
    """
    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.
    """
    mean1 = np.mean(arr1, axis=-1, keepdims=True)
    mean2 = np.mean(arr2, axis=-1, keepdims=True)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(
        numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
    )


def average_ranks(arr):
    """
    Computes the ranks of the elemtns of the given array along the last
    dimension. For ties, the ranks are _averaged_.
    Returns an array of the same dimension of `arr`. 
    """
    # 1) Generate the ranks for each subarray, with ties broken arbitrarily
    sorted_inds = np.argsort(arr, axis=-1)  # Sorted indices
    ranks, ranges = np.empty_like(arr), np.empty_like(arr)
    ranges = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))
    # Put ranks by sorted indices; this creates an array containing the ranks of
    # the elements in each subarray of `arr`
    np.put_along_axis(ranks, sorted_inds, ranges, -1)
    ranks = ranks.astype(int)

    # 2) Create an array where each entry maps a UNIQUE element in `arr` to a
    # unique index for that subarray
    sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
    diffs = np.diff(sorted_arr, axis=-1)
    del sorted_arr  # Garbage collect
    # Pad with an extra zero at the beginning of every subarray
    pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
    del diffs  # Garbage collect
    # Wherever the diff is not 0, assign a value of 1; this gives a set of
    # small indices for each set of unique values in the sorted array after
    # taking a cumulative sum
    pad_diffs[pad_diffs != 0] = 1
    unique_inds = np.cumsum(pad_diffs, axis=-1).astype(int)
    del pad_diffs  # Garbage collect

    # 3) Average the ranks wherever the entries of the `arr` were identical
    # `unique_inds` contains elements that are indices to an array that stores
    # the average of the ranks of each unique element in the original array
    unique_maxes = np.zeros_like(arr)  # Maximum ranks for each unique index
    # Each subarray will contain unused entries if there are no repeats in that
    # subarray; this is a sacrifice made for vectorization; c'est la vie
    # Using `put_along_axis` will put the _last_ thing seen in `ranges`, which
    # result in putting the maximum rank in each unique location
    np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
    # We can compute the average rank for each bucket (from the maximum rank for
    # each bucket) using some algebraic manipulation
    diff = np.diff(unique_maxes, prepend=-1, axis=-1)  # Note: prepend -1!
    unique_avgs = unique_maxes - ((diff - 1) / 2)
    del unique_maxes, diff  # Garbage collect

    # 4) Using the averaged ranks in `unique_avgs`, fill them into where they
    # belong
    avg_ranks = np.take_along_axis(
        unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1
    )

    return avg_ranks


def spearman_corr(arr1, arr2):
    """
    Computes the Spearman correlation in the last dimension of `arr1` and
    `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are
    both A x B x L arrays, then the correlation of corresponding L-arrays will
    be computed and returned in an A x B array. The Spearman correlation is
    computed by taking the Pearson correlation of the ranks of the input. For
    ties, the average rank is used (this is the behavior of
    `scipy.stats.spearmanr`).
    """
    ranks1, ranks2 = average_ranks(arr1), average_ranks(arr2)
    return pearson_corr(ranks1, ranks2)
