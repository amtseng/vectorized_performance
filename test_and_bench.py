import numpy as np
import sklearn.metrics
import scipy.stats
import scipy.spatial.distance
import vec_metrics
from datetime import datetime
import warnings


def test_vectorized_jsd():
    np.random.seed(20191110)
    num_vecs, vec_len = 1000, 1000
    input_size = (num_vecs, vec_len)
    arr1 = np.random.random(input_size)
    arr2 = np.random.random(input_size)
    # Make some rows 0
    arr1[-1] = 0
    arr2[-1] = 0
    arr1[-2] = 0
    arr2[-3] = 0

    print("Testing JSD...")
    jsd_scipy = np.empty(num_vecs)
    a = datetime.now()
    for i in range(num_vecs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ignore warnings when computing JSD with scipy, to avoid
            # warnings when there are no true positives
            jsd_scipy[i] = scipy.spatial.distance.jensenshannon(
                arr1[i], arr2[i]
            )
    jsd_scipy = np.square(jsd_scipy)
    b = datetime.now()
    print("\tTime to compute (SciPy): %dms" % ((b - a).microseconds / 1000))

    a = datetime.now()
    jsd_vec = vec_metrics.jensen_shannon_distance(arr1, arr2)
    b = datetime.now()
    print("\tTime to compute (vec): %dms" % ((b - a).microseconds / 1000))

    jsd_scipy = np.nan_to_num(jsd_scipy)
    jsd_vec = np.nan_to_num(jsd_vec)

    print("\tSame result? %s" % np.allclose(jsd_scipy, jsd_vec))


def test_vectorized_corr():
    np.random.seed(20191110)
    num_corrs, corr_len = 500, 1000
    arr1 = np.random.randint(100, size=(num_corrs, corr_len))
    arr2 = np.random.randint(100, size=(num_corrs, corr_len))

    print("Testing Pearson correlation...")
    pears_scipy = np.empty(num_corrs)
    a = datetime.now()
    for i in range(num_corrs):
        pears_scipy[i] = scipy.stats.pearsonr(arr1[i], arr2[i])[0]
    b = datetime.now()
    print("\tTime to compute (SciPy): %dms" % ((b - a).microseconds / 1000))

    a = datetime.now()
    pears_vect = vec_metrics.pearson_corr(arr1, arr2) 
    b = datetime.now()
    print("\tTime to compute (vec): %dms" % ((b - a).microseconds / 1000))
    print("\tSame result? %s" % np.allclose(pears_vect, pears_scipy))

    print("Testing Spearman correlation...")
    spear_scipy = np.empty(num_corrs)
    a = datetime.now()
    for i in range(num_corrs):
        spear_scipy[i] = scipy.stats.spearmanr(arr1[i], arr2[i])[0]
    b = datetime.now()
    print("\tTime to compute (SciPy): %dms" % ((b - a).microseconds / 1000))

    a = datetime.now()
    spear_vect = vec_metrics.spearman_corr(arr1, arr2) 
    b = datetime.now()
    print("\tTime to compute (vec): %dms" % ((b - a).microseconds / 1000))
    print("\tSame result? %s" % np.allclose(spear_vect, spear_scipy))


def test_vectorized_auprc():
    np.random.seed(20191110)
    num_vecs, vec_len = 500, 1000
    input_size = (num_vecs, vec_len)
    true_vals = []
    pred_vals = np.random.randint(5, size=input_size) / 10
    pred_vals = np.concatenate([pred_vals] * 4)
    # Normal inputs
    true_vals.append(np.random.randint(2, size=input_size))
    # Include some -1
    true_vals.append(np.random.randint(2, size=input_size))
    rand_mask = np.random.randint(2, size=input_size).astype(bool)
    true_vals[1][rand_mask] = -1
    # All positives
    true_vals.append(np.ones_like(true_vals[0]))
    # All negatives
    true_vals.append(np.zeros_like(true_vals[0]))
    true_vals = np.concatenate(true_vals)

    print("Testing auPRC...")
    auprc_scipy = np.empty(pred_vals.shape[0])
    a = datetime.now()
    for i in range(pred_vals.shape[0]):
        t, p = true_vals[i], pred_vals[i]
        mask = (t == 0) | (t == 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ignore warnings when computing auPRC with sklearn, to avoid
            # warnings when there are no true positives
            auprc_scipy[i] = sklearn.metrics.average_precision_score(
                t[mask], p[mask]
            )
    b = datetime.now()
    print("\tTime to compute (SKLearn): %dms" % ((b - a).microseconds / 1000))

    a = datetime.now()
    auprc_vec = vec_metrics.auprc_score(true_vals, pred_vals)
    b = datetime.now()
    print("\tTime to compute (vec): %dms" % ((b - a).microseconds / 1000))

    auprc_scipy = np.nan_to_num(auprc_scipy)
    auprc_vec = np.nan_to_num(auprc_vec)

    print("\tSame result? %s" % np.allclose(auprc_scipy, auprc_vec))


if __name__ == "__main__":
    test_vectorized_jsd()
    test_vectorized_corr()
    test_vectorized_auprc()
