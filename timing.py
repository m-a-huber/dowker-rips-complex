import functools
import time

import numpy as np
from numba import jit, prange  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from drips_complex import DripsComplex  # type: ignore
from timer import Timer  # type: ignore


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


def get_ripser_input(
    V,
    W,
):
    @jit(nopython=True, parallel=True)
    def _ripser_input_numba(dm):
        n = dm.shape[0]
        ripser_input = np.empty((n, n), dtype=np.float32)
        for i in prange(n):
            for j in range(i, n):
                dist = np.min(np.maximum(dm[i], dm[j]))
                ripser_input[i, j] = dist
                ripser_input[j, i] = dist
        return ripser_input

    return _ripser_input_numba(
        pairwise_distances(
            X=V,
            Y=W,
            metric="euclidean",
        )
    )


n, dim = 5000, 512
ratio_vertices = 0.9
X = np.random.randn(n, dim)
V, W = train_test_split(X, train_size=ratio_vertices)

# with Timer():
#     get_ripser_input(V, W)

drc = DripsComplex(verbose=True)
with Timer():
    drc.fit_transform([V, W], n_threads=1)
with Timer():
    drc.fit_transform([V, W], n_threads=-1)
