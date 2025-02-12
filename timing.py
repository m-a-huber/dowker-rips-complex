import functools
import sys
import time

import numpy as np
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


n, dim, n_threads = list(map(int, sys.argv[1:]))
ratio_vertices = 0.9
X = np.random.randn(n, dim)
V, W = train_test_split(X, train_size=ratio_vertices)

drc = DripsComplex(verbose=True)
with Timer():
    drc.fit_transform([V, W], n_threads=n_threads)
