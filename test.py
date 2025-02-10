import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from drips_complex import DripsComplex

if __name__ == "__main__":
    n = 2500
    ratio_vertices = 0.9
    X = np.random.randn(n, 512)
    V, W = train_test_split(X, train_size=ratio_vertices)
    # print(f"{V.shape = }")
    # print(f"{W.shape = }")
    drc = DripsComplex(
        verbose=True,
    ).fit(
        V,
        W,
        n_threads=-1,
    )
    # print("Success.")
